from __future__ import annotations

import hmac
import ipaddress
import logging
from collections.abc import AsyncIterator
from typing import Any, Protocol
from urllib.parse import parse_qs, urlsplit

from korshmer.utils.audit_log import OUTCOME_DENIED, audit
from korshmer.utils.secure_random import new_id

from .operations import Operation, OpType
from .rate_limit import TokenBucket

try:
    import websockets

    _HAS_WEBSOCKETS = True
except ImportError:
    _HAS_WEBSOCKETS = False


class _PeerConnection(Protocol):
    def __aiter__(self) -> AsyncIterator[str | bytes]: ...


SERVER_ONLY_OP_TYPES = frozenset(
    {
        OpType.PEER_JOINED.value,
        OpType.PEER_LEFT.value,
        OpType.ERROR.value,
    }
)


DEFAULT_HOST = "127.0.0.1"
TOKEN_QUERY_FIELD = "token"
POLICY_VIOLATION = 1008


def is_loopback_host(host: str | None) -> bool:
    name = (host or "").strip().lower()
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name.strip("[]")).is_loopback
    except ValueError:
        return False


def _request_parts(websocket: Any, path: str | None) -> tuple[str, Any]:
    request = getattr(websocket, "request", None)
    target = getattr(request, "path", None) or getattr(websocket, "path", path)
    headers = getattr(request, "headers", None) or getattr(
        websocket, "request_headers", None
    )
    return str(target or ""), headers or {}


def _presented_token(target: str, headers: Any) -> str:
    values = parse_qs(urlsplit(target).query).get(TOKEN_QUERY_FIELD)
    if values:
        return str(values[0])
    authorization = str(headers.get("Authorization") or "")
    scheme, _, value = authorization.partition(" ")
    return value.strip() if scheme.lower() == "bearer" else ""


def _origin_allowed(headers: Any) -> bool:
    origin = headers.get("Origin")
    if origin is None:
        return True
    try:
        return is_loopback_host(urlsplit(str(origin)).hostname)
    except ValueError:
        return False


class CollaborationServer:
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = 8765,
        max_messages_per_second: float = 50.0,
        burst: int = 100,
        max_violations: int = 50,
        max_sessions: int = 1000,
        max_peers_per_session: int = 64,
        max_operations_per_session: int = 1000,
        max_message_bytes: int = 1024 * 1024,
        token: str | None = None,
    ):
        if token is not None and not (isinstance(token, str) and token):
            raise ValueError("token must be a non-empty string")
        if token is None and not is_loopback_host(host):
            raise ValueError(
                f"A shared secret token is required to listen on {host!r}; "
                "only 127.0.0.1, localhost and ::1 work without one"
            )
        for name, value in (
            ("max_sessions", max_sessions),
            ("max_peers_per_session", max_peers_per_session),
            ("max_operations_per_session", max_operations_per_session),
            ("max_message_bytes", max_message_bytes),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        TokenBucket(max_messages_per_second, burst)
        self.host = host
        self.port = port
        self.token = token
        self.max_messages_per_second = max_messages_per_second
        self.burst = burst
        self.max_violations = max_violations
        self.max_sessions = max_sessions
        self.max_peers_per_session = max_peers_per_session
        self.max_operations_per_session = max_operations_per_session
        self.max_message_bytes = max_message_bytes
        self._client_sessions: set[str] = set()
        self.sessions: dict[str, dict[str, Any]] = {}

        self.peers: dict[str, dict[str, Any]] = {}
        self.server: Any | None = None
        self.logger = logging.getLogger(__name__)

    async def start(self):
        if not _HAS_WEBSOCKETS:
            raise ImportError(
                "websockets library required for collaboration. "
                "Install with: pip install websockets"
            )
        self.server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            max_size=self.max_message_bytes,
        )
        sockets = getattr(self.server, "sockets", None) or ()
        if sockets:
            self.port = sockets[0].getsockname()[1]
        self.logger.info(
            f"Collaboration server started at {self.host}:{self.port}"
        )

    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.logger.info("Collaboration server stopped")

    def create_session(self, session_id: str | None = None) -> str:
        if session_id is None:
            session_id = new_id()
        self.sessions[session_id] = {
            "created": None,
            "peers": set(),
            "operations": [],
        }
        self.peers[session_id] = {}
        return session_id

    def get_session_url(self, session_id: str) -> str:
        return f"korshmer://session/{session_id}@{self.host}:{self.port}"

    async def _handle_client(
        self, websocket: _PeerConnection, path: str | None = None
    ):
        refusal = self._refusal(websocket, path)
        if refusal:
            self.logger.warning(f"Refused a collaboration peer: {refusal}")
            audit("collaboration.connect", OUTCOME_DENIED, reason=refusal)
            await self._close(websocket, refusal)
            return
        peer_id = new_id()
        joined: list[str] = []
        bucket = TokenBucket(self.max_messages_per_second, self.burst)
        violations = 0

        try:
            async for message in websocket:
                if not bucket.allow():
                    violations += 1
                    if violations >= self.max_violations:
                        self.logger.warning(
                            f"Closing peer {peer_id}: rate limit exceeded"
                        )
                        await self._drop_peer_everywhere(joined, peer_id)
                        await self._close(websocket)
                        return
                    await self._send_error(websocket, "Rate limit exceeded")
                    continue
                if len(message) > self.max_message_bytes:
                    await self._send_error(websocket, "Message too large")
                    continue
                try:
                    operation = Operation.from_json(message)
                    if operation.op_type in SERVER_ONLY_OP_TYPES:
                        raise ValueError(
                            f"Operation type '{operation.op_type}' "
                            "is reserved for the server"
                        )
                    operation.peer_id = peer_id
                    session_id = operation.session_id

                    if session_id not in self.sessions:
                        if len(self.sessions) >= self.max_sessions:
                            await self._send_error(
                                websocket, "Too many sessions"
                            )
                            continue
                        self.create_session(session_id)
                        self._client_sessions.add(session_id)

                    if (
                        peer_id not in self.peers[session_id]
                        and len(self.peers[session_id])
                        >= self.max_peers_per_session
                    ):
                        await self._send_error(websocket, "Session is full")
                        continue

                    if peer_id not in self.peers[session_id]:
                        self.peers[session_id][peer_id] = websocket
                        if session_id not in joined:
                            joined.append(session_id)
                        await self.broadcast(
                            session_id,
                            Operation(
                                op_type=OpType.PEER_JOINED.value,
                                session_id=session_id,
                                peer_id=peer_id,
                            ),
                            exclude_peer=peer_id,
                        )

                    history = self.sessions[session_id]["operations"]
                    history.append(operation)
                    if len(history) > self.max_operations_per_session:
                        del history[
                            : len(history) - self.max_operations_per_session
                        ]
                    await self.broadcast(
                        session_id, operation, exclude_peer=peer_id
                    )

                except ValueError as e:
                    self.logger.error(f"Invalid message: {e}")
                    await self._send_error(websocket, str(e))
                except Exception as e:
                    self.logger.error(f"Error handling message: {e}")

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self._drop_peer_everywhere(joined, peer_id)

    async def _drop_peer_everywhere(self, session_ids: list[str], peer_id: str):
        for session_id in session_ids:
            await self._drop_peer(session_id, peer_id)

    async def _drop_peer(self, session_id: str | None, peer_id: str):
        if session_id and session_id in self.peers:
            if self.peers[session_id].pop(peer_id, None) is None:
                return
            if (
                not self.peers[session_id]
                and session_id in self._client_sessions
            ):
                self._client_sessions.discard(session_id)
                self.sessions.pop(session_id, None)
                self.peers.pop(session_id, None)
                return
            await self.broadcast(
                session_id,
                Operation(
                    op_type=OpType.PEER_LEFT.value,
                    session_id=session_id,
                    peer_id=peer_id,
                ),
                exclude_peer=peer_id,
            )

    def _refusal(self, websocket: Any, path: str | None) -> str:
        target, headers = _request_parts(websocket, path)
        if not _origin_allowed(headers):
            return "Origin not allowed"
        if self.token is not None and not hmac.compare_digest(
            _presented_token(target, headers).encode("utf-8"),
            self.token.encode("utf-8"),
        ):
            return "Unauthorized"
        return ""

    async def _close(self, websocket, reason: str = "Rate limit exceeded"):
        try:
            await websocket.close(code=POLICY_VIOLATION, reason=reason)
        except Exception as e:
            self.logger.error(f"Failed to close peer connection: {e}")

    async def _send_error(self, websocket, reason: str):
        error = Operation(
            op_type=OpType.ERROR.value,
            session_id="server",
            peer_id="server",
            data={"error": reason},
        )
        try:
            await websocket.send(error.to_json())
        except Exception as e:
            self.logger.error(f"Failed to send error to peer: {e}")

    async def broadcast(
        self,
        session_id: str,
        operation: Operation,
        exclude_peer: str | None = None,
    ):
        if session_id not in self.peers:
            return

        message = operation.to_json()
        disconnected = set()

        for pid, websocket in list(self.peers[session_id].items()):
            if exclude_peer is not None and pid == exclude_peer:
                continue
            try:
                await websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(pid)

        for pid in disconnected:
            await self._drop_peer(session_id, pid)
