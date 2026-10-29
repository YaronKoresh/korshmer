from __future__ import annotations

import logging
from typing import Any

from korshmer.utils.secure_random import new_id

from .operations import Operation, OpType
from .rate_limit import TokenBucket

try:
    import websockets
    from websockets.server import WebSocketServerProtocol

    _HAS_WEBSOCKETS = True
except ImportError:
    _HAS_WEBSOCKETS = False


SERVER_ONLY_OP_TYPES = frozenset(
    {
        OpType.PEER_JOINED.value,
        OpType.PEER_LEFT.value,
        OpType.ERROR.value,
    }
)


class CollaborationServer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        max_messages_per_second: float = 50.0,
        burst: int = 100,
        max_violations: int = 50,
        max_sessions: int = 1000,
        max_peers_per_session: int = 64,
        max_operations_per_session: int = 1000,
        max_message_bytes: int = 1024 * 1024,
    ):
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
        self.logger.info(
            f"Collaboration server started at {self.host}:{self.port}"
        )

    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.logger.info("Collaboration server stopped")

    def create_session(self, session_id: str = None) -> str:
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
        self, websocket: WebSocketServerProtocol, path: str | None = None
    ):
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

    async def _close(self, websocket):
        try:
            await websocket.close(code=1008, reason="Rate limit exceeded")
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
        self, session_id: str, operation: Operation, exclude_peer: str = None
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
