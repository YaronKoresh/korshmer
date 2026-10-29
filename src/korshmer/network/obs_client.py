from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import threading
from collections.abc import Callable
from urllib.parse import urlsplit

from korshmer.utils.audit_log import OUTCOME_DENIED, OUTCOME_FAILED, audit
from korshmer.utils.secure_random import new_id

try:
    import websockets

    _HAS_WEBSOCKETS = True
except ImportError:
    _HAS_WEBSOCKETS = False

_logger = logging.getLogger(__name__)

DEFAULT_URL = "ws://127.0.0.1:4455"


SUBPROTOCOL = "obswebsocket.json"


RPC_VERSION = 1

OP_HELLO = 0
OP_IDENTIFY = 1
OP_IDENTIFIED = 2
OP_REQUEST = 6
OP_REQUEST_RESPONSE = 7


MAX_SCENE_NAME = 256


DEFAULT_TIMEOUT = 5.0
MAX_TIMEOUT = 300.0


def _valid_ws_url(url: str) -> bool:
    try:
        parts = urlsplit(url)
        return parts.scheme in ("ws", "wss") and bool(parts.hostname)
    except ValueError:
        return False


def auth_string(password: str, salt: str, challenge: str) -> str:
    secret = base64.b64encode(
        hashlib.sha256((password + salt).encode("utf-8")).digest()
    ).decode("ascii")
    return base64.b64encode(
        hashlib.sha256((secret + challenge).encode("utf-8")).digest()
    ).decode("ascii")


class ObsClient:
    def __init__(
        self,
        url: str = DEFAULT_URL,
        password: str | None = None,
        on_status: Callable[[str], None] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or not 0 < timeout <= MAX_TIMEOUT
        ):
            raise ValueError(
                "OBS timeout must be a number of seconds above 0 and at "
                f"most {MAX_TIMEOUT}, got {timeout!r}"
            )
        self.url = url
        self._password = password or None
        self._on_status = on_status
        self.timeout = timeout
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue | None = None
        self._stop_event: asyncio.Event | None = None
        self._ready = threading.Event()
        self._connected = False
        self._recording = False
        self._lock = threading.Lock()

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def recording(self) -> bool:
        return self._recording

    def _status(self, text: str) -> None:
        _logger.info(text)
        if self._on_status is not None:
            try:
                self._on_status(text)
            except Exception:
                _logger.debug("status callback failed", exc_info=True)

    def start(self) -> bool:
        if not _HAS_WEBSOCKETS:
            self._status("OBS needs the 'websockets' package.")
            return False
        if not isinstance(self.url, str) or not _valid_ws_url(self.url):
            self._status(
                f"OBS address must be a ws:// or wss:// URL, got {self.url!r}."
            )
            return False
        if self._thread is not None and self._thread.is_alive():
            return self._connected
        self._ready.clear()
        self._connected = False
        self._recording = False
        self._thread = threading.Thread(
            target=self._run, name="korshmer-obs", daemon=True
        )
        self._thread.start()
        self._ready.wait(self.timeout + 1.0)
        return self._connected

    def stop(self) -> None:
        loop, event = self._loop, self._stop_event
        if loop is not None and event is not None:
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                pass
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(self.timeout + 1.0)
        self._thread = None

    def start_recording(self) -> bool:
        with self._lock:
            if self._recording:
                return False
            if not self._send("StartRecord"):
                return False
            self._recording = True
            return True

    def stop_recording(self) -> bool:
        with self._lock:
            if not self._recording:
                return False
            if not self._send("StopRecord"):
                return False
            self._recording = False
            return True

    def switch_scene(self, scene_name: str) -> bool:
        if (
            not isinstance(scene_name, str)
            or not scene_name.strip()
            or len(scene_name) > MAX_SCENE_NAME
            or any(ord(c) < 32 or ord(c) == 127 for c in scene_name)
        ):
            return False
        return self._send("SetCurrentProgramScene", {"sceneName": scene_name})

    def _send(
        self, request_type: str, request_data: dict | None = None
    ) -> bool:
        loop, queue = self._loop, self._queue
        if not self._connected or loop is None or queue is None:
            return False
        try:
            loop.call_soon_threadsafe(
                queue.put_nowait, (request_type, request_data)
            )
        except RuntimeError:
            return False
        return True

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as e:
            _logger.error("OBS client stopped: %s", type(e).__name__)
        finally:
            was_connected = self._connected
            self._connected = False
            self._recording = False
            self._loop = self._queue = self._stop_event = None
            self._ready.set()
            if was_connected and not self._stopping:
                self._status("OBS connection lost.")

    _stopping = False

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._stop_event = asyncio.Event()
        self._stopping = False
        try:
            async with websockets.connect(
                self.url,
                subprotocols=[SUBPROTOCOL],
                open_timeout=self.timeout,
            ) as ws:
                await asyncio.wait_for(self._handshake(ws), self.timeout)
                self._connected = True
                self._ready.set()
                self._status("OBS connected.")
                await self._serve(ws)
        except Exception as e:
            if not self._connected:
                audit(
                    "obs.connect",
                    OUTCOME_FAILED,
                    url=self.url,
                    error=type(e).__name__,
                )
                self._status(
                    f"OBS could not be reached at {self.url} "
                    f"({type(e).__name__})."
                )
            else:
                _logger.debug("OBS connection ended: %s", type(e).__name__)

    async def _handshake(self, ws) -> None:
        hello = json.loads(await ws.recv())
        if hello.get("op") != OP_HELLO:
            raise ConnectionError("OBS did not send Hello")
        data = {"rpcVersion": RPC_VERSION}
        auth = hello.get("d", {}).get("authentication")
        if auth:
            if not self._password:
                audit(
                    "obs.login",
                    OUTCOME_DENIED,
                    reason="password_missing",
                    url=self.url,
                )
                raise PermissionError("OBS needs a password")
            data["authentication"] = auth_string(
                self._password, auth["salt"], auth["challenge"]
            )
        await ws.send(json.dumps({"op": OP_IDENTIFY, "d": data}))
        reply = json.loads(await ws.recv())
        if reply.get("op") != OP_IDENTIFIED:
            audit(
                "obs.login",
                OUTCOME_DENIED,
                reason="identification_refused",
                url=self.url,
            )
            raise PermissionError("OBS refused the identification")
        audit("obs.login", url=self.url, authenticated=bool(auth))

    async def _serve(self, ws) -> None:
        sender = asyncio.ensure_future(self._send_loop(ws))
        receiver = asyncio.ensure_future(self._receive_loop(ws))
        stopper = asyncio.ensure_future(self._stop_event.wait())
        done, pending = await asyncio.wait(
            {sender, receiver, stopper}, return_when=asyncio.FIRST_COMPLETED
        )
        self._stopping = stopper in done
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    async def _send_loop(self, ws) -> None:
        while True:
            request_type, request_data = await self._queue.get()
            body = {
                "requestType": request_type,
                "requestId": new_id(),
            }
            if request_data:
                body["requestData"] = request_data
            await ws.send(json.dumps({"op": OP_REQUEST, "d": body}))

    async def _receive_loop(self, ws) -> None:
        async for raw in ws:
            try:
                message = json.loads(raw)
            except ValueError:
                continue
            if message.get("op") != OP_REQUEST_RESPONSE:
                continue
            data = message.get("d", {})
            status = data.get("requestStatus", {})
            if not status.get("result", False):
                if data.get("requestType") == "StartRecord":
                    self._recording = False
                self._status(
                    f"OBS {data.get('requestType', 'request')} failed: "
                    f"{status.get('comment', 'no reason given')}"
                )
