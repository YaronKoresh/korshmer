import asyncio
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from korshmer.collaboration.operations import Operation, OpType
from korshmer.collaboration.server import (
    DEFAULT_HOST,
    SERVER_ONLY_OP_TYPES,
    CollaborationServer,
)

_logger = logging.getLogger(__name__)

HOST_PEER_ID = "host"
SERVER_TIMEOUT_S = 5.0
EDIT_OP_TYPES = frozenset(op.value for op in OpType) - SERVER_ONLY_OP_TYPES
UNAVAILABLE_FEATURES = (
    "join_session",
    "edit_callbacks",
    "chat",
    "resource_locks",
)


class CollaborationManager:
    def __init__(
        self,
        base_dir: Path,
        host: str = DEFAULT_HOST,
        port: int = 0,
        token: str | None = None,
    ):
        self.base_dir = Path(base_dir)
        self.host = host
        self.port = port
        self.token = token
        self.session_active = False
        self.current_session_id: str | None = None
        self.session_name: str | None = None
        self._lock = threading.RLock()
        self._server: CollaborationServer | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def _run(self, coroutine) -> Any:
        if self._loop is None:
            coroutine.close()
            raise RuntimeError("The collaboration server is not running")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result(SERVER_TIMEOUT_S)

    def _stop_loop(self) -> None:
        loop, thread = self._loop, self._thread
        self._loop = self._thread = self._server = None
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(SERVER_TIMEOUT_S)
        if loop is not None and not loop.is_running():
            loop.close()

    def start_session(
        self, project_path: Path, session_name: str
    ) -> str | None:
        with self._lock:
            if self.session_active and self._server is not None:
                return self._server.get_session_url(
                    str(self.current_session_id)
                )
            try:
                server = CollaborationServer(
                    host=self.host, port=self.port, token=self.token
                )
            except ValueError as e:
                _logger.error(f"Collaboration session not started: {e}")
                return None
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever,
                name="korshmer-collaboration",
                daemon=True,
            )
            self._server, self._loop, self._thread = server, loop, thread
            thread.start()
            try:
                self._run(server.start())
            except Exception as e:
                _logger.error(f"Collaboration session not started: {e}")
                self._stop_loop()
                return None
            self.current_session_id = server.create_session()
            self.session_name = str(session_name)
            self.session_active = True
            _logger.info(
                f"Collaboration session '{session_name}' for {project_path} "
                f"is listening on {server.host}:{server.port}"
            )
            return server.get_session_url(self.current_session_id)

    def leave_session(self) -> bool:
        with self._lock:
            server = self._server
            if server is not None:
                try:
                    self._run(server.stop())
                except Exception as e:
                    _logger.error(f"Collaboration server did not stop: {e}")
                self._stop_loop()
            self.session_active = False
            self.current_session_id = None
            self.session_name = None
            return True

    def send_edit(self, edit_data: dict[str, Any]) -> bool:
        with self._lock:
            server, session_id = self._server, self.current_session_id
            if not self.session_active or server is None or not session_id:
                return False
            data = dict(edit_data)
            op_type = data.pop("op_type", OpType.SYNC_STATE.value)
            if op_type not in EDIT_OP_TYPES:
                _logger.error(f"Edit of unknown kind not sent: {op_type!r}")
                return False
            operation = Operation(
                op_type=op_type,
                session_id=session_id,
                peer_id=HOST_PEER_ID,
                data=data,
            )
            try:
                self._run(server.broadcast(session_id, operation))
            except Exception as e:
                _logger.error(f"Edit not sent: {e}")
                return False
            return True

    def _unavailable(self, feature: str) -> bool:
        _logger.info(
            f"Collaboration feature not available: {feature}. This version "
            "hosts a session and sends edits to the musicians who joined."
        )
        return False

    def join_session(self, _session_id: str, _user_name: str) -> bool:
        return self._unavailable("join_session")

    def register_edit_callback(
        self, _callback: Callable[[dict[str, Any]], None]
    ) -> bool:
        return self._unavailable("edit_callbacks")

    def send_chat_message(self, _message: str) -> bool:
        return self._unavailable("chat")

    def register_chat_callback(
        self, _callback: Callable[[str, str], None]
    ) -> bool:
        return self._unavailable("chat")

    def lock_resource(self, _resource_id: str) -> bool:
        return self._unavailable("resource_locks")

    def unlock_resource(self, _resource_id: str) -> bool:
        return self._unavailable("resource_locks")

    def get_participants(self) -> list[dict[str, Any]]:
        with self._lock:
            server, session_id = self._server, self.current_session_id
            if server is None or not session_id:
                return []
            return [
                {"peer_id": peer_id}
                for peer_id in list(server.peers.get(session_id, {}))
            ]

    def get_session_status(self) -> dict[str, Any]:
        with self._lock:
            server, session_id = self._server, self.current_session_id
            address = url = None
            if self.session_active and server is not None and session_id:
                address = f"{server.host}:{server.port}"
                url = server.get_session_url(session_id)
            return {
                "active": address is not None,
                "session_id": session_id,
                "name": self.session_name,
                "address": address,
                "url": url,
                "token_required": self.token is not None,
                "participants": self.get_participants(),
                "locked_resources": [],
                "unavailable": list(UNAVAILABLE_FEATURES),
            }
