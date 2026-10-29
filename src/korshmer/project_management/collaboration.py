import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any


class CollaborationManager:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.session_active = False
        self.current_session_id = None
        self.participants = []
        logging.info(
            "Collaboration manager initialized (stub - WebSocket server pending)"
        )

    def start_session(
        self, project_path: Path, session_name: str
    ) -> str | None:
        logging.warning("Real-time collaboration not yet implemented")
        return None

    def join_session(self, session_id: str, _user_name: str) -> bool:
        logging.warning("Session join not yet implemented")
        return False

    def leave_session(self) -> bool:
        self.session_active = False
        self.current_session_id = None
        return True

    def send_edit(self, _edit_data: dict[str, Any]) -> bool:
        logging.warning("Edit broadcasting not yet implemented")
        return False

    def register_edit_callback(
        self, callback: Callable[[dict[str, Any]], None]
    ) -> bool:
        logging.warning("Edit callback registration not yet implemented")
        return False

    def send_chat_message(self, message: str) -> bool:
        logging.warning("Chat not yet implemented")
        return False

    def register_chat_callback(
        self, callback: Callable[[str, str], None]
    ) -> bool:
        logging.warning("Chat callback registration not yet implemented")
        return False

    def get_participants(self) -> list[dict[str, Any]]:
        return self.participants

    def lock_resource(self, _resource_id: str) -> bool:
        logging.warning("Resource locking not yet implemented")
        return False

    def unlock_resource(self, _resource_id: str) -> bool:
        logging.warning("Resource unlocking not yet implemented")
        return False

    def get_session_status(self) -> dict[str, Any]:
        return {
            "active": self.session_active,
            "session_id": self.current_session_id,
            "participants": self.participants,
            "locked_resources": [],
        }
