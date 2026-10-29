from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from korshmer.utils.secure_random import new_id

from .operations import Operation

try:
    import websockets

    _HAS_WEBSOCKETS = True
except ImportError:
    _HAS_WEBSOCKETS = False


class CollaborationClient:
    def __init__(self, server_url: str, session_id: str, peer_id: str = None):
        if not _HAS_WEBSOCKETS:
            raise ImportError(
                "websockets library required for collaboration. "
                "Install with: pip install websockets"
            )
        self.server_url = server_url
        self.session_id = session_id
        self.peer_id = peer_id or new_id()
        self.websocket: Any | None = None
        self.on_operation_callback: Callable[[Operation], None] | None = None
        self.logger = logging.getLogger(__name__)

    async def connect(self):
        try:
            self.websocket = await websockets.connect(self.server_url)
            asyncio.create_task(self._receive_loop())
            self.logger.info(
                f"Connected to collaboration server: {self.server_url}"
            )
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            raise

    async def disconnect(self):
        if self.websocket:
            await self.websocket.close()
            self.logger.info("Disconnected from collaboration server")

    async def send_operation(self, op: Operation):
        if not self.websocket:
            self.logger.error("Not connected to server")
            return
        try:
            await self.websocket.send(op.to_json())
        except Exception as e:
            self.logger.error(f"Failed to send operation: {e}")

    def on_operation(self, callback: Callable[[Operation], None]):
        self.on_operation_callback = callback

    async def _receive_loop(self):
        try:
            if not self.websocket:
                return
            async for message in self.websocket:
                try:
                    operation = Operation.from_json(message)
                    if self.on_operation_callback:
                        self.on_operation_callback(operation)
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
        except Exception as e:
            self.logger.error(f"Receive loop error: {e}")
