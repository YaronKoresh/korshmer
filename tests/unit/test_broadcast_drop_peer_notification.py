from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from korshmer.collaboration.server import CollaborationServer


@pytest.fixture
def server():
    srv = CollaborationServer.__new__(CollaborationServer)
    srv.peers = {}
    srv.sessions = {}
    srv._client_sessions = set()
    srv._rate_limiters = {}
    srv.logger = MagicMock()
    return srv


@pytest.mark.asyncio
async def test_broadcast_sends_peer_left_on_disconnect(server):
    import websockets.exceptions

    from korshmer.collaboration.operations import Operation

    dead_ws = AsyncMock()
    dead_ws.send = AsyncMock(
        side_effect=websockets.exceptions.ConnectionClosed(None, None)
    )
    alive_ws = AsyncMock()

    server.peers["s1"] = {"dead": dead_ws, "alive": alive_ws}
    server._client_sessions.add("s1")

    op = Operation(op_type="edit", session_id="s1", peer_id="alive", data={})
    await server.broadcast("s1", op)

    sent_messages = [call.args[0] for call in alive_ws.send.call_args_list]
    peer_left_found = any('"peer_left"' in m for m in sent_messages)
    assert peer_left_found, (
        "Expected a PEER_LEFT message to be sent to remaining peers"
    )
    assert "dead" not in server.peers.get("s1", {})
