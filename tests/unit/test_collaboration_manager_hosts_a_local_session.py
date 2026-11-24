import asyncio
import socket

import pytest

from korshmer.collaboration.operations import Operation, OpType
from korshmer.project_management.collaboration import CollaborationManager

websockets = pytest.importorskip("websockets")


@pytest.fixture
def manager(tmp_path):
    m = CollaborationManager(tmp_path)
    yield m
    m.leave_session()


def port_of(status):
    return int(str(status.get("address")).rsplit(":", 1)[1])


def port_is_open(port):
    with socket.socket() as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_starting_a_session_opens_a_server_on_this_computer(manager, tmp_path):
    link = manager.start_session(tmp_path / "song.kproj", "Jam")
    status = manager.get_session_status()
    assert status["active"] is True
    assert status.get("name") == "Jam"
    assert str(status.get("address")).startswith("127.0.0.1:")
    assert link == status.get("url")
    assert status["session_id"] in link
    assert port_is_open(port_of(status))


def test_nothing_is_shared_before_a_session_starts(manager):
    status = manager.get_session_status()
    assert status["active"] is False
    assert status.get("address", "missing") is None
    assert status["participants"] == []


def test_leaving_the_session_closes_the_server(manager, tmp_path):
    assert manager.start_session(tmp_path / "song.kproj", "Jam")
    port = port_of(manager.get_session_status())
    assert manager.leave_session() is True
    assert manager.get_session_status()["active"] is False
    assert not port_is_open(port)


def test_starting_twice_keeps_the_running_session(manager, tmp_path):
    first = manager.start_session(tmp_path / "song.kproj", "Jam")
    assert first is not None
    assert manager.start_session(tmp_path / "song.kproj", "Again") == first


async def join(ws, session_id):
    await ws.send(
        Operation(
            op_type=OpType.CURSOR_MOVE.value,
            session_id=session_id,
            peer_id="guest",
        ).to_json()
    )
    await ws.send("this is not an operation")
    answer = Operation.from_json(await asyncio.wait_for(ws.recv(), 3))
    assert answer.op_type == OpType.ERROR.value


def test_an_edit_reaches_a_musician_who_joined(manager, tmp_path):
    assert manager.start_session(tmp_path / "song.kproj", "Jam")
    status = manager.get_session_status()

    async def guest():
        async with websockets.connect(f"ws://{status['address']}") as ws:
            await join(ws, status["session_id"])
            seen = len(manager.get_participants())
            sent = manager.send_edit(
                {"op_type": OpType.TEMPO_CHANGE.value, "bpm": 123.4}
            )
            received = Operation.from_json(await asyncio.wait_for(ws.recv(), 3))
            return seen, sent, received

    seen, sent, received = run(guest())
    assert (seen, sent) == (1, True)
    assert received.op_type == OpType.TEMPO_CHANGE.value
    assert received.data == {"bpm": 123.4}


def test_an_edit_without_a_session_is_not_sent_as_a_safeguard(manager):
    assert manager.send_edit({"op_type": OpType.NOTE_ADD.value}) is False


def test_an_edit_of_an_unknown_kind_is_not_sent(manager, tmp_path):
    assert manager.start_session(tmp_path / "song.kproj", "Jam")
    assert manager.send_edit({"op_type": "peer_joined"}) is False
    assert manager.send_edit({"op_type": "made_up"}) is False


def test_a_session_shared_on_the_network_asks_guests_for_the_secret(tmp_path):
    try:
        manager = CollaborationManager(tmp_path, host="0.0.0.0", token="s3cret")
    except TypeError:
        manager = CollaborationManager(tmp_path)
    try:
        assert manager.start_session(tmp_path / "song.kproj", "Jam")
        status = manager.get_session_status()
        port = port_of(status)

        async def knock(address):
            async with websockets.connect(address) as ws:
                try:
                    await join(ws, status["session_id"])
                except websockets.exceptions.ConnectionClosed:
                    return "refused"
                return "joined"

        base = f"ws://127.0.0.1:{port}"
        assert run(knock(base)) == "refused"
        assert run(knock(f"{base}/?token=s3cret")) == "joined"
        assert "s3cret" not in str(status)
    finally:
        manager.leave_session()


def test_features_the_server_does_not_have_say_so_plainly(manager, caplog):
    with caplog.at_level("DEBUG"):
        assert manager.join_session("abc", "Dana") is False
        assert manager.send_chat_message("hi") is False
        assert manager.lock_resource("track-1") is False
        assert manager.unlock_resource("track-1") is False
        assert manager.register_edit_callback(lambda edit: None) is False
        assert manager.register_chat_callback(lambda who, text: None) is False
    text = " ".join(r.getMessage() for r in caplog.records).lower()
    assert "not yet implemented" not in text
    assert "not available" in text
    assert set(manager.get_session_status().get("unavailable", ())) >= {
        "join_session",
        "chat",
        "resource_locks",
    }
