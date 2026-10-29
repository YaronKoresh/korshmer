import asyncio
import inspect

import pytest

from korshmer.collaboration.operations import Operation, OpType
from korshmer.collaboration.server import CollaborationServer


class ScriptedWS:
    def __init__(self, count, session_id="s1"):
        self.count = count
        self.session_id = session_id
        self.sent = []

    def __aiter__(self):
        async def gen():
            for _ in range(self.count):
                yield Operation(
                    op_type=OpType.CURSOR_MOVE.value,
                    session_id=self.session_id,
                    peer_id="x",
                ).to_json()

        return gen()

    async def send(self, message):
        self.sent.append(message)

    async def close(self, code=1000, reason=""):
        pass


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_handler_accepts_single_connection_argument():
    s = CollaborationServer()
    sig = inspect.signature(s._handle_client)
    sig.bind(object())


def test_handler_runs_with_one_argument():
    s = CollaborationServer()
    sid = s.create_session("s1")
    _run(s._handle_client(ScriptedWS(2)))
    assert len(s.sessions[sid]["operations"]) == 2


def test_peer_removed_and_peer_left_sent_after_normal_close():
    s = CollaborationServer()
    sid = s.create_session("s1")
    other = ScriptedWS(0)
    s.peers[sid]["B"] = other
    _run(s._handle_client(ScriptedWS(3), "/"))
    assert list(s.peers[sid]) == ["B"]
    left = [
        m
        for m in other.sent
        if Operation.from_json(m).op_type == OpType.PEER_LEFT.value
    ]
    assert len(left) == 1


def test_drop_peer_twice_sends_one_peer_left():
    s = CollaborationServer()
    sid = s.create_session("s1")
    other = ScriptedWS(0)
    s.peers[sid]["A"] = ScriptedWS(0)
    s.peers[sid]["B"] = other
    _run(s._drop_peer(sid, "A"))
    _run(s._drop_peer(sid, "A"))
    assert len(other.sent) == 1


def test_real_websocket_client_connects_and_leave_is_announced():
    websockets = pytest.importorskip("websockets")

    async def scenario():
        s = CollaborationServer(host="127.0.0.1", port=0)
        await s.start()
        port = s.server.sockets[0].getsockname()[1]
        uri = f"ws://127.0.0.1:{port}"

        def msg(peer):
            return Operation(
                op_type=OpType.CURSOR_MOVE.value,
                session_id="s1",
                peer_id=peer,
            ).to_json()

        a = await websockets.connect(uri)
        await a.send(msg("a"))
        await asyncio.sleep(0.1)
        b = await websockets.connect(uri)
        await b.send(msg("b"))
        await asyncio.sleep(0.1)
        await b.close()
        seen = []
        try:
            while True:
                m = await asyncio.wait_for(a.recv(), 0.5)
                seen.append(Operation.from_json(m).op_type)
        except (asyncio.TimeoutError, TimeoutError):
            pass
        remaining = len(s.peers["s1"])
        await a.close()
        await s.stop()
        return seen, remaining

    seen, remaining = _run(scenario())
    assert OpType.PEER_LEFT.value in seen
    assert remaining == 1


class SwitchingWS:
    def __init__(self, session_ids):
        self.session_ids = session_ids
        self.sent = []

    def __aiter__(self):
        async def gen():
            for sid in self.session_ids:
                yield Operation(
                    op_type=OpType.CURSOR_MOVE.value,
                    session_id=sid,
                    peer_id="x",
                ).to_json()

        return gen()

    async def send(self, message):
        self.sent.append(message)

    async def close(self, code=1000, reason=""):
        pass


def test_peer_removed_from_every_session_it_joined():
    s = CollaborationServer()
    watchers = {}
    for sid in ("A", "B"):
        s.create_session(sid)
        watchers[sid] = ScriptedWS(0)
        s.peers[sid]["W"] = watchers[sid]
    _run(s._handle_client(SwitchingWS(["A", "B"])))
    assert list(s.peers["A"]) == ["W"]
    assert list(s.peers["B"]) == ["W"]
    for ws in watchers.values():
        left = [
            m
            for m in ws.sent
            if Operation.from_json(m).op_type == OpType.PEER_LEFT.value
        ]
        assert len(left) == 1


class RecordingWS(ScriptedWS):
    def __init__(self, ops):
        super().__init__(0)
        self.ops = ops

    def __aiter__(self):
        async def gen():
            for op in self.ops:
                yield op.to_json()

        return gen()


def _types(ws):
    return [Operation.from_json(m).op_type for m in ws.sent]


def test_relayed_operation_carries_the_server_assigned_peer_id():
    s = CollaborationServer()
    sid = s.create_session("s1")
    other = ScriptedWS(0)
    s.peers[sid]["B"] = other
    forged = Operation(
        op_type=OpType.NOTE_ADD.value, session_id=sid, peer_id="B"
    )
    _run(s._handle_client(RecordingWS([forged])))
    relayed = [
        Operation.from_json(m)
        for m in other.sent
        if Operation.from_json(m).op_type == OpType.NOTE_ADD.value
    ]
    assert len(relayed) == 1
    assert relayed[0].peer_id != "B"
    assert s.sessions[sid]["operations"][0].peer_id == relayed[0].peer_id


def test_client_cannot_send_server_only_operation_types():
    s = CollaborationServer()
    sid = s.create_session("s1")
    other = ScriptedWS(0)
    s.peers[sid]["B"] = other
    sender = RecordingWS(
        [
            Operation(op_type=t.value, session_id=sid, peer_id="B")
            for t in (OpType.PEER_LEFT, OpType.PEER_JOINED, OpType.ERROR)
        ]
    )
    _run(s._handle_client(sender))
    assert s.sessions[sid]["operations"] == []
    assert _types(sender) == [OpType.ERROR.value] * 3
    assert OpType.PEER_LEFT.value not in _types(other)[1:-1]
    assert _types(other).count(OpType.ERROR.value) == 0


def test_server_only_type_does_not_create_a_session():
    s = CollaborationServer()
    bad = Operation(
        op_type=OpType.PEER_LEFT.value, session_id="new", peer_id="x"
    )
    _run(s._handle_client(RecordingWS([bad])))
    assert "new" not in s.sessions
