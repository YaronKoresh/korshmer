import asyncio

from korshmer.collaboration.operations import Operation, OpType
from korshmer.collaboration.server import CollaborationServer


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, message):
        self.sent.append(message)


def _run(coro):
    return asyncio.run(coro)


def _op(session_id, peer_id, op_type=OpType.NOTE_ADD.value):
    return Operation(op_type=op_type, session_id=session_id, peer_id=peer_id)


class TestBroadcastExcludesSender:
    def test_sender_excluded(self):
        s = CollaborationServer()
        sid = s.create_session("s1")
        a, b = FakeWS(), FakeWS()
        s.peers[sid] = {"A": a, "B": b}
        _run(s.broadcast(sid, _op(sid, "A"), exclude_peer="A"))
        assert len(a.sent) == 0
        assert len(b.sent) == 1

    def test_no_exclusion_sends_to_all(self):
        s = CollaborationServer()
        sid = s.create_session("s1")
        a, b = FakeWS(), FakeWS()
        s.peers[sid] = {"A": a, "B": b}
        _run(s.broadcast(sid, _op(sid, "A")))
        assert len(a.sent) == 1
        assert len(b.sent) == 1

    def test_each_peer_gets_action_once(self):
        s = CollaborationServer()
        sid = s.create_session("s1")
        a, b = FakeWS(), FakeWS()
        s.peers[sid] = {"A": a, "B": b}
        _run(s.broadcast(sid, _op(sid, "A"), exclude_peer="A"))
        _run(s.broadcast(sid, _op(sid, "B"), exclude_peer="B"))

        assert len(a.sent) == 1
        assert len(b.sent) == 1

    def test_signature_accepts_exclude_peer_keyword(self):
        s = CollaborationServer()
        sid = s.create_session("s1")

        _run(s.broadcast(sid, _op(sid, "A"), exclude_peer="A"))

    def test_unknown_session_is_noop(self):
        s = CollaborationServer()
        _run(s.broadcast("missing", _op("missing", "A"), exclude_peer="A"))


class TestPeerMembership:
    def test_peer_id_keyed_by_string(self):
        s = CollaborationServer()
        sid = s.create_session("s1")
        ws = FakeWS()
        s.peers[sid]["peer-1"] = ws
        assert "peer-1" in s.peers[sid]
        assert "peer-2" not in s.peers[sid]
