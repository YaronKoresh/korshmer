import asyncio

import pytest

from korshmer.collaboration.operations import Operation, OpType
from korshmer.collaboration.rate_limit import TokenBucket
from korshmer.collaboration.server import CollaborationServer


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FloodWS:
    def __init__(self, count, session_id="s1"):
        self.count = count
        self.session_id = session_id
        self.sent = []
        self.closed = None

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
        self.closed = (code, reason)


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class TestTokenBucket:
    def test_allows_burst_then_blocks(self):
        clock = FakeClock()
        b = TokenBucket(rate=10, burst=3, clock=clock)
        assert [b.allow() for _ in range(4)] == [True, True, True, False]

    def test_refills_over_time(self):
        clock = FakeClock()
        b = TokenBucket(rate=2, burst=3, clock=clock)
        for _ in range(3):
            b.allow()
        clock.now += 0.5
        assert b.allow() is True
        assert b.allow() is False

    def test_refill_is_capped_at_burst(self):
        clock = FakeClock()
        b = TokenBucket(rate=10, burst=2, clock=clock)
        clock.now += 1000
        assert [b.allow() for _ in range(3)] == [True, True, False]

    @pytest.mark.parametrize("rate,burst", [(0, 1), (-1, 1), (1, 0)])
    def test_invalid_arguments_rejected(self, rate, burst):
        with pytest.raises(ValueError):
            TokenBucket(rate=rate, burst=burst)


class TestServerRateLimit:
    def test_flood_is_limited_and_peer_closed(self):
        s = CollaborationServer(
            max_messages_per_second=1, burst=5, max_violations=10
        )
        sid = s.create_session("s1")
        other = FloodWS(0)
        s.peers[sid]["B"] = other
        ws = FloodWS(1000)
        _run(s._handle_client(ws, "/"))
        assert len(s.sessions[sid]["operations"]) == 5
        assert ws.closed == (1008, "Rate limit exceeded")
        assert len(ws.sent) == 9
        assert Operation.from_json(ws.sent[0]).op_type == OpType.ERROR.value

    def test_peer_removed_after_close(self):
        s = CollaborationServer(
            max_messages_per_second=1, burst=1, max_violations=2
        )
        sid = s.create_session("s1")
        other = FloodWS(0)
        s.peers[sid]["B"] = other
        _run(s._handle_client(FloodWS(20), "/"))
        assert list(s.peers[sid]) == ["B"]
        left = [
            Operation.from_json(m).op_type
            for m in other.sent
            if Operation.from_json(m).op_type == OpType.PEER_LEFT.value
        ]
        assert len(left) == 1

    def test_normal_traffic_is_not_limited(self):
        s = CollaborationServer()
        sid = s.create_session("s1")
        ws = FloodWS(100)
        _run(s._handle_client(ws, "/"))
        assert len(s.sessions[sid]["operations"]) == 100
        assert ws.closed is None
        assert ws.sent == []
