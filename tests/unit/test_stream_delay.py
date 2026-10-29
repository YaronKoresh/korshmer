import pytest

from korshmer.network.delay_buffer import MAX_DELAY, DelayBuffer
from korshmer.network.multi_streamer import MultiStreamer


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class Sink:
    running = True

    def __init__(self):
        self.got = []

    def start(self):
        return True

    def stop(self):
        pass

    def push(self, pcm):
        self.got.append(pcm)
        return True


def make(delay):
    clock, sink = Clock(), Sink()
    multi = MultiStreamer()
    multi._delay = DelayBuffer(clock=clock)
    multi.add_target("t", sink)
    multi.set_delay(delay)
    return clock, sink, multi


def test_block_arrives_not_before_the_delay():
    clock, sink, multi = make(2.0)
    multi.push(b"\x01\x02")
    assert sink.got == []
    clock.now += 1.9
    multi.push(b"")
    assert sink.got == []
    clock.now += 0.1
    multi.push(b"")
    assert sink.got == [b"\x01\x02"]


def test_zero_delay_is_immediate():
    _, sink, multi = make(0)
    assert multi.push(b"ab") == 1
    assert sink.got == [b"ab"]


def test_silenced_window_is_replaced_by_silence():
    clock, sink, multi = make(2.0)
    multi.push(b"\x01\x01")
    clock.now += 1.0
    multi.push(b"\x02\x02")
    clock.now += 0.5
    assert multi.silence_last(0.5) == 1
    clock.now += 2.0
    multi.push(b"")
    assert sink.got == [b"\x01\x01", b"\x00\x00"]


def test_changing_delay_loses_no_blocks():
    clock, sink, multi = make(2.0)
    multi.push(b"a")
    clock.now += 0.5
    multi.push(b"b")
    multi.set_delay(5.0)
    clock.now += 4.0
    multi.push(b"")
    assert sink.got == []
    multi.set_delay(0.0)
    multi.push(b"c")
    assert sink.got == [b"a", b"b", b"c"]


def test_stop_discards_buffer():
    clock, sink, multi = make(2.0)
    multi.push(b"a")
    multi.stop()
    clock.now += 5
    multi.push(b"")
    assert sink.got == []


def test_memory_is_capped():
    clock = Clock()
    buf = DelayBuffer(10.0, max_bytes=4, clock=clock)
    assert buf.push(b"aa") == []
    assert buf.push(b"bb") == []
    assert buf.push(b"cc") == [b"aa"]
    assert buf.buffered_bytes == 4


@pytest.mark.parametrize(
    "bad", [-1, MAX_DELAY + 1, float("nan"), "2", None, True]
)
def test_invalid_delay_rejected(bad):
    with pytest.raises(ValueError):
        DelayBuffer().set_delay(bad)
