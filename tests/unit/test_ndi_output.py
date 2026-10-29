import builtins
import sys

import numpy as np
import pytest

from korshmer.network.multi_streamer import MultiStreamer
from korshmer.network.ndi_output import NdiOutput, valid_source_name
from tests.unit.test_youtube_live import wait_for


class FakeSender:
    def __init__(self, name, rate, channels):
        self.name, self.rate, self.channels = name, rate, channels
        self.blocks = []
        self.closed = False

    def write(self, planar):
        self.blocks.append(planar)

    def close(self):
        self.closed = True


def make(**kw):
    senders = []

    def factory(*a):
        senders.append(FakeSender(*a))
        return senders[-1]

    return NdiOutput(sender_factory=factory, **kw), senders


def test_push_reaches_sender_as_planar_float_and_stop_closes():
    out, senders = make(source_name="Show 1")
    assert out.start()
    pcm = np.array([[16384, -16384], [0, 32767]], dtype="<i2").tobytes()
    assert out.push(pcm)
    assert wait_for(lambda: senders[0].blocks)
    blk = senders[0].blocks[0]
    assert blk.shape == (2, 2) and blk.dtype == np.float32
    assert blk[0, 0] == pytest.approx(0.5) and blk[1, 0] == pytest.approx(-0.5)
    assert senders[0].name == "Show 1" and senders[0].channels == 2
    out.stop()
    assert senders[0].closed and not out.running
    assert out.push(pcm) is False


def test_missing_library_start_false_without_exception(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("cyndilib"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for mod in [m for m in sys.modules if m.startswith("cyndilib")]:
        monkeypatch.delitem(sys.modules, mod)
    status = []
    out = NdiOutput(on_status=status.append)
    assert out.start() is False
    assert "not available" in status[0]
    assert out.push(b"ab") is False


def test_factory_error_returns_false():
    def boom(*a):
        raise OSError("no network")

    status = []
    out = NdiOutput(sender_factory=boom, on_status=status.append)
    assert out.start() is False and "OSError" in status[0]


def test_push_never_blocks_and_drops_oldest():
    out, _ = make(max_queue=2)
    out._running = True
    assert out.push(b"1") and out.push(b"2")
    assert out.push(b"3") is False
    assert out.dropped_blocks == 1


def test_sender_failure_stops_output_and_reports():
    class Failing(FakeSender):
        def write(self, planar):
            raise RuntimeError("x")

    status = []
    out = NdiOutput(sender_factory=Failing, on_status=status.append)
    assert out.start()
    out.push(b"\x00\x00\x00\x00")
    assert wait_for(lambda: not out.running)
    assert any("failed" in t for t in status)


def test_partial_frame_is_ignored_and_validation():
    out, senders = make()
    out.start()
    out.push(b"\x01")
    out.push(b"\x00\x00\x00\x00\x00")
    assert wait_for(lambda: len(senders[0].blocks) == 1)
    assert senders[0].blocks[0].shape == (2, 1)
    out.stop()
    for bad in ("", "a" * 65, "x\ny", "../x", None):
        assert not valid_source_name(bad)
    with pytest.raises(ValueError):
        NdiOutput(source_name="bad/name")
    with pytest.raises(ValueError):
        NdiOutput(channels=6)


def test_works_as_multistreamer_target():
    out, senders = make()
    multi = MultiStreamer()
    multi.add_target("ndi", out)
    assert multi.start() == {"ndi": True}
    assert multi.push(b"\x00\x00\x00\x00") == 1
    assert wait_for(lambda: senders[0].blocks)
    multi.stop()


def test_real_cyndilib_if_installed():
    pytest.importorskip("cyndilib")
    out = NdiOutput(source_name="korshmer-unit-test")
    assert out.start()
    assert out.push(np.zeros((480, 2), dtype="<i2").tobytes())
    out.stop()
    assert not out.running
