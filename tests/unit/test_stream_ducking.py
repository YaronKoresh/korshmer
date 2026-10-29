import numpy as np
import pytest

from korshmer.network.ducking import Ducker
from korshmer.network.multi_streamer import MultiStreamer

SR = 48000


def pcm(value, frames=SR, channels=2):
    return np.full(frames * channels, value, dtype="<i2").tobytes()


def samples(data):
    return np.frombuffer(data, dtype="<i2")


def db(a, b):
    return 20 * np.log10(abs(float(a)) / abs(float(b)))


class Sink:
    running = True

    def __init__(self):
        self.got = []

    def start(self):
        return True

    def stop(self):
        pass

    def push(self, data):
        self.got.append(data)
        return True


def test_silent_commentary_leaves_program_unchanged():
    d = Ducker(SR, 2)
    block = pcm(8000)
    assert d.process(block, pcm(0)) == block
    assert d.process(block, None) == block


def test_commentary_lowers_program_by_the_chosen_depth():
    d = Ducker(SR, 2, depth_db=12, attack_ms=10)
    out = samples(d.process(pcm(8000), pcm(1000)))
    program_only = out[-2] - 1000
    assert abs(db(program_only, 8000) + 12) < 1.0


def test_level_returns_to_normal_within_release_time():
    d = Ducker(SR, 2, depth_db=12, attack_ms=10, release_ms=100)
    d.process(pcm(8000), pcm(1000))
    out = samples(d.process(pcm(8000, frames=SR // 5), pcm(0, frames=SR // 5)))
    assert d.gain == pytest.approx(1.0)
    assert out[-1] == 8000
    assert out[0] < 8000


def test_gain_changes_smoothly_between_blocks():
    d = Ducker(SR, 2, depth_db=12, attack_ms=50)
    out = samples(d.process(pcm(8000, frames=480), pcm(1000, frames=480)))
    steps = np.abs(np.diff(out[::2].astype(int)))
    assert steps.max() < 200


def test_output_is_clipped_not_wrapped():
    d = Ducker(SR, 2, depth_db=0, attack_ms=1)
    out = samples(d.process(pcm(30000, frames=100), pcm(30000, frames=100)))
    assert out.max() == 32767


def test_disabled_is_identical():
    d = Ducker(SR, 2, enabled=False)
    block = pcm(8000)
    assert d.process(block, pcm(5000)) is block


def test_multistreamer_ducks_and_off_is_unchanged():
    sink, multi = Sink(), MultiStreamer()
    multi.add_target("t", sink)
    multi.push(pcm(8000, frames=10), pcm(5000, frames=10))
    assert sink.got[0] == pcm(8000, frames=10)
    multi.set_ducking(Ducker(SR, 2, depth_db=12, attack_ms=1))
    multi.push(pcm(8000, frames=SR // 10), pcm(5000, frames=SR // 10))
    assert samples(sink.got[1])[-1] != 8000


@pytest.mark.parametrize(
    "kw",
    [
        {"depth_db": -1},
        {"depth_db": float("nan")},
        {"attack_ms": 0},
        {"threshold_db": 5},
    ],
)
def test_invalid_parameters_rejected(kw):
    with pytest.raises(ValueError):
        Ducker(SR, 2, **kw)
