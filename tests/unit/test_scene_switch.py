import numpy as np
import pytest

from korshmer.network.multi_streamer import MultiStreamer
from korshmer.network.scene_crossfade import CrossfadeMixer, SceneSwitcher

SR = 1000


def pcm(value, frames, channels=2):
    return np.full(frames * channels, value, dtype="<i2").tobytes()


def left(data):
    return np.frombuffer(data, dtype="<i2")[::2].astype(float)


def fade(curve="equal_power", ms=1000):
    m = CrossfadeMixer(SR, 2, curve=curve, initial="A")
    m.switch("B", ms)
    return m


def test_start_equals_old_and_end_equals_new():
    m = fade()
    out = left(m.mix({"A": pcm(10000, SR), "B": pcm(20000, SR)}))
    assert out[0] == pytest.approx(10000, abs=30)
    assert out[-1] == pytest.approx(20000, abs=400)
    assert not m.transitioning
    assert left(m.mix({"A": pcm(10000, 10), "B": pcm(20000, 10)}))[0] == 20000


def test_equal_power_midpoint_sums_to_one():
    m = fade()
    out = left(m.mix({"A": pcm(10000, SR), "B": pcm(10000, SR)}))

    mid = out[SR // 2]
    assert mid / 10000 == pytest.approx(np.sqrt(0.5) + np.sqrt(0.5), abs=0.02)
    a = left(CrossfadeMixer(SR, 2, initial="A").mix({"A": pcm(10000, SR)}))
    assert a[0] == 10000


def test_gains_power_sum_is_one_throughout():
    m = fade()
    a = left(m.mix({"A": pcm(10000, SR), "B": pcm(0, SR)})) / 10000
    m = fade()
    b = left(m.mix({"A": pcm(0, SR), "B": pcm(10000, SR)})) / 10000
    assert np.allclose(a**2 + b**2, 1.0, atol=0.01)


def test_linear_amplitude_sums_to_one():
    m = fade("linear")
    out = left(m.mix({"A": pcm(10000, SR), "B": pcm(10000, SR)}))
    assert np.allclose(out, 10000, atol=20)


def test_block_length_is_fixed_and_split_blocks_match():
    m1, m2 = fade(), fade()
    whole = m1.mix({"A": pcm(10000, SR), "B": pcm(20000, SR)})
    parts = b"".join(
        m2.mix({"A": pcm(10000, 100), "B": pcm(20000, 100)}) for _ in range(10)
    )
    assert len(whole) == SR * 4 and parts == whole


def test_switch_in_the_middle_keeps_continuity():
    m = fade()
    m.mix({"A": pcm(10000, SR // 2), "B": pcm(10000, SR // 2)})
    before = left(m.mix({"A": pcm(10000, 1), "B": pcm(0, 1)}))[0]
    m.switch("C", 500)
    first = left(m.mix({"A": pcm(10000, 1), "B": pcm(0, 1), "C": pcm(0, 1)}))[0]
    assert abs(first - before) < 300
    out = left(m.mix({"A": pcm(0, SR), "B": pcm(0, SR), "C": pcm(10000, SR)}))
    assert out[-1] == pytest.approx(10000, abs=30)


def test_zero_duration_is_immediate():
    m = CrossfadeMixer(SR, 2, initial="A")
    m.switch("B", 0)
    assert left(m.mix({"A": pcm(1, 4), "B": pcm(7, 4)})).tolist() == [7] * 4


def test_invalid_input():
    with pytest.raises(ValueError):
        CrossfadeMixer(0)
    with pytest.raises(ValueError):
        CrossfadeMixer(SR, 2, curve="x")
    with pytest.raises(ValueError):
        CrossfadeMixer(SR).switch("", 10)
    with pytest.raises(ValueError):
        CrossfadeMixer(SR).switch("B", float("nan"))


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


class FakeObs:
    def __init__(self):
        self.scenes = []

    def switch_scene(self, name):
        self.scenes.append(name)
        return True


def test_switcher_and_multistreamer_push_sources():
    obs, mixer, sink, multi = (
        FakeObs(),
        CrossfadeMixer(SR, 2, initial="A"),
        Sink(),
        MultiStreamer(),
    )
    multi.add_target("t", sink)
    multi.set_mixer(mixer)
    assert SceneSwitcher(obs, mixer).switch("B", 100) is True
    assert obs.scenes == ["B"]
    multi.push_sources({"A": pcm(10000, 200), "B": pcm(20000, 200)})
    assert left(sink.got[0])[-1] == 20000
    assert len(sink.got[0]) == 200 * 4
