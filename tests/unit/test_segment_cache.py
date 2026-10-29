import numpy as np
import pytest
import soundfile as sf

import korshmer.core.audio_engine as audio_engine_module
from korshmer.constants import SEGMENT_CACHE_MAX_BYTES
from korshmer.core.audio_engine import AudioEngine
from korshmer.core.segment_cache import SegmentCache

MB = 1024 * 1024


def _block(megabytes):
    return np.zeros(megabytes * MB, dtype=np.uint8)


def test_budget_is_100_megabytes():
    assert SEGMENT_CACHE_MAX_BYTES == 100 * MB


class TestSegmentCache:
    def test_stores_and_returns_value(self):
        cache = SegmentCache(10 * MB)
        value = _block(1)
        assert cache.put(("a", 0, 1), value) is True
        assert cache.get(("a", 0, 1)) is value
        assert cache.get(("missing", 0, 1)) is None
        assert cache.size_bytes == MB

    def test_evicts_least_recently_used_when_full(self):
        cache = SegmentCache(100 * MB)
        cache.put("a", _block(40))
        cache.put("b", _block(40))
        assert cache.get("a") is not None
        cache.put("c", _block(40))
        assert cache.get("b") is None
        assert cache.get("a") is not None
        assert cache.get("c") is not None
        assert cache.size_bytes == 80 * MB

    def test_total_never_exceeds_budget(self):
        cache = SegmentCache(100 * MB)
        for index in range(30):
            cache.put(index, _block(7))
            assert cache.size_bytes <= 100 * MB
        assert len(cache) == 100 // 7

    def test_segment_larger_than_budget_is_not_stored(self):
        cache = SegmentCache(10 * MB)
        assert cache.put("big", _block(11)) is False
        assert cache.get("big") is None
        assert cache.size_bytes == 0

    def test_oversized_replacement_drops_old_value(self):
        cache = SegmentCache(10 * MB)
        cache.put("k", _block(1))
        assert cache.put("k", _block(20)) is False
        assert cache.get("k") is None

    def test_clear_empties_cache(self):
        cache = SegmentCache(10 * MB)
        cache.put("a", _block(1))
        cache.clear()
        assert len(cache) == 0
        assert cache.size_bytes == 0

    @pytest.mark.parametrize("bad", [0, -1, 1.5, True, "100"])
    def test_invalid_budget_is_rejected(self, bad):
        with pytest.raises(ValueError):
            SegmentCache(bad)


@pytest.fixture
def two_wavs(tmp_path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    ramp = np.linspace(-0.9, 0.9, 4000, dtype="float32").reshape(-1, 1)
    sf.write(str(first), ramp, 44100, subtype="FLOAT")
    sf.write(str(second), -ramp, 44100, subtype="FLOAT")
    return str(first), str(second)


@pytest.fixture
def no_audio_device(monkeypatch):
    monkeypatch.setattr(
        audio_engine_module.sd,
        "query_devices",
        lambda kind=None: {"default_samplerate": 44100.0},
    )


def test_repeated_segment_read_hits_disk_once(two_wavs, monkeypatch):
    first, _ = two_wavs
    engine = AudioEngine()
    engine.current_path = first
    calls = []
    real_read = audio_engine_module.read_audio_frames

    def counting_read(path, start, stop):
        calls.append((path, start, stop))
        return real_read(path, start, stop)

    monkeypatch.setattr(audio_engine_module, "read_audio_frames", counting_read)

    a = engine.read_segment(100, 500)
    b = engine.read_segment(100, 500)

    assert len(calls) == 1
    assert a is b
    assert a.shape == (400, 1)


def test_segment_read_without_loaded_file_returns_none():
    engine = AudioEngine()
    assert engine.read_segment(0, 10) is None


def test_bad_segment_range_returns_none(two_wavs):
    first, _ = two_wavs
    engine = AudioEngine()
    engine.current_path = first
    assert engine.read_segment(0, 10**9) is None
    assert len(engine._segment_cache) == 0


def test_loading_a_new_file_clears_segments(two_wavs, no_audio_device):
    first, second = two_wavs
    engine = AudioEngine()
    assert engine.load(first) is True
    engine.read_segment(0, 100)
    assert len(engine._segment_cache) == 1

    assert engine.load(second) is True
    assert len(engine._segment_cache) == 0
    segment = engine.read_segment(0, 100)
    expected, _ = sf.read(second, dtype="float32", always_2d=True)
    np.testing.assert_allclose(segment, expected[0:100], atol=1e-6)


def test_memory_pressure_clears_segments(two_wavs, no_audio_device):
    first, _ = two_wavs
    engine = AudioEngine()
    engine.current_path = first
    engine.read_segment(0, 100)
    engine._current_buffer_size = engine._default_buffer_size
    engine._reduce_buffer_size()
    assert len(engine._segment_cache) == 0
