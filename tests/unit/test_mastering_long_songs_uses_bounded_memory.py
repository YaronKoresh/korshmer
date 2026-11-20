import tracemalloc

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from korshmer.audio_processing import smart_mastering
from korshmer.audio_processing.smart_mastering import (
    SmartMastering,
    SmartMasteringConfig,
)

RATE = 44100
CHECKED_SAMPLES = (1000, 44100, 123457, 300001, 500000, 700003, 900001, 1000000)
ORIGINAL_LEFT = (
    0.0004473396185485661,
    -0.18963647129046893,
    0.13636504577202757,
    -0.34029554425016995,
    0.07428231659955262,
    -0.636900963865664,
    0.28104331566228025,
    -0.24295782839360017,
)
ORIGINAL_RIGHT = (
    0.0014058126609902651,
    -0.05705934033013892,
    0.10776221601119208,
    -0.31326009797969256,
    0.07144689452147213,
    -0.6024906983417344,
    0.2929822871907886,
    -0.24832903360261038,
)
ORIGINAL_LOUDNESS = 0.28018293823536894
ORIGINAL_PEAK = 0.933254300796991


def _song(seconds=24):
    time = np.arange(int(seconds * RATE)) / RATE
    rng = np.random.default_rng(12)
    bass = (
        0.35
        * np.sin(2 * np.pi * 55 * time)
        * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * time))
    )
    chord = 0.12 * sum(
        np.sin(2 * np.pi * pitch * time + index)
        for index, pitch in enumerate((220, 277.18, 329.63, 440))
    )
    hats = 0.05 * rng.normal(0, 1, time.shape) * (np.mod(time, 0.25) < 0.03)
    lead = (
        0.15
        * np.sin(2 * np.pi * 880 * time + 2 * np.sin(2 * np.pi * 5 * time))
        * (np.mod(time, 2.0) < 1.2)
    )
    left = bass + chord + hats + lead
    right = bass + 0.8 * chord + np.roll(hats, 200) + 0.9 * lead
    return np.stack([left, right]) * np.clip(time / 2.0, 0, 1)


@pytest.fixture(scope="module")
def mastered():
    song = _song()
    tracemalloc.start()
    result, rate = SmartMastering(SmartMasteringConfig()).process(song, RATE)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return song, result, rate, peak


def test_the_mastered_sound_is_the_same_as_before_the_memory_work(mastered):
    _song_in, result, rate, _peak = mastered
    picked = list(CHECKED_SAMPLES)

    assert rate == RATE
    assert result[0, picked] == pytest.approx(ORIGINAL_LEFT, rel=1e-6, abs=1e-9)
    assert result[1, picked] == pytest.approx(
        ORIGINAL_RIGHT, rel=1e-6, abs=1e-9
    )
    assert np.sqrt(np.mean(result**2)) == pytest.approx(
        ORIGINAL_LOUDNESS, rel=1e-7
    )
    assert np.abs(result).max() == pytest.approx(ORIGINAL_PEAK, rel=1e-7)


def test_mastering_holds_far_fewer_copies_of_the_song_at_once(mastered):
    song, _result, _rate, peak = mastered

    assert peak / song.nbytes < 24.0


def test_mastering_does_not_change_the_song_it_was_given():
    song = _song(6)
    untouched = song.copy()

    SmartMastering(SmartMasteringConfig()).process(song, RATE)

    assert np.array_equal(song, untouched)


def test_saturating_in_pieces_gives_exactly_the_same_sound(monkeypatch):
    rng = np.random.default_rng(3)
    band = rng.uniform(-1.0, 1.0, (2, 70_000))
    monkeypatch.setattr(smart_mastering, "SATURATION_BLOCK", 10_000_000)
    whole = smart_mastering._saturate_analog_warmth(band, RATE, 2.0, 4)

    monkeypatch.setattr(smart_mastering, "SATURATION_BLOCK", 8_192)
    pieces = smart_mastering._saturate_analog_warmth(band, RATE, 2.0, 4)

    assert np.array_equal(pieces, whole)


def test_filtering_one_channel_at_a_time_gives_exactly_the_same_sound():
    rng = np.random.default_rng(4)
    stereo = rng.uniform(-1.0, 1.0, (2, 50_000))
    band = butter(4, [0.1, 0.3], btype="band", output="sos")

    assert np.array_equal(
        smart_mastering._zero_phase(band, stereo),
        sosfiltfilt(band, stereo, axis=-1),
    )
    assert np.array_equal(
        smart_mastering._zero_phase(band, stereo[0]),
        sosfiltfilt(band, stereo[0], axis=-1),
    )


def test_loudest_channel_power_matches_measuring_both_channels_together():
    from scipy.ndimage import uniform_filter1d

    rng = np.random.default_rng(5)
    stereo = rng.uniform(-1.0, 1.0, (2, 20_000))

    together = np.max(
        uniform_filter1d(stereo**2, size=441, axis=-1, mode="nearest"), axis=0
    )

    assert np.array_equal(smart_mastering._peak_power(stereo, 441), together)


def test_a_long_song_works_on_fewer_bands_at_once_than_a_short_one():
    minute = 2 * 60 * RATE * 8

    short = smart_mastering._bands_at_once(40, minute)
    long = smart_mastering._bands_at_once(40, 10 * minute)

    assert long == 1
    assert short >= long
    assert smart_mastering._bands_at_once(40, 1_000 * minute) == 1


def test_the_number_of_bands_at_once_never_changes_the_result(monkeypatch):
    song = _song(6)
    several, _rate = SmartMastering(SmartMasteringConfig()).process(song, RATE)

    monkeypatch.setattr(smart_mastering, "_bands_at_once", lambda *_: 1)
    one, _rate = SmartMastering(SmartMasteringConfig()).process(song, RATE)

    assert np.array_equal(one, several)


def test_a_song_too_long_for_the_free_memory_is_refused_with_the_numbers(
    monkeypatch,
):
    class _Memory:
        available = 200_000_000

    monkeypatch.setattr(
        smart_mastering.psutil, "virtual_memory", lambda: _Memory()
    )

    with pytest.raises(MemoryError) as refusal:
        SmartMastering(SmartMasteringConfig()).process(_song(24), RATE)

    message = str(refusal.value)
    assert "0.2 GB" in message
    assert "needs about" in message
    assert "shorter part" in message


def test_the_memory_estimate_grows_with_the_length_of_the_song():
    one_minute = smart_mastering.mastering_memory_bytes(60 * RATE, RATE, RATE)
    ten_minutes = smart_mastering.mastering_memory_bytes(600 * RATE, RATE, RATE)

    assert ten_minutes == pytest.approx(10 * one_minute, rel=1e-6)
    assert ten_minutes < 8e9


def test_lowering_the_rate_in_pieces_gives_exactly_the_same_sound(monkeypatch):
    from scipy import signal

    rng = np.random.default_rng(6)
    raised = rng.uniform(-1.0, 1.0, (2, 4 * 30_000 + 3)).astype(np.float32)
    kernel = smart_mastering._build_oversampling_kernel(RATE, 4, 12.0)
    whole = signal.resample_poly(raised, 1, 4, axis=-1, window=kernel)

    monkeypatch.setattr(smart_mastering, "LIMITER_BLOCK", 4_096)
    pieces = smart_mastering._lowered_rate(raised, 4, kernel)

    assert pieces.shape == whole.shape
    assert np.array_equal(pieces, whole.astype(np.float64))
