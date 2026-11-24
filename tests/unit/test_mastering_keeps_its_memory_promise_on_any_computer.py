import tracemalloc

import numpy as np
import pytest

from korshmer.audio_processing import smart_mastering
from korshmer.audio_processing.smart_mastering import (
    SmartMastering,
    SmartMasteringConfig,
    mastering_memory_bytes,
)
from korshmer.utils import worker_count

RATE = 44100


def _song(seconds):
    time = np.arange(int(seconds * RATE)) / RATE
    rng = np.random.default_rng(21)
    tone = 0.3 * np.sin(2 * np.pi * 110 * time) + 0.1 * np.sin(
        2 * np.pi * 1760 * time
    )
    noise = 0.05 * rng.normal(0, 1, time.shape)
    left = tone + noise
    right = 0.9 * tone + np.roll(noise, 300)
    return np.stack([left, right]) * np.clip(time / 2.0, 0, 1)


def _peak_while_mastering(song):
    tracemalloc.start()
    SmartMastering(SmartMasteringConfig()).process(song, RATE)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


@pytest.mark.parametrize("cores", [4, 16, 64])
def test_a_song_is_mastered_within_the_memory_it_asks_for_whatever_the_core_count(
    monkeypatch, cores
):
    monkeypatch.setattr(worker_count, "cpu_cores", lambda: cores)
    song = _song(24)

    peak = _peak_while_mastering(song)

    assert peak <= mastering_memory_bytes(song.shape[-1], RATE, RATE)


@pytest.mark.parametrize("seconds", [12, 60 * 60])
def test_a_computer_with_many_cores_masters_with_as_little_memory_as_one_with_few(
    monkeypatch, seconds
):
    song_bytes = 2 * seconds * RATE * 8
    bands = smart_mastering.EXCITER_BAND_COUNT
    monkeypatch.setattr(worker_count, "cpu_cores", lambda: 4)
    few = smart_mastering._bands_at_once(bands, song_bytes)
    monkeypatch.setattr(worker_count, "cpu_cores", lambda: 32)
    many = smart_mastering._bands_at_once(bands, song_bytes)

    assert many == few
    assert many <= smart_mastering.EXCITER_MAX_BANDS_AT_ONCE


def test_more_cores_never_change_the_mastered_sound(monkeypatch):
    song = _song(4)
    monkeypatch.setattr(worker_count, "cpu_cores", lambda: 2)
    few, _rate = SmartMastering(SmartMasteringConfig()).process(song, RATE)
    monkeypatch.setattr(worker_count, "cpu_cores", lambda: 32)
    many, _rate = SmartMastering(SmartMasteringConfig()).process(song, RATE)

    assert np.array_equal(few, many)


def test_the_band_budget_counts_what_each_band_really_holds():
    hour = 60 * 60 * RATE * 2 * 8
    allowed = smart_mastering.EXCITER_WORKING_BYTES
    bands = smart_mastering._bands_at_once(40, hour // 20)

    assert bands * 4 * (hour // 20) <= allowed
