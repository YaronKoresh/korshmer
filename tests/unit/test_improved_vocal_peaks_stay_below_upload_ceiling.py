import numpy as np
from scipy.signal import resample_poly

from korshmer.audio_processing import clip_level

RATE = 44100


def _sung_line_with_sharp_consonants():
    seconds = np.arange(4 * RATE) / RATE
    voice = 0.05 * np.sin(2 * np.pi * 220.0 * seconds)
    voice += 0.02 * np.sin(2 * np.pi * 660.0 * seconds)
    edge = np.sin(2 * np.pi * (RATE / 4) * seconds + np.pi / 4)
    for start in range(RATE // 2, 4 * RATE, RATE // 2):
        voice[start : start + 6] += 0.9 * edge[start : start + 6]
    return voice.astype(np.float32)


def _peak_between_samples_db(audio):
    finer = resample_poly(np.asarray(audio, dtype=np.float64), 4, 1)
    return 20.0 * np.log10(np.max(np.abs(finer)))


def test_improved_vocal_peaks_between_samples_stay_below_upload_ceiling():
    improved, _readings = clip_level.improve(
        _sung_line_with_sharp_consonants(), RATE
    )

    assert _peak_between_samples_db(improved) <= clip_level.PEAK_LIMIT_DB + 0.1


def test_improved_vocal_reports_its_peak_between_samples():
    improved, readings = clip_level.improve(
        _sung_line_with_sharp_consonants(), RATE
    )
    peak = next(r for r in readings if r.name == clip_level.PEAK)

    assert abs(peak.after - _peak_between_samples_db(improved)) <= 0.1
