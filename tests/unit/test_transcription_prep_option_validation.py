import numpy as np
import pytest

from korshmer.audio_processing.transcription_prep import (
    PrepOptions,
    options_for_stem,
    process_audio,
    variants_for_stem,
)

NOT_NUMBERS = [float("nan"), float("inf"), -float("inf"), "x", None, True]

FIELDS = [
    ("highpass_hz", "Highpass frequency", [-1, 20001]),
    ("harmonic_margin", "Harmonic margin", [0.5, 11]),
    ("transient_amount", "Transient amount", [-0.1, 1.1]),
    ("octave_mix", "Octave mix", [-0.1, 1.1]),
    ("normalize_db", "Normalize level", [-61, 0.1]),
]


@pytest.mark.parametrize("field,label,out_of_range", FIELDS)
def test_bad_values_rejected(field, label, out_of_range):
    for value in NOT_NUMBERS + out_of_range:
        with pytest.raises(ValueError, match=label):
            PrepOptions(**{field: value})


def test_defaults_and_edges_still_work():
    PrepOptions()
    PrepOptions(
        highpass_hz=0,
        harmonic_margin=1,
        transient_amount=1,
        octave_mix=1,
        normalize_db=-60,
    )
    assert options_for_stem("bass.wav").band_limit
    assert len(variants_for_stem("piano.wav")) == 2


def test_audio_stays_finite_with_valid_options():
    t = np.arange(22050) / 22050
    audio = 0.3 * np.sin(2 * np.pi * 220 * t)
    out, _ = process_audio(audio, 22050, PrepOptions(normalize_db=-6))
    assert np.all(np.isfinite(out))
