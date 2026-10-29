import numpy as np
import pytest

from korshmer.audio_processing.distortion_noise import (
    Distortion,
    NoiseGate,
    NoiseReduction,
    Saturation,
)

NOT_NUMBERS = [float("nan"), float("inf"), -float("inf"), "x", None, True]
HUGE = [1e18, -1e18]

NUMERIC_CALLS = [
    (Distortion, "soft_clip", "drive_db"),
    (Distortion, "hard_clip", "threshold"),
    (Distortion, "bit_crusher", "sample_rate_reduction"),
    (Distortion, "fuzz", "gain_db"),
    (Distortion, "fuzz", "texture"),
    (Saturation, "tape_saturation", "drive"),
    (Saturation, "tube_saturation", "drive"),
    (Saturation, "transistor_saturation", "drive"),
    (Saturation, "harmonic_exciter", "amount"),
]

INTEGER_CALLS = [
    (Distortion, "bit_crusher", "bits"),
    (Saturation, "harmonic_exciter", "harmonic"),
]


@pytest.fixture
def tone():
    return np.sin(np.arange(4410) / 7.0)


@pytest.mark.parametrize(("cls", "method", "param"), NUMERIC_CALLS)
@pytest.mark.parametrize("value", NOT_NUMBERS + HUGE)
def test_numeric_parameter_rejected(cls, method, param, value, tone):
    with pytest.raises(ValueError):
        getattr(cls(), method)(tone, **{param: value})


@pytest.mark.parametrize(("cls", "method", "param"), INTEGER_CALLS)
@pytest.mark.parametrize("value", NOT_NUMBERS + [1.5, 0, -1, 10**18])
def test_integer_parameter_rejected(cls, method, param, value, tone):
    with pytest.raises(ValueError):
        getattr(cls(), method)(tone, **{param: value})


@pytest.mark.parametrize("param", ["threshold_db", "attack_ms", "release_ms"])
@pytest.mark.parametrize("value", NOT_NUMBERS + HUGE)
def test_noise_gate_parameter_rejected(param, value, tone):
    with pytest.raises(ValueError):
        NoiseGate(44100).process(tone, **{param: value})


@pytest.mark.parametrize("value", NOT_NUMBERS + [-0.1, 1.1, 1e18])
def test_noise_reduction_amount_rejected(value, tone):
    with pytest.raises(ValueError):
        NoiseReduction(44100).process(tone, reduction_amount=value)


def test_defaults_still_finite(tone):
    dist = Distortion()
    sat = Saturation()
    outputs = [
        dist.soft_clip(tone),
        dist.hard_clip(tone),
        dist.bit_crusher(tone),
        dist.fuzz(tone),
        sat.tape_saturation(tone),
        sat.tube_saturation(tone),
        sat.transistor_saturation(tone),
        sat.harmonic_exciter(tone),
        NoiseGate(44100).process(tone),
    ]
    for out in outputs:
        assert np.all(np.isfinite(out))
