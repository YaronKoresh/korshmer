import numpy as np
import pytest

from korshmer.audio_processing.dynamics import Compressor, Limiter
from korshmer.audio_processing.effects import (
    Chorus,
    Delay,
    Flanger,
    Phaser,
    Reverb,
    Tremolo,
)
from korshmer.audio_processing.validation import (
    validate_effect_integer,
    validate_effect_number,
)

NOT_NUMBERS = [float("nan"), float("inf"), -float("inf"), "x", None, True]

NUMERIC_PARAMS = [
    (Reverb, "room_size"),
    (Reverb, "wet"),
    (Reverb, "pre_delay_ms"),
    (Delay, "feedback"),
    (Delay, "mix"),
    (Chorus, "rate_hz"),
    (Chorus, "depth"),
    (Chorus, "mix"),
    (Flanger, "rate_hz"),
    (Flanger, "depth"),
    (Flanger, "mix"),
    (Phaser, "rate_hz"),
    (Phaser, "depth"),
    (Phaser, "mix"),
    (Tremolo, "rate_hz"),
    (Tremolo, "depth"),
    (Compressor, "threshold_db"),
    (Compressor, "attack_ms"),
    (Compressor, "release_ms"),
    (Compressor, "makeup_db"),
    (Limiter, "threshold_db"),
    (Limiter, "release_ms"),
]


@pytest.fixture
def tone():
    return np.sin(np.arange(4410) / 7.0)


@pytest.mark.parametrize(("cls", "param"), NUMERIC_PARAMS)
@pytest.mark.parametrize("value", NOT_NUMBERS + [1e18, -1e18])
def test_numeric_parameter_rejected(cls, param, value, tone):
    with pytest.raises(ValueError):
        cls(44100).process(tone, **{param: value})


@pytest.mark.parametrize("param", ["stages", "block_size"])
@pytest.mark.parametrize("value", NOT_NUMBERS + [1.5, 0, -1, 10**9, 10**18])
def test_phaser_integer_parameter_rejected(param, value, tone):
    with pytest.raises(ValueError):
        Phaser(44100).process(tone, **{param: value})


@pytest.mark.parametrize(
    "cls",
    [Reverb, Delay, Chorus, Flanger, Phaser, Tremolo, Compressor, Limiter],
)
def test_defaults_still_finite(cls, tone):
    out = cls(44100).process(tone)
    assert np.all(np.isfinite(out))


def test_helpers_return_value():
    assert validate_effect_number(0.5, "Mix", 0.0, 1.0) == 0.5
    assert validate_effect_number(np.float64(1.0), "Mix", 0.0, 1.0) == 1.0
    assert validate_effect_integer(np.int64(4), "Stages", 1, 12) == 4
