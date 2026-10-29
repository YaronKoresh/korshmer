import warnings

import numpy as np
import pytest

from korshmer.audio_processing.distortion_noise import (
    Distortion,
    NoiseGate,
    NoiseReduction,
    Saturation,
)
from korshmer.audio_processing.dynamics import (
    Compressor,
    Limiter,
    MultiBandCompressor,
)
from korshmer.audio_processing.effects import (
    Chorus,
    Delay,
    Flanger,
    Phaser,
    Reverb,
    StereoWidener,
    Tremolo,
)
from korshmer.audio_processing.equalizers import GraphicEQ, ParametricEQ
from korshmer.audio_processing.metering import Analyzer, AudioMetering
from korshmer.audio_processing.range_eq import RangeEQ
from korshmer.audio_processing.spectral_analysis import SpectralAnalyzer
from korshmer.audio_processing.validation import (
    EMPTY_AUDIO_MESSAGE,
    is_empty_audio,
    validate_audio_not_empty,
)

EMPTY_INPUTS = [
    pytest.param(np.array([], dtype=np.float32), id="mono"),
    pytest.param(np.zeros((2, 0), dtype=np.float32), id="stereo"),
]


def _peq_with_band():
    eq = ParametricEQ(44100)
    eq.add_band(1000, 6.0, 1.0)
    return eq


PROCESSORS = [
    ("Compressor", lambda a: Compressor().process(a)),
    ("MultiBandCompressor", lambda a: MultiBandCompressor().process(a)),
    ("Limiter", lambda a: Limiter().process(a)),
    ("Reverb", lambda a: Reverb().process(a)),
    ("Delay", lambda a: Delay().process(a)),
    ("Chorus", lambda a: Chorus().process(a)),
    ("Flanger", lambda a: Flanger().process(a)),
    ("Phaser", lambda a: Phaser().process(a)),
    ("Tremolo", lambda a: Tremolo().process(a)),
    ("StereoWidener", lambda a: StereoWidener().process(a)),
    ("ParametricEQ", lambda a: _peq_with_band().process(a)),
    ("ParametricEQ-empty", lambda a: ParametricEQ().process(a)),
    ("GraphicEQ", lambda a: GraphicEQ().process(a)),
    ("RangeEQ", lambda a: RangeEQ().process(a, 200, 2000, 6.0)),
    ("soft_clip", lambda a: Distortion().soft_clip(a)),
    ("hard_clip", lambda a: Distortion().hard_clip(a)),
    ("bit_crusher", lambda a: Distortion().bit_crusher(a)),
    ("tape_saturation", lambda a: Saturation().tape_saturation(a)),
    ("NoiseGate", lambda a: NoiseGate().process(a)),
    ("NoiseReduction", lambda a: NoiseReduction().process(a)),
    ("pitch_shift", lambda a: SpectralAnalyzer().pitch_shift(a, 2)),
]

METERS = [
    ("peak_level", lambda a: AudioMetering().peak_level(a)),
    ("rms_level", lambda a: AudioMetering().rms_level(a)),
    ("crest_factor", lambda a: AudioMetering().crest_factor(a)),
    ("dynamic_range", lambda a: AudioMetering().dynamic_range(a)),
    ("true_peak", lambda a: AudioMetering().true_peak(a)),
    ("lufs", lambda a: AudioMetering().lufs(a)),
    ("stereo_correlation", lambda a: AudioMetering().stereo_correlation(a)),
    ("spectrum_analysis", lambda a: AudioMetering().spectrum_analysis(a)),
    ("analyze_audio", lambda a: Analyzer().analyze_audio(a)),
    ("frequency_analysis", lambda a: Analyzer().frequency_analysis(a)),
    ("stft", lambda a: SpectralAnalyzer().stft(a)),
    ("mfcc", lambda a: SpectralAnalyzer().mfcc(a)),
    (
        "learn_noise_profile",
        lambda a: NoiseReduction().learn_noise_profile(a),
    ),
]


class TestHelpers:
    def test_is_empty_audio_uses_size_not_len(self):
        assert is_empty_audio(np.zeros((2, 0))) is True
        assert is_empty_audio(np.array([])) is True
        assert is_empty_audio(None) is True
        assert is_empty_audio([]) is True
        assert is_empty_audio(np.zeros(3)) is False
        assert is_empty_audio(np.zeros((2, 3))) is False

    def test_validate_audio_not_empty(self):
        audio = np.ones(4)
        assert validate_audio_not_empty(audio) is audio
        with pytest.raises(ValueError, match=EMPTY_AUDIO_MESSAGE):
            validate_audio_not_empty(np.zeros((2, 0)))


@pytest.mark.parametrize(
    "name,func", PROCESSORS, ids=[p[0] for p in PROCESSORS]
)
@pytest.mark.parametrize("audio", EMPTY_INPUTS)
def test_processors_return_empty_copy(name, func, audio):
    result = func(audio)
    assert isinstance(result, np.ndarray)
    assert result.shape == audio.shape
    assert result.dtype == audio.dtype
    assert result is not audio


@pytest.mark.parametrize(
    "name,func", PROCESSORS, ids=[p[0] for p in PROCESSORS]
)
def test_processors_reject_none(name, func):
    with pytest.raises(ValueError):
        func(None)


@pytest.mark.parametrize("name,func", METERS, ids=[m[0] for m in METERS])
@pytest.mark.parametrize("audio", EMPTY_INPUTS)
def test_meters_raise_clear_error(name, func, audio):
    with pytest.raises(ValueError, match=EMPTY_AUDIO_MESSAGE):
        func(audio)


def test_reverb_adds_no_tail_to_empty_input():
    assert len(Reverb().process(np.array([], dtype=np.float32))) == 0
    assert Reverb().process(np.zeros((2, 0), dtype=np.float32)).shape == (2, 0)


def test_rms_level_empty_no_nan_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=EMPTY_AUDIO_MESSAGE):
            AudioMetering().rms_level(np.array([]))


def test_keyword_audio_argument_is_checked():
    out = Compressor().process(audio_data=np.zeros((2, 0), dtype=np.float32))
    assert out.shape == (2, 0)
    with pytest.raises(ValueError, match=EMPTY_AUDIO_MESSAGE):
        AudioMetering().peak_level(audio_data=np.array([]))


def test_smart_mastering_empty_returns_tuple():
    from korshmer.audio_processing.smart_mastering import SmartMastering

    y = np.zeros((2, 0), dtype=np.float32)
    out, sr = SmartMastering().process(y, 44100)
    assert out.shape == (2, 0)
    assert out.dtype == np.float32
    assert sr == 44100


def test_regular_audio_unchanged():
    sr = 44100
    t = np.arange(sr) / sr
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    meter = AudioMetering(sr)
    assert meter.peak_level(sine) == pytest.approx(
        20 * np.log10(np.max(np.abs(sine)) + 1e-10)
    )
    assert meter.rms_level(sine) == pytest.approx(
        20 * np.log10(np.sqrt(np.mean(sine**2)) + 1e-10)
    )
    expected = np.tanh(sine)
    assert Distortion().soft_clip(sine, 0).shape == sine.shape
    np.testing.assert_allclose(Distortion().soft_clip(sine, 0), expected)
    reverb_out = Reverb(sr).process(sine)
    assert len(reverb_out) == len(sine) + 2 * sr
