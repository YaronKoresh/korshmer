from unittest import mock

import numpy as np
import pytest

from korshmer.audio_processing.distortion_noise import NoiseReduction
from korshmer.audio_processing.metering import Analyzer, AudioMetering
from korshmer.audio_processing.spectral_analysis import SpectralAnalyzer
from korshmer.audio_processing.validation import (
    MAX_N_FFT,
    MIN_N_FFT,
    validate_n_fft,
)


@pytest.fixture
def audio():
    rng = np.random.default_rng(0)
    return rng.standard_normal(16384).astype(np.float64)


class TestValidateNFft:
    @pytest.mark.parametrize("value", [256, 512, 1024, 2048, 4096, 8192])
    def test_valid_values_unchanged(self, value):
        assert validate_n_fft(value) == value

    def test_numpy_integer_accepted(self):
        assert validate_n_fft(np.int64(1024)) == 1024

    def test_bounds_constants(self):
        assert (MIN_N_FFT, MAX_N_FFT) == (256, 8192)
        assert SpectralAnalyzer.MIN_N_FFT == MIN_N_FFT
        assert SpectralAnalyzer.MAX_N_FFT == MAX_N_FFT

    @pytest.mark.parametrize("value", [0, -1024, 128, 16384])
    def test_out_of_range(self, value):
        with pytest.raises(ValueError, match="n_fft must be between"):
            validate_n_fft(value)

    @pytest.mark.parametrize("value", [1000, 3000])
    def test_not_power_of_two(self, value):
        with pytest.raises(ValueError, match="power of 2"):
            validate_n_fft(value)

    @pytest.mark.parametrize(
        "value", [True, 2048.0, float("nan"), "2048", None]
    )
    def test_wrong_types(self, value):
        with pytest.raises(ValueError, match="n_fft"):
            validate_n_fft(value)


class TestEntryPoints:
    def test_stft_rejects_before_librosa(self, audio):
        with mock.patch(
            "korshmer.audio_processing.spectral_analysis.librosa.stft"
        ) as stft:
            with pytest.raises(ValueError, match="power of 2"):
                SpectralAnalyzer().stft(audio, n_fft=1000)
            stft.assert_not_called()

    def test_n_fft_checked_before_hop_length(self, audio):
        with pytest.raises(ValueError, match="n_fft"):
            SpectralAnalyzer().stft(audio, n_fft=128, hop_length=512)

    @pytest.mark.parametrize(
        "call",
        [
            lambda a: SpectralAnalyzer().phase_vocoder(a, 1.5, n_fft=1000),
            lambda a: SpectralAnalyzer().pitch_shift(a, 2, n_fft=1000),
            lambda a: SpectralAnalyzer().formant_preserving_pitch_shift(
                a, 2, n_fft=1000
            ),
            lambda a: SpectralAnalyzer().spectral_centroid(a, n_fft=1000),
            lambda a: SpectralAnalyzer().spectral_flatness(a, n_fft=1000),
            lambda a: SpectralAnalyzer().mfcc(a, n_fft=1000),
            lambda a: SpectralAnalyzer().chroma(a, n_fft=1000),
            lambda a: AudioMetering().spectrum_analysis(a, n_fft=16384),
            lambda a: AudioMetering().spectral_centroid(a, n_fft=16384),
            lambda a: Analyzer().frequency_analysis(a, n_fft=16384),
            lambda a: NoiseReduction().learn_noise_profile(a, n_fft=100),
            lambda a: NoiseReduction().process(a, n_fft=100),
        ],
    )
    def test_entry_points_reject(self, audio, call):
        with pytest.raises(ValueError, match="n_fft"):
            call(audio)

    def test_valid_n_fft_still_works(self, audio):
        spec = SpectralAnalyzer().stft(audio, n_fft=1024, hop_length=256)
        assert spec.shape[0] == 513
        result = Analyzer().frequency_analysis(audio, n_fft=4096)
        assert result["spectrum"].shape[0] == 2049

    def test_zero_crossing_frame_length_not_restricted(self, audio):
        zcr = SpectralAnalyzer().zero_crossing_rate(
            audio, frame_length=1000, hop_length=250
        )
        assert zcr.ndim == 1
