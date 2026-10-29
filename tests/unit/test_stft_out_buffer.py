from unittest import mock

import numpy as np
import pytest

from korshmer.audio_processing import spectral_analysis
from korshmer.audio_processing.spectral_analysis import SpectralAnalyzer

N_FFT = 2048
HOP = 512


@pytest.fixture
def analyzer():
    return SpectralAnalyzer(44100)


@pytest.fixture
def audio():
    rng = np.random.default_rng(3)
    return rng.standard_normal(20000).astype(np.float32)


def _buffer_for(audio, n_fft=N_FFT, hop=HOP):
    shape = audio.shape[:-1] + (1 + n_fft // 2, 1 + audio.shape[-1] // hop)
    return np.zeros(shape, dtype=np.complex64)


def test_stft_without_out_is_unchanged(analyzer, audio):
    first = analyzer.stft(audio, n_fft=N_FFT, hop_length=HOP)
    second = analyzer.stft(audio, n_fft=N_FFT, hop_length=HOP)
    assert first.dtype == np.complex64
    np.testing.assert_array_equal(first, second)


def test_stft_writes_into_out_and_returns_it(analyzer, audio):
    reference = analyzer.stft(audio, n_fft=N_FFT, hop_length=HOP)
    buf = _buffer_for(audio)
    result = analyzer.stft(audio, n_fft=N_FFT, hop_length=HOP, out=buf)
    assert result is buf
    np.testing.assert_array_equal(buf, reference)


def test_stft_fallback_copies_into_out_when_librosa_lacks_out(
    analyzer, audio, monkeypatch
):
    reference = analyzer.stft(audio, n_fft=N_FFT, hop_length=HOP)
    monkeypatch.setattr(spectral_analysis, "_LIBROSA_STFT_HAS_OUT", False)
    buf = _buffer_for(audio)
    result = analyzer.stft(audio, n_fft=N_FFT, hop_length=HOP, out=buf)
    assert result is buf
    np.testing.assert_array_equal(buf, reference)


def test_phase_vocoder_with_out_matches_without(analyzer, audio):
    plain = analyzer.phase_vocoder(audio, 0.5, n_fft=N_FFT, hop_length=HOP)
    buf = _buffer_for(audio)
    with_out = analyzer.phase_vocoder(
        audio, 0.5, n_fft=N_FFT, hop_length=HOP, out=buf
    )
    np.testing.assert_allclose(
        with_out, plain, atol=1e-3 * float(np.max(np.abs(plain)))
    )
    assert np.any(buf != 0)


@pytest.mark.parametrize(
    "make_buffer, message",
    [
        (lambda a: np.zeros((1025, 3), dtype=np.complex64), "has shape"),
        (lambda a: np.zeros((513, 40), dtype=np.complex64), "has shape"),
        (lambda a: np.zeros((1025, 40), dtype=np.complex128), "has dtype"),
        (
            lambda a: np.zeros((1025, 40), dtype=np.float32),
            "must be a complex array",
        ),
        (lambda a: [[0j]], "must be a numpy array"),
    ],
)
def test_bad_out_rejected_before_librosa_runs(
    analyzer, audio, make_buffer, message
):
    with mock.patch.object(spectral_analysis.librosa, "stft") as stft_call:
        with pytest.raises(ValueError, match=message):
            analyzer.stft(
                audio, n_fft=N_FFT, hop_length=HOP, out=make_buffer(audio)
            )
        stft_call.assert_not_called()


def test_bad_out_rejected_in_phase_vocoder(analyzer, audio):
    with mock.patch.object(spectral_analysis.librosa, "stft") as stft_call:
        with pytest.raises(ValueError, match="has shape"):
            analyzer.phase_vocoder(
                audio,
                0.5,
                n_fft=N_FFT,
                hop_length=HOP,
                out=np.zeros((4, 4), dtype=np.complex64),
            )
        stft_call.assert_not_called()
