import gc
import inspect

import librosa
import numpy as np

from .validation import (
    MAX_HOP_LENGTH,
    MAX_N_FFT,
    MIN_HOP_LENGTH,
    MIN_N_FFT,
    passthrough_empty_audio,
    reject_empty_audio,
    validate_effect_integer,
    validate_hop_length,
    validate_n_fft,
    validate_pitch_shift_semitones,
    validate_sample_rate,
)

_LIBROSA_STFT_HAS_OUT = "out" in inspect.signature(librosa.stft).parameters


def _check_stft_out(out, audio, n_fft: int, hop_length: int) -> None:
    if not isinstance(out, np.ndarray):
        raise ValueError(f"out must be a numpy array, got {type(out).__name__}")
    if not np.iscomplexobj(out):
        raise ValueError(f"out must be a complex array, got dtype {out.dtype}")
    shape = np.shape(audio)
    expected_shape = shape[:-1] + (1 + n_fft // 2, 1 + shape[-1] // hop_length)
    if out.shape != expected_shape:
        raise ValueError(
            f"out has shape {out.shape}, expected {expected_shape} for this input"
        )
    expected_dtype = librosa.util.dtype_r2c(np.asarray(audio).dtype)
    if out.dtype != expected_dtype:
        raise ValueError(
            f"out has dtype {out.dtype}, expected {np.dtype(expected_dtype)} for this input"
        )


def _stft_into(audio, n_fft: int, hop_length: int, window=None, out=None):
    kwargs = {"n_fft": n_fft, "hop_length": hop_length}
    if window is not None:
        kwargs["window"] = window
    if out is None:
        return librosa.stft(audio, **kwargs)
    if _LIBROSA_STFT_HAS_OUT:
        return librosa.stft(audio, out=out, **kwargs)
    out[...] = librosa.stft(audio, **kwargs)
    return out


class SpectralAnalyzer:
    MIN_HOP_LENGTH: int = MIN_HOP_LENGTH
    MAX_HOP_LENGTH: int = MAX_HOP_LENGTH
    MIN_N_FFT: int = MIN_N_FFT
    MAX_N_FFT: int = MAX_N_FFT

    def __init__(self, sample_rate: int = 44100) -> None:
        self.sr: int = validate_sample_rate(sample_rate)

    def _validate_hop_length(self, hop_length: int, n_fft: int) -> int:
        return validate_hop_length(hop_length, n_fft)

    @reject_empty_audio
    def stft(
        self,
        audio: np.ndarray,
        n_fft: int = 2048,
        hop_length: int = 512,
        window: str = "hann",
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        validate_n_fft(n_fft)
        self._validate_hop_length(hop_length, n_fft)
        if out is not None:
            _check_stft_out(out, audio, n_fft, hop_length)
        return _stft_into(audio, n_fft, hop_length, window=window, out=out)

    @passthrough_empty_audio
    def phase_vocoder(
        self,
        audio: np.ndarray,
        rate: float,
        n_fft: int = 2048,
        hop_length: int | None = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        validate_n_fft(n_fft)
        if hop_length is None:
            hop_length = n_fft // 4
        validate_hop_length(hop_length, n_fft)
        if out is not None:
            _check_stft_out(out, audio, n_fft, hop_length)

        stft_matrix = _stft_into(audio, n_fft, hop_length, out=out)
        stretched_stft = librosa.phase_vocoder(
            stft_matrix, rate=rate, hop_length=hop_length
        )
        stretched_audio = librosa.istft(
            stretched_stft,
            hop_length=hop_length,
            length=int(len(audio) / rate),
        )

        return stretched_audio

    @passthrough_empty_audio
    def pitch_shift(
        self,
        audio: np.ndarray,
        n_steps: float,
        n_fft: int = 2048,
    ) -> np.ndarray:
        validate_pitch_shift_semitones(n_steps)
        validate_n_fft(n_fft)
        return librosa.effects.pitch_shift(
            audio, sr=self.sr, n_steps=n_steps, n_fft=n_fft
        )

    @passthrough_empty_audio
    def formant_preserving_pitch_shift(
        self,
        audio: np.ndarray,
        n_steps: float,
        n_fft: int = 2048,
    ) -> np.ndarray:
        validate_pitch_shift_semitones(n_steps)
        fft_size = validate_n_fft(n_fft)
        hop_length = fft_size // 4

        stft_matrix = librosa.stft(audio, n_fft=fft_size, hop_length=hop_length)
        magnitude = np.abs(stft_matrix)
        phase = np.angle(stft_matrix)

        pitch_scale = 2 ** (n_steps / 12.0)
        formant_scale = 1.0

        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=fft_size)
        new_freqs = freqs * pitch_scale / formant_scale

        new_magnitude = np.zeros_like(magnitude)
        for i in range(magnitude.shape[1]):
            new_magnitude[:, i] = np.interp(freqs, new_freqs, magnitude[:, i])

        new_stft = new_magnitude * np.exp(1j * phase)
        shifted_audio = librosa.istft(
            new_stft, hop_length=hop_length, length=len(audio)
        )

        del stft_matrix, magnitude, phase, new_magnitude, new_stft
        gc.collect()

        return librosa.effects.time_stretch(
            shifted_audio, rate=1.0 / pitch_scale
        )

    @reject_empty_audio
    def spectral_centroid(
        self,
        audio: np.ndarray,
        n_fft: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        validate_n_fft(n_fft)
        validate_hop_length(hop_length, n_fft)
        return librosa.feature.spectral_centroid(
            y=audio, sr=self.sr, n_fft=n_fft, hop_length=hop_length
        )[0]

    @reject_empty_audio
    def spectral_flatness(
        self,
        audio: np.ndarray,
        n_fft: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        validate_n_fft(n_fft)
        validate_hop_length(hop_length, n_fft)
        return librosa.feature.spectral_flatness(
            y=audio, n_fft=n_fft, hop_length=hop_length
        )[0]

    @reject_empty_audio
    def zero_crossing_rate(
        self,
        audio: np.ndarray,
        frame_length: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        frame_length = validate_effect_integer(
            frame_length, "frame_length", 1, MAX_N_FFT
        )
        validate_hop_length(hop_length, frame_length)
        return librosa.feature.zero_crossing_rate(
            audio, frame_length=frame_length, hop_length=hop_length
        )[0]

    @reject_empty_audio
    def mfcc(
        self,
        audio: np.ndarray,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        n_mfcc = validate_effect_integer(n_mfcc, "n_mfcc", 1, 128)
        validate_n_fft(n_fft)
        validate_hop_length(hop_length, n_fft)
        return librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
        )

    @reject_empty_audio
    def chroma(
        self,
        audio: np.ndarray,
        n_fft: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        validate_n_fft(n_fft)
        validate_hop_length(hop_length, n_fft)
        return librosa.feature.chroma_stft(
            y=audio, sr=self.sr, n_fft=n_fft, hop_length=hop_length
        )
