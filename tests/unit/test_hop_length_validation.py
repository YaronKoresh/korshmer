import os
import tempfile

import numpy as np
import pytest

from korshmer.audio_processing.metering import AudioMetering
from korshmer.audio_processing.spectral_analysis import SpectralAnalyzer
from korshmer.audio_processing.validation import (
    MAX_HOP_LENGTH,
    MIN_HOP_LENGTH,
    validate_hop_length,
)


@pytest.fixture
def audio():
    rng = np.random.default_rng(0)
    return rng.standard_normal(8192).astype(np.float64)


class TestValidateHopLength:
    def test_returns_valid_values_unchanged(self):
        assert validate_hop_length(512, n_fft=2048) == 512
        assert validate_hop_length(1) == 1
        assert validate_hop_length(MAX_HOP_LENGTH) == MAX_HOP_LENGTH

    @pytest.mark.parametrize("bad", [0, -1, -512])
    def test_rejects_zero_and_negative(self, bad):
        with pytest.raises(ValueError, match=str(bad)):
            validate_hop_length(bad, n_fft=2048)

    def test_rejects_hop_greater_than_n_fft(self):
        with pytest.raises(ValueError, match="n_fft"):
            validate_hop_length(1024, n_fft=512)

    def test_n_fft_check_skipped_when_none(self):

        assert validate_hop_length(4096) == 4096

    def test_rejects_above_max(self):
        with pytest.raises(ValueError, match=str(MAX_HOP_LENGTH)):
            validate_hop_length(MAX_HOP_LENGTH + 1, n_fft=MAX_HOP_LENGTH + 1)

    def test_min_hop_length_is_one(self):
        assert MIN_HOP_LENGTH == 1


class TestSpectralAnalyzerHopLength:
    METHODS = [
        ("spectral_centroid", "n_fft"),
        ("spectral_flatness", "n_fft"),
        ("zero_crossing_rate", "frame_length"),
        ("mfcc", "n_fft"),
        ("chroma", "n_fft"),
    ]

    @pytest.mark.parametrize("method,_bound", METHODS)
    @pytest.mark.parametrize("bad", [0, -512])
    def test_rejects_zero_and_negative(self, audio, method, _bound, bad):
        analyzer = SpectralAnalyzer()
        with pytest.raises(ValueError, match=str(bad)):
            getattr(analyzer, method)(audio, hop_length=bad)

    def test_stft_rejects_zero(self, audio):
        with pytest.raises(ValueError, match="0"):
            SpectralAnalyzer().stft(audio, hop_length=0)

    def test_phase_vocoder_rejects_computed_zero_hop(self, audio):

        with pytest.raises(ValueError):
            SpectralAnalyzer().phase_vocoder(audio, rate=1.0, n_fft=2)

    @pytest.mark.parametrize("method,_bound", METHODS)
    def test_defaults_still_work(self, audio, method, _bound):
        result = getattr(SpectralAnalyzer(), method)(audio)
        assert result is not None
        assert np.size(result) > 0

    def test_class_delegates_to_shared_validator(self, audio):

        with pytest.raises(ValueError):
            SpectralAnalyzer()._validate_hop_length(0, 2048)


class TestAudioMeteringHopLength:
    METHODS = [
        ("spectrum_analysis", "n_fft"),
        ("spectral_centroid", "n_fft"),
        ("zero_crossing_rate", "frame_length"),
    ]

    @pytest.mark.parametrize("method,_bound", METHODS)
    @pytest.mark.parametrize("bad", [0, -512])
    def test_rejects_zero_and_negative(self, audio, method, _bound, bad):
        metering = AudioMetering()
        with pytest.raises(ValueError, match=str(bad)):
            getattr(metering, method)(audio, hop_length=bad)

    @pytest.mark.parametrize("method,_bound", METHODS)
    def test_defaults_still_work(self, audio, method, _bound):
        result = getattr(AudioMetering(), method)(audio)
        assert result is not None
        assert np.size(result) > 0


class TestComputedHopGuards:
    def test_apply_lufs_short_audio_returns_unchanged(self):
        from korshmer.audio_processing.smart_mastering import SmartMastering

        master = SmartMastering()

        short = np.zeros(4410, dtype=np.float64)
        out = master._apply_lufs(short, -14.0)
        assert out.shape == short.shape
        assert np.all(np.isfinite(out))

    def test_apply_lufs_normal_audio_still_processes(self):
        from korshmer.audio_processing.smart_mastering import SmartMastering

        master = SmartMastering()
        rng = np.random.default_rng(1)
        y = (0.1 * rng.standard_normal(44100)).astype(np.float64)
        out = master._apply_lufs(y, -14.0)
        assert out.shape == y.shape
        assert np.all(np.isfinite(out))

    def test_smooth_envelope_tiny_frame_length_no_zero_hop(self):
        from korshmer.audio_processing.vocal_processing import (
            VocalSyllableDetector,
        )

        detector = VocalSyllableDetector()

        env = detector._smooth_envelope(
            np.random.default_rng(2).standard_normal(1000).astype(np.float64),
            frame_length=2,
        )
        assert env.shape[0] == 1000

    def test_generate_cc_from_audio_zero_step_ms_does_not_crash(self):
        soundfile = pytest.importorskip("soundfile")
        pretty_midi = pytest.importorskip("pretty_midi")

        from korshmer.midi_algorithms import generate_cc_from_audio

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "clip.wav")
            rng = np.random.default_rng(3)
            soundfile.write(
                path,
                (0.1 * rng.standard_normal(4410)).astype(np.float32),
                44100,
            )

            pm = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(program=0)
            inst.notes.append(
                pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5)
            )
            pm.instruments.append(inst)

            out = generate_cc_from_audio(pm, path, step_ms=0)
            assert out is pm
            assert len(out.instruments[0].control_changes) > 0
