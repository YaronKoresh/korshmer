import numpy as np
import pytest

from korshmer.audio_processing.generate_bands import generate_bands
from korshmer.audio_processing.smart_mastering import (
    LimiterParams,
    MultibandParams,
    SmartMastering,
    SmartMasteringConfig,
    benchmark,
    freq_cut,
    validation_report,
)


def test_slope_db_zero_is_flat():
    sr = 44100
    noise = np.random.randn(2, sr).astype(np.float32)
    cfg = SmartMasteringConfig(
        sample_rate=sr, slope_db=0.0, correction_strength=1.0
    )
    m = SmartMastering(cfg)
    f_axis, _, out_db = validation_report(m, noise, sr)
    low = out_db[(f_axis >= 100) & (f_axis <= 500)]
    high = out_db[(f_axis >= 8000) & (f_axis <= 16000)]
    assert abs(np.mean(low) - np.mean(high)) < 1.0


def test_slope_db_changes_spectrum():
    sr = 44100
    noise = np.random.randn(2, sr).astype(np.float32)
    flat = SmartMastering(
        SmartMasteringConfig(
            sample_rate=sr, slope_db=0.0, correction_strength=1.0
        )
    )
    bassy = SmartMastering(
        SmartMasteringConfig(
            sample_rate=sr, slope_db=6.0, correction_strength=1.0
        )
    )
    f_axis, _, flat_out = validation_report(flat, noise, sr)
    _, _, bass_out = validation_report(bassy, noise, sr)
    low = (f_axis >= 50) & (f_axis <= 200)
    high = (f_axis >= 8000) & (f_axis <= 16000)
    assert np.mean(bass_out[low]) > np.mean(flat_out[low])
    assert np.mean(bass_out[high]) < np.mean(flat_out[high])


def test_initialization_default():
    m = SmartMastering()
    assert m.sr == 44100
    assert isinstance(m.mix_params, object)


def test_mono_input_produces_stereo():
    m = SmartMastering()
    mono = np.random.randn(44100).astype(np.float32)
    out, _ = m.process(mono, sr=44100)
    assert out.ndim == 2
    assert out.shape[0] == 2


def test_limiter_never_exceeds_ceiling():
    cfg = SmartMasteringConfig(
        limiter=LimiterParams(drive_db=0.0, ceil_db=-3.0)
    )
    m = SmartMastering(cfg)
    test = np.ones((2, 44100), dtype=np.float32) * 10.0
    out = m._apply_limiter(test, drive_db=0.0, ceil_db=-3.0)
    assert np.max(np.abs(out)) <= 10 ** (-3.0 / 20)


def test_profile_records_sections():
    m = SmartMastering()
    stats = benchmark(m, duration_sec=0.01)
    assert "multiband_compress" in stats
    assert stats["multiband_compress"] > 0


def test_spectral_slope_negative():
    m = SmartMastering()
    f = np.linspace(1, m.sr / 2, 100)
    t = m._build_target_spectrum(f)
    assert np.all(np.diff(t) <= 0)


def test_validation_report_shape():
    m = SmartMastering()
    noise = np.random.randn(44100).astype(np.float32)
    f_axis, in_db, out_db = validation_report(m, noise, 44100)
    assert f_axis.ndim == 1
    assert in_db.shape == out_db.shape == f_axis.shape


def test_stereo_widen_effect():
    m = SmartMastering()
    left = np.ones(100)
    right = np.zeros(100)
    stereo = np.stack([left, right])
    widened = m._stereo_widen(stereo, width=2.0)

    assert np.isclose(np.sum(stereo**2), np.sum(widened**2), rtol=1e-3)

    assert not np.allclose(widened[0], widened[1])


def test_freq_cut_removes_extremes():
    SmartMastering()
    sr = 44100

    t = np.linspace(0, 1, sr)
    sig = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 15000 * t)
    filtered = freq_cut(sig, sr, low_cut=100, high_cut=14000)

    assert np.std(filtered) < np.std(sig)


def test_spectral_balance_mono_audio():
    m = SmartMastering()
    audio = np.random.randn(44100)
    result = m._phase(audio, "minimal")

    assert result.shape == audio.shape
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()


def test_spectral_balance_stereo_audio():
    m = SmartMastering()
    audio = np.random.randn(2, 44100)
    result = m._phase(audio, "minimal")

    assert result.shape == audio.shape
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()


def test_spectral_balance_preserves_shape():
    m = SmartMastering()
    for shape in [(1000,), (2, 1000), (5, 1000)]:
        audio = np.random.randn(*shape)
        result = m._phase(audio, "minimal")
        assert result.shape == audio.shape


def test_spectral_balance_silent_audio():
    m = SmartMastering()
    audio = np.zeros(44100)
    result = m._phase(audio, "minimal")

    assert result.shape == audio.shape
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()


def test_spectral_balance_handles_transposed_input():
    m = SmartMastering()
    audio = np.random.randn(44100, 2)
    result = m._phase(audio, "minimal")

    assert result.shape == audio.shape


def test_phase_channel_last_input_matches_channel_first():
    m = SmartMastering()
    audio = np.random.randn(2, 4096)
    expected = m._phase(audio, "minimal")
    result = m._phase(audio.T, "minimal")

    assert result.shape == (4096, 2)
    assert np.allclose(result, expected.T)


def test_spectral_balance_energy_redistribution():
    m = SmartMastering()
    audio = np.random.randn(8192)
    result = m._phase(audio, "minimal")

    assert not np.allclose(audio, result)
    assert np.std(result) > 0


def test_spectral_balance_low_frequency_content():
    m = SmartMastering()
    t = np.linspace(0, 1, 44100)
    audio = np.sin(2 * np.pi * 100 * t)
    result = m._phase(audio, "minimal")

    assert result.shape == audio.shape
    assert not np.isnan(result).any()


def test_spectral_balance_high_frequency_content():
    m = SmartMastering()
    t = np.linspace(0, 1, 44100)
    audio = np.sin(2 * np.pi * 5000 * t)
    result = m._phase(audio, "minimal")

    assert result.shape == audio.shape
    assert not np.isnan(result).any()


def test_spectral_balance_numerical_stability():
    m = SmartMastering()
    audio = np.random.randn(1000) * 1e-10
    result = m._phase(audio, "minimal")
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()

    audio = np.random.randn(1000) * 1e10
    result = m._phase(audio, "minimal")
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()


def test_process_short_signal():
    import warnings

    m = SmartMastering()
    audio = np.zeros(1, dtype=np.float32)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out, _ = m.process(audio, sr=44100)
    assert not w, f"unexpected warnings: {w}"
    assert np.all(np.isfinite(out))


def test_process_zero_signal():
    import warnings

    m = SmartMastering()
    audio = np.zeros(44100, dtype=np.float32)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out, _ = m.process(audio, sr=44100)
    assert not w, f"unexpected warnings: {w}"
    assert np.all(np.isfinite(out))


def test_generate_bands_via_config():
    cfg = SmartMasteringConfig(multiband=MultibandParams(num_bands=3))
    m = SmartMastering(cfg)
    assert len(m.multiband_params.bands) == 3
    fcs = [b.fc for b in m.multiband_params.bands]
    expected = generate_bands(cfg.low_cut, cfg.high_cut, 3)

    assert np.allclose(fcs, expected)


def test_multiband_compress_variable_bands():
    cfg = SmartMasteringConfig(multiband=MultibandParams(num_bands=5))
    m = SmartMastering(cfg)

    audio = np.random.randn(2, 44100).astype(np.float32)
    out, _ = m.process(audio, sr=44100)
    assert out.ndim == 2
    assert out.shape[0] == 2
    assert not np.isinf(out).any()


class TestMasteringProcess:
    def setup_method(self):
        self.mastering = SmartMastering()

    def test_process_mono_audio(self):
        audio = np.random.randn(44100)
        result, _ = self.mastering.process(audio)

        assert result.ndim == 2
        assert result.shape[0] == 2
        assert result.shape[1] == 44100

    def test_process_stereo_audio(self):
        audio = np.random.randn(2, 44100)
        result, _ = self.mastering.process(audio)

        assert result.shape == audio.shape
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()

    def test_process_applies_spectral_balance(self):
        audio = np.random.randn(8192)
        result, _ = self.mastering.process(audio)

        assert not np.allclose(audio, result.squeeze())

    def test_process_limits_peaks(self):

        audio = np.random.randn(1000) * 5.0
        result, _ = self.mastering.process(audio)

        assert np.abs(result).max() < np.abs(audio).max()

    def test_process_silent_audio(self):
        audio = np.zeros(44100)
        result, _ = self.mastering.process(audio)

        assert not np.isnan(result).any()
        assert not np.isinf(result).any()

    def test_process_preserves_sample_count(self):
        for length in [1000, 8192, 44100]:
            audio = np.random.randn(length)
            result, _ = self.mastering.process(audio)
            assert result.shape[1] == length

    def test_process_normalization(self):
        audio = np.random.randn(44100)
        result, _ = self.mastering.process(audio)

        assert np.abs(result).max() <= 2.0

    def test_process_channel_independence(self):

        left = np.random.randn(8192)
        right = np.random.randn(8192)
        audio = np.vstack([left, right])

        result, _ = self.mastering.process(audio)

        assert not np.allclose(result[0], result[1])

    def test_process_consistency(self):
        audio = np.random.randn(1000)

        result1, _ = self.mastering.process(audio)
        result2, _ = self.mastering.process(audio)

        assert np.allclose(result1, result2)


class TestEdgeCases:
    def setup_method(self):
        self.mastering = SmartMastering()

    def test_very_short_audio(self):
        audio = np.random.randn(100)
        result, _ = self.mastering.process(audio)

        assert result.shape[1] == 100
        assert not np.isnan(result).any()

    def test_very_long_audio(self):

        audio = np.random.randn(441000)
        result, _ = self.mastering.process(audio)

        assert result.shape[1] == 441000
        assert not np.isnan(result).any()

    def test_multichannel_audio(self):

        audio = np.random.randn(6, 8192)
        result, _ = self.mastering.process(audio)

        assert result.shape == audio.shape

    def test_extreme_values(self):

        audio = np.random.randn(1000) * 1e6
        result, _ = self.mastering.process(audio)
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()

        audio = np.random.randn(1000) * 1e-6
        result, _ = self.mastering.process(audio)
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()

    def test_dc_offset_handling(self):
        audio = np.random.randn(1000) + 2.0
        result, _ = self.mastering.process(audio)

        assert not np.isnan(result).any()
        assert not np.isinf(result).any()

    def test_alternating_polarity(self):
        audio = np.array([1, -1] * 500)
        result, _ = self.mastering.process(audio)

        assert result.shape[1] == 1000
        assert not np.isnan(result).any()


def test_audio_processors_forwards_parameters(tmp_path):
    pytest.importorskip("demucs")
    import os

    import soundfile as sf

    from korshmer.audio_processors import process_smart_mastering_file

    path = tmp_path / "sample.wav"
    data = np.random.randn(4410, 2)
    sf.write(str(path), data, 44100)

    out = process_smart_mastering_file(
        str(path), slope_db=1.0, target_lufs=-12.0
    )
    assert os.path.exists(out)
    processed, sr = sf.read(out)
    assert processed.shape[0] == data.shape[0]
    assert sr == 44100
