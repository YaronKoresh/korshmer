import numpy as np
import pytest

from korshmer.audio_processing.equalizers import ParametricEQ


class TestParametricEQInitialization:
    def test_default_sample_rate(self):
        eq = ParametricEQ()
        assert eq.sr == 44100

    def test_custom_sample_rate(self):
        eq = ParametricEQ(sample_rate=48000)
        assert eq.sr == 48000

    def test_empty_bands_on_init(self):
        eq = ParametricEQ()
        assert eq.bands == []


class TestAddBand:
    def test_add_single_band(self):
        eq = ParametricEQ()
        eq.add_band(1000, 6.0, 1.0)
        assert len(eq.bands) == 1
        assert eq.bands[0]["freq"] == 1000
        assert eq.bands[0]["gain_db"] == 6.0
        assert eq.bands[0]["q"] == 1.0

    def test_add_multiple_bands(self):
        eq = ParametricEQ()
        eq.add_band(1000, 6.0, 1.0)
        eq.add_band(2000, -3.0, 2.0)
        eq.add_band(500, 3.0, 0.7)
        assert len(eq.bands) == 3

    def test_add_band_default_q(self):
        eq = ParametricEQ()
        eq.add_band(1000, 6.0)
        assert eq.bands[0]["q"] == 1.0


class TestClearBands:
    def test_clear_empty_bands(self):
        eq = ParametricEQ()
        eq.clear_bands()
        assert eq.bands == []

    def test_clear_existing_bands(self):
        eq = ParametricEQ()
        eq.add_band(1000, 6.0, 1.0)
        eq.add_band(2000, -3.0, 2.0)
        eq.clear_bands()
        assert eq.bands == []


class TestProcess:
    def test_process_mono_audio_no_bands(self):
        eq = ParametricEQ()
        audio = np.random.randn(1000)
        result = eq.process(audio)
        np.testing.assert_array_equal(result, audio)

    def test_process_stereo_audio_no_bands(self):
        eq = ParametricEQ()
        audio = np.random.randn(2, 1000)
        result = eq.process(audio)
        np.testing.assert_array_equal(result, audio)

    def test_process_mono_audio_with_band(self):
        eq = ParametricEQ()
        eq.add_band(1000, 6.0, 1.0)
        audio = np.random.randn(1000)
        result = eq.process(audio)
        assert result.shape == audio.shape
        assert np.isfinite(result).all()

    def test_process_stereo_audio_with_band(self):
        eq = ParametricEQ()
        eq.add_band(1000, 6.0, 1.0)
        audio = np.random.randn(2, 1000)
        result = eq.process(audio)
        assert result.shape == audio.shape
        assert np.isfinite(result).all()

    def test_process_with_multiple_bands(self):
        eq = ParametricEQ()
        eq.add_band(500, 3.0, 1.0)
        eq.add_band(1000, 6.0, 1.5)
        eq.add_band(2000, -3.0, 0.7)
        audio = np.random.randn(2, 1000)
        result = eq.process(audio)
        assert result.shape == audio.shape
        assert np.isfinite(result).all()

    def test_process_correct_usage_pattern(self):

        eq = ParametricEQ(sample_rate=44100)
        freq = 1000
        gain = 6.0
        q = 1.5

        audio = np.random.randn(2, 44100)

        eq.add_band(freq, gain, q)
        result = eq.process(audio)

        assert result.shape == audio.shape
        assert np.isfinite(result).all()

        assert len(eq.bands) == 1
        assert eq.bands[0]["freq"] == freq
        assert eq.bands[0]["gain_db"] == gain
        assert eq.bands[0]["q"] == q


class TestQFactorValidation:
    def test_default_q_valid(self):
        eq = ParametricEQ()
        eq.add_band(1000, 6.0)
        assert eq.bands[0]["q"] == 1.0

    @pytest.mark.parametrize(
        "q", [0.1, 1.0, 1.4, 20, np.float64(2.0), np.float32(3.0)]
    )
    def test_valid_q_including_bounds(self, q):
        eq = ParametricEQ()
        eq.add_band(1000, 6.0, q)
        result = eq.process(np.random.randn(2, 4410))
        assert np.isfinite(result).all()

    @pytest.mark.parametrize(
        "q",
        [
            0,
            -1.0,
            0.05,
            20.01,
            25,
            float("nan"),
            float("inf"),
            "abc",
            None,
            True,
        ],
    )
    def test_invalid_q_rejected_in_add_band(self, q):
        eq = ParametricEQ()
        with pytest.raises(ValueError, match="Q factor"):
            eq.add_band(1000, 6.0, q)
        assert len(eq.bands) == 0

    def test_graphic_eq_still_works(self):
        from korshmer.audio_processing.equalizers import GraphicEQ

        geq = GraphicEQ()
        geq.set_gain(3, 6.0)
        result = geq.process(np.random.randn(2, 4410))
        assert np.isfinite(result).all()

    def test_process_parametric_eq_fails_before_writing(
        self, tmp_path, monkeypatch
    ):
        import soundfile as sf

        from korshmer import audio_processors

        monkeypatch.setattr(
            audio_processors, "_PROC_DIR", str(tmp_path / "out")
        )
        wav = tmp_path / "in.wav"
        sf.write(str(wav), np.random.randn(4410, 2) * 0.1, 44100)
        with pytest.raises(ValueError, match="Q factor"):
            audio_processors.process_parametric_eq(str(wav), 1000, 6.0, q=0)
        out_dir = tmp_path / "out"
        assert not out_dir.exists() or not any(out_dir.rglob("*.wav"))
