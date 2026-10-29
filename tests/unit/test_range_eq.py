import numpy as np

from korshmer.audio_processing.range_eq import RangeEQ


class TestRangeEQInitialization:
    def test_default_sample_rate(self):
        eq = RangeEQ()
        assert eq.sr == 44100

    def test_custom_sample_rate(self):
        eq = RangeEQ(sample_rate=48000)
        assert eq.sr == 48000


class TestShapedWeight:
    def test_shaped_weight_zero_curve(self):
        eq = RangeEQ()
        pos = 0.5

        result = eq._get_shaped_weight(pos, curve_val=0)

        assert abs(result - pos) < 1e-6

    def test_shaped_weight_positive_curve(self):
        eq = RangeEQ()
        pos = 0.5
        curve_val = 1.0

        result = eq._get_shaped_weight(pos, curve_val)

        assert result < pos

    def test_shaped_weight_negative_curve(self):
        eq = RangeEQ()
        pos = 0.5
        curve_val = -1.0

        result = eq._get_shaped_weight(pos, curve_val)

        assert result > pos

    def test_shaped_weight_at_zero(self):
        eq = RangeEQ()

        result = eq._get_shaped_weight(0, curve_val=1.0)

        assert abs(result) < 1e-10

    def test_shaped_weight_at_one(self):
        eq = RangeEQ()

        result = eq._get_shaped_weight(1, curve_val=1.0)

        assert abs(result - 1.0) < 1e-6


class TestPeakingFilter:
    def test_design_peaking_sos_shape(self):
        eq = RangeEQ()
        freq = 1000.0
        gain_db = 6.0
        bw_oct = 1.0

        sos = eq._design_peaking_sos(freq, gain_db, bw_oct)

        assert sos.shape == (6,)

    def test_design_peaking_sos_zero_gain(self):
        eq = RangeEQ()
        freq = 1000.0
        gain_db = 0.0
        bw_oct = 1.0

        sos = eq._design_peaking_sos(freq, gain_db, bw_oct)

        assert sos.shape == (6,)
        assert np.isfinite(sos).all()

    def test_design_peaking_sos_positive_gain(self):
        eq = RangeEQ()
        freq = 1000.0
        gain_db = 6.0
        bw_oct = 1.0

        sos = eq._design_peaking_sos(freq, gain_db, bw_oct)

        assert np.isfinite(sos).all()

    def test_design_peaking_sos_negative_gain(self):
        eq = RangeEQ()
        freq = 1000.0
        gain_db = -6.0
        bw_oct = 1.0

        sos = eq._design_peaking_sos(freq, gain_db, bw_oct)

        assert np.isfinite(sos).all()


class TestProcessAudio:
    def test_process_mono_audio(self):
        eq = RangeEQ(sample_rate=44100)
        audio = np.random.randn(44100)

        result = eq.process(audio, total_gain_db=0.0)

        assert result.shape == audio.shape
        assert np.isfinite(result).all()

    def test_process_stereo_audio(self):
        eq = RangeEQ(sample_rate=44100)
        audio = np.random.randn(2, 44100)

        result = eq.process(audio, total_gain_db=0.0)

        assert result.shape == audio.shape
        assert np.isfinite(result).all()

    def test_process_with_boost(self):
        eq = RangeEQ(sample_rate=44100)
        audio = np.random.randn(44100)

        result = eq.process(
            audio, freq_start=1000, freq_end=2000, total_gain_db=6.0
        )

        assert result.shape == audio.shape
        assert np.isfinite(result).all()

    def test_process_with_cut(self):
        eq = RangeEQ(sample_rate=44100)
        audio = np.random.randn(44100)

        result = eq.process(
            audio, freq_start=1000, freq_end=2000, total_gain_db=-6.0
        )

        assert result.shape == audio.shape
        assert np.isfinite(result).all()

    def test_process_zero_gain_no_change(self):
        eq = RangeEQ(sample_rate=44100)
        audio = np.random.randn(1000)

        result = eq.process(audio, total_gain_db=0.0)

        assert result.shape == audio.shape

    def test_process_inverted_frequency_range(self):
        eq = RangeEQ(sample_rate=44100)
        audio = np.random.randn(44100)

        result = eq.process(
            audio, freq_start=2000, freq_end=1000, total_gain_db=3.0
        )

        assert result.shape == audio.shape
        assert np.isfinite(result).all()

    def test_process_very_narrow_range(self):
        eq = RangeEQ(sample_rate=44100)
        audio = np.random.randn(44100)

        result = eq.process(
            audio, freq_start=1000, freq_end=1001, total_gain_db=6.0
        )

        assert result.shape == audio.shape
        assert np.isfinite(result).all()

    def test_process_above_nyquist_clamped(self):
        eq = RangeEQ(sample_rate=44100)
        audio = np.random.randn(44100)

        result = eq.process(
            audio, freq_start=1000, freq_end=30000, total_gain_db=6.0
        )

        assert result.shape == audio.shape
        assert np.isfinite(result).all()

    def test_process_with_peak_position(self):
        eq = RangeEQ(sample_rate=44100)
        audio = np.random.randn(44100)

        result1 = eq.process(
            audio,
            freq_start=200,
            freq_end=4000,
            total_gain_db=6.0,
            peak_pos_pct=25,
        )

        result2 = eq.process(
            audio,
            freq_start=200,
            freq_end=4000,
            total_gain_db=6.0,
            peak_pos_pct=75,
        )

        assert result1.shape == audio.shape
        assert result2.shape == audio.shape
        assert np.isfinite(result1).all()
        assert np.isfinite(result2).all()

    def test_process_with_curve_shaping(self):
        eq = RangeEQ(sample_rate=44100)
        audio = np.random.randn(44100)

        result = eq.process(
            audio,
            freq_start=200,
            freq_end=4000,
            total_gain_db=6.0,
            curve_left=1.0,
            curve_right=-1.0,
        )

        assert result.shape == audio.shape
        assert np.isfinite(result).all()

    def test_process_empty_audio(self):
        eq = RangeEQ(sample_rate=44100)
        audio = np.array([])

        result = eq.process(audio, total_gain_db=6.0)

        assert result.shape == audio.shape

    def test_process_preserves_dtype(self):
        eq = RangeEQ(sample_rate=44100)
        audio = np.random.randn(1000).astype(np.float32)

        result = eq.process(audio, total_gain_db=0.0)

        assert result.dtype in [np.float32, np.float64]
