import numpy as np
import pytest

from korshmer.audio_processing.basic_effects import (
    apply_filter as apply_filter_basic,
)


class TestApplyFilterTypeMapping:
    @pytest.fixture()
    def sine_44100(self):
        sr = 44100
        t = np.linspace(0, 1, sr, endpoint=False)
        return np.sin(2 * np.pi * 200 * t).astype(np.float32), sr

    @pytest.mark.parametrize(
        "long_form,short_form",
        [("lowpass", "low"), ("highpass", "high")],
    )
    def test_long_form_names_accepted_array_mode(
        self, sine_44100, long_form, short_form
    ):
        audio, sr = sine_44100
        result = apply_filter_basic(
            audio, sr, filter_type=long_form, cutoff=1000
        )
        assert result is not None
        assert len(result) == len(audio)

    @pytest.mark.parametrize("short_form", ["low", "high"])
    def test_short_form_names_still_work_array_mode(
        self, sine_44100, short_form
    ):
        audio, sr = sine_44100
        result = apply_filter_basic(
            audio, sr, filter_type=short_form, cutoff=1000
        )
        assert result is not None
        assert len(result) == len(audio)

    def test_lowpass_default_applies_lowpass_not_highpass(self, sine_44100):
        audio, sr = sine_44100
        low_result = apply_filter_basic(
            audio, sr, filter_type="low", cutoff=1000
        )
        default_result = apply_filter_basic(audio, sr, cutoff=1000)
        np.testing.assert_allclose(default_result, low_result, atol=1e-6)

    def test_file_mode_lowpass_maps_correctly(self, tmp_path, sine_44100):
        import soundfile as sf

        audio, sr = sine_44100
        in_path = str(tmp_path / "input.wav")
        out_path = str(tmp_path / "output.wav")
        sf.write(in_path, audio, sr)

        apply_filter_basic(
            in_path, out_path, filter_type="lowpass", cutoff=1000
        )

        out_low = str(tmp_path / "output_low.wav")
        apply_filter_basic(in_path, out_low, filter_type="low", cutoff=1000)

        y1, _ = sf.read(out_path)
        y2, _ = sf.read(out_low)
        np.testing.assert_allclose(y1, y2, atol=1e-6)

    def test_file_mode_unknown_type_raises(self, tmp_path, sine_44100):
        import soundfile as sf

        audio, sr = sine_44100
        in_path = str(tmp_path / "input.wav")
        out_path = str(tmp_path / "output.wav")
        sf.write(in_path, audio, sr)

        with pytest.raises(ValueError, match="Unknown filter_type"):
            apply_filter_basic(
                in_path, out_path, filter_type="notch", cutoff=1000
            )

    def test_file_mode_highpass_long_form(self, tmp_path, sine_44100):
        import soundfile as sf

        audio, sr = sine_44100
        in_path = str(tmp_path / "input.wav")
        sf.write(in_path, audio, sr)

        out_long = str(tmp_path / "output_highpass.wav")
        apply_filter_basic(
            in_path, out_long, filter_type="highpass", cutoff=1000
        )

        out_short = str(tmp_path / "output_high.wav")
        apply_filter_basic(in_path, out_short, filter_type="high", cutoff=1000)

        y1, _ = sf.read(out_long)
        y2, _ = sf.read(out_short)
        np.testing.assert_allclose(y1, y2, atol=1e-6)

    def test_default_filter_type_not_highpass_file_mode(
        self, tmp_path, sine_44100
    ):
        import soundfile as sf

        audio, sr = sine_44100
        in_path = str(tmp_path / "input.wav")
        sf.write(in_path, audio, sr)

        out_default = str(tmp_path / "output_default.wav")
        apply_filter_basic(in_path, out_default, cutoff=1000)

        out_high = str(tmp_path / "output_high.wav")
        apply_filter_basic(in_path, out_high, filter_type="high", cutoff=1000)

        y_default, _ = sf.read(out_default)
        y_high, _ = sf.read(out_high)
        assert not np.allclose(y_default, y_high, atol=1e-3)
