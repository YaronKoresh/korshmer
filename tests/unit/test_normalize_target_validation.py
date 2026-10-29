import json
import logging

import numpy as np
import pytest

from korshmer.audio_processing.validation import (
    MAX_NORMALIZE_TARGET_DB,
    MIN_NORMALIZE_TARGET_DB,
    validate_normalize_target_db,
)
from korshmer.project_management.export_manager import (
    ExportManager,
    validate_preset_normalize_target,
)


@pytest.fixture
def sine():
    t = np.linspace(0, 1.0, 44100, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float64)


@pytest.fixture
def manager(tmp_path):
    return ExportManager(base_dir=tmp_path)


class TestValidateNormalizeTargetDb:
    def test_bounds_constants(self):
        assert (MIN_NORMALIZE_TARGET_DB, MAX_NORMALIZE_TARGET_DB) == (
            -60.0,
            0.0,
        )

    @pytest.mark.parametrize("value", [-60, -60.0, -18, -0.1, 0, 0.0])
    def test_valid_values_unchanged(self, value):
        assert validate_normalize_target_db(value) == value

    def test_numpy_values_accepted(self):
        assert validate_normalize_target_db(np.float64(-3.0)) == -3.0
        assert validate_normalize_target_db(np.int64(-12)) == -12

    @pytest.mark.parametrize("value", [0.1, 6, -60.1, -200])
    def test_out_of_range(self, value):
        with pytest.raises(
            ValueError, match="Normalize target must be between"
        ):
            validate_normalize_target_db(value)

    @pytest.mark.parametrize(
        "value",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
            "abc",
            "-3",
            None,
            True,
            False,
            np.bool_(True),
        ],
    )
    def test_wrong_types_and_non_finite(self, value):
        with pytest.raises(
            ValueError, match="Normalize target must be a finite number"
        ):
            validate_normalize_target_db(value)


class TestValidatePresetNormalizeTarget:
    @pytest.mark.parametrize(
        "config",
        [
            {"normalize": "none", "target_peak": 6},
            {"normalize": "rms", "target_rms": -18, "target_peak": 6},
            {"normalize": "peak", "target_peak": -0.1, "target_rms": -100},
            {"normalize": "lufs", "target_lufs": -14, "target_peak": 6},
            {"format": "wav"},
            {"normalize": "peak"},
            {"normalize": "rms"},
        ],
    )
    def test_only_used_key_is_checked(self, config):
        validate_preset_normalize_target(config)

    @pytest.mark.parametrize(
        "config",
        [
            {"normalize": "peak", "target_peak": 6},
            {"normalize": "peak", "target_peak": None},
            {"normalize": "rms", "target_rms": -100},
            {"normalize": "rms", "target_rms": "loud"},
        ],
    )
    def test_invalid_target_raises(self, config):
        with pytest.raises(ValueError, match="Normalize target"):
            validate_preset_normalize_target(config)

    def test_error_names_the_key(self):
        with pytest.raises(ValueError, match="target_rms"):
            validate_preset_normalize_target(
                {"normalize": "rms", "target_rms": -100}
            )

    def test_non_mapping_rejected(self):
        with pytest.raises(ValueError, match="mapping"):
            validate_preset_normalize_target(["peak"])


class TestNormalizeAudio:
    def test_peak_reaches_target(self, manager, sine):
        out = manager._normalize_audio(sine, "peak", {"target_peak": -6.0})
        assert np.max(np.abs(out)) == pytest.approx(10 ** (-6.0 / 20))

    def test_rms_reaches_target(self, manager, sine):
        out = manager._normalize_audio(sine, "rms", {"target_rms": -20.0})
        rms = np.sqrt(np.mean(out**2))
        assert rms == pytest.approx(10 ** (-20.0 / 20))

    @pytest.mark.parametrize("value", [-60, 0])
    def test_edges_work(self, manager, sine, value):
        out = manager._normalize_audio(sine, "peak", {"target_peak": value})
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize(
        "value", [6, 0.1, -200, float("nan"), "abc", None, True]
    )
    def test_invalid_peak_raises(self, manager, sine, value):
        with pytest.raises(ValueError, match="Normalize target"):
            manager._normalize_audio(sine, "peak", {"target_peak": value})

    def test_invalid_rms_raises(self, manager, sine):
        with pytest.raises(ValueError, match="Normalize target"):
            manager._normalize_audio(sine, "rms", {"target_rms": -100})

    def test_none_mode_ignores_unused_key(self, manager, sine):
        out = manager._normalize_audio(sine, "none", {"target_peak": 6})
        np.testing.assert_array_equal(out, sine)

    def test_validated_before_gain_computation(self, manager, sine):

        with pytest.raises(ValueError):
            manager._normalize_audio(sine, "peak", {"target_peak": "abc"})


class TestPresets:
    def test_builtin_presets_valid(self, manager):
        for config in manager.export_presets.values():
            validate_preset_normalize_target(config)
        assert manager.export_presets["CD Quality"]["target_peak"] == -0.1
        assert manager.export_presets["MP3 High"]["target_rms"] == -18

    def test_invalid_preset_in_file_skipped_with_warning(
        self, tmp_path, caplog
    ):
        presets_file = tmp_path / "saved_data" / "export_presets.json"
        presets_file.parent.mkdir(parents=True)
        presets_file.write_text(
            json.dumps(
                {
                    "Too Loud": {"normalize": "peak", "target_peak": 6},
                    "CD Quality": {"normalize": "peak", "target_peak": 3},
                    "Quiet": {"normalize": "rms", "target_rms": -24},
                    "Unused Key": {"normalize": "none", "target_peak": 6},
                }
            ),
            encoding="utf-8",
        )
        with caplog.at_level(logging.WARNING):
            manager = ExportManager(base_dir=tmp_path)

        assert "Too Loud" not in manager.export_presets

        assert manager.export_presets["CD Quality"]["target_peak"] == -0.1
        assert manager.export_presets["Quiet"]["target_rms"] == -24
        assert "Unused Key" in manager.export_presets
        assert "Too Loud" in caplog.text
        assert "target_peak" in caplog.text

    def test_non_object_presets_file_ignored(self, tmp_path):
        presets_file = tmp_path / "saved_data" / "export_presets.json"
        presets_file.parent.mkdir(parents=True)
        presets_file.write_text("[1, 2, 3]", encoding="utf-8")
        manager = ExportManager(base_dir=tmp_path)
        assert "CD Quality" in manager.export_presets

    def test_save_preset_rejects_invalid_target(self, manager):
        with pytest.raises(ValueError, match="Normalize target"):
            manager.save_preset(
                "Broken", {"normalize": "peak", "target_peak": 6}
            )
        assert "Broken" not in manager.export_presets
        assert not manager.presets_file.exists()

    def test_save_preset_accepts_valid_target(self, manager):
        manager.save_preset("Ok", {"normalize": "rms", "target_rms": -14})
        saved = json.loads(manager.presets_file.read_text(encoding="utf-8"))
        assert saved["Ok"]["target_rms"] == -14


class TestExportFile:
    def test_invalid_target_fails_with_clear_log(
        self, manager, tmp_path, sine, caplog
    ):
        sf = pytest.importorskip("soundfile")
        source = tmp_path / "in.wav"
        sf.write(str(source), sine, 44100)
        manager.export_presets["Bad"] = {
            "format": "wav",
            "sample_rate": 44100,
            "normalize": "peak",
            "target_peak": 6,
        }
        output = tmp_path / "out.wav"
        with caplog.at_level(logging.ERROR):
            ok = manager._export_file(source, output, "Bad", {})
        assert ok is False
        assert not output.exists()
        assert "invalid normalize target" in caplog.text
        assert "'Bad'" in caplog.text

    def test_valid_target_exports(self, manager, tmp_path, sine):
        sf = pytest.importorskip("soundfile")
        source = tmp_path / "in.wav"
        sf.write(str(source), sine, 44100)
        output = tmp_path / "out.wav"
        assert manager._export_file(source, output, "CD Quality", {}) is True
        data, _ = sf.read(str(output))
        assert np.max(np.abs(data)) == pytest.approx(
            10 ** (-0.1 / 20), abs=1e-3
        )
