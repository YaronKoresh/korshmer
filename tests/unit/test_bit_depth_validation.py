import numpy as np
import pytest

from korshmer.audio_processing.validation import (
    ALLOWED_EXPORT_BIT_DEPTHS,
    validate_bit_depth,
)
from korshmer.project_management.export_manager import (
    ExportManager,
    validate_preset_bit_depth,
)


class TestValidateBitDepth:
    @pytest.mark.parametrize("depth", [16, 24, 32])
    def test_accepts_allowed(self, depth):
        assert validate_bit_depth(depth) == depth

    def test_accepts_numpy_integer(self):
        assert validate_bit_depth(np.int32(24)) == 24

    @pytest.mark.parametrize("depth", [8, 20, 0, -16, 64])
    def test_rejects_out_of_set(self, depth):
        with pytest.raises(ValueError):
            validate_bit_depth(depth)

    @pytest.mark.parametrize("depth", ["24", 24.0, True, False, None, 16.0])
    def test_rejects_wrong_type(self, depth):
        with pytest.raises(ValueError):
            validate_bit_depth(depth)

    def test_allowed_constant(self):
        assert ALLOWED_EXPORT_BIT_DEPTHS == (16, 24, 32)


class TestPresetBitDepth:
    def test_missing_bit_depth_ok(self):
        validate_preset_bit_depth({"format": "mp3", "bitrate": 320})

    def test_invalid_bit_depth_raises(self):
        with pytest.raises(ValueError):
            validate_preset_bit_depth({"format": "wav", "bit_depth": 20})


class TestExportManagerPresets:
    def _mgr(self, tmp_path):
        return ExportManager(base_dir=tmp_path)

    def test_save_preset_rejects_invalid_bit_depth(self, tmp_path):
        mgr = self._mgr(tmp_path)
        with pytest.raises(ValueError):
            mgr.save_preset("Bad", {"format": "wav", "bit_depth": 20})
        assert "Bad" not in mgr.export_presets

    @pytest.mark.parametrize("depth", [16, 24, 32])
    def test_save_preset_accepts_valid(self, tmp_path, depth):
        mgr = self._mgr(tmp_path)
        mgr.save_preset("Good", {"format": "wav", "bit_depth": depth})
        assert mgr.export_presets["Good"]["bit_depth"] == depth

    def test_load_skips_invalid_preset(self, tmp_path):
        import json

        presets_file = tmp_path / "saved_data" / "export_presets.json"
        presets_file.parent.mkdir(parents=True, exist_ok=True)
        presets_file.write_text(
            json.dumps(
                {
                    "Broken": {"format": "wav", "bit_depth": 8},
                    "Fine": {"format": "wav", "bit_depth": 24},
                }
            )
        )
        mgr = self._mgr(tmp_path)
        assert "Broken" not in mgr.export_presets
        assert "Fine" in mgr.export_presets

        assert "CD Quality" in mgr.export_presets

    def test_get_subtype_no_silent_fallback(self, tmp_path):
        mgr = self._mgr(tmp_path)
        assert mgr._get_subtype(24, "wav") == "PCM_24"
        assert mgr._get_subtype(16, "wav") == "PCM_16"
        with pytest.raises(ValueError):
            mgr._get_subtype(20, "wav")

    def test_get_subtype_flac_32_downgrades_to_24(self, tmp_path):
        mgr = self._mgr(tmp_path)
        assert mgr._get_subtype(32, "flac") == "PCM_24"
