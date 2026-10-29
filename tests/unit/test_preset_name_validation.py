import json
from unittest.mock import MagicMock, patch

import pytest

from korshmer.mixins.presets_action import PresetsMixin
from korshmer.services.preset_service import (
    DEFAULT_PRESET,
    PresetService,
    validate_preset_name,
)

SERVICE_MODULE = "korshmer.services.preset_service"
DIALOG_MODULE = "korshmer.mixins.presets_action"


@pytest.fixture
def presets_file(tmp_path):
    path = tmp_path / "presets.json"
    with patch(f"{SERVICE_MODULE}.PRESETS_FILE", str(path)):
        yield path


class TestValidatePresetName:
    @pytest.mark.parametrize("name", ["Default", "default", " DEFAULT ", ""])
    def test_reserved_and_blank_rejected(self, name):
        assert validate_preset_name(name) is None

    @pytest.mark.parametrize("name", ["   ", "\t", None, 5])
    def test_whitespace_and_non_string_rejected(self, name):
        assert validate_preset_name(name) is None

    def test_valid_name_is_stripped(self):
        assert validate_preset_name("  Warm Lead  ") == "Warm Lead"


class TestPresetServiceNames:
    def test_default_key_in_file_does_not_replace_builtin(self, presets_file):
        presets_file.write_text(
            json.dumps({"Default": {"octave": "CUSTOM"}}), encoding="utf-8"
        )
        loaded = PresetService.load_all()
        assert loaded["Default"] == DEFAULT_PRESET

    def test_blank_key_in_file_is_ignored(self, presets_file):
        presets_file.write_text(
            json.dumps({"  ": {"octave": "x"}, "Keep": {"grid": "1/16"}}),
            encoding="utf-8",
        )
        loaded = PresetService.load_all()
        assert "  " not in loaded
        assert loaded["Keep"] == {"grid": "1/16"}

    def test_valid_name_round_trips(self, presets_file):
        PresetService.save_all({"Warm Lead": {"grid": "1/16"}})
        loaded = PresetService.load_all()
        assert loaded["Warm Lead"] == {"grid": "1/16"}
        assert loaded["Default"] == DEFAULT_PRESET


class _Dialog(PresetsMixin):
    def __init__(self):
        self.presets = {"Default": dict(DEFAULT_PRESET)}
        for attr in (
            "combo_octave",
            "combo_root",
            "combo_scale",
            "combo_grid",
            "combo_velocity",
            "combo_presets",
            "slider_quantize",
            "slider_swing",
            "slider_vel_val",
            "rad_merged",
            "chk_cc",
            "chk_gm",
            "chk_wav32",
        ):
            setattr(self, attr, MagicMock())
        for combo in (
            "combo_octave",
            "combo_root",
            "combo_scale",
            "combo_grid",
        ):
            getattr(self, combo).currentText.return_value = "Original"
        self.combo_velocity.currentText.return_value = "Fixed"
        self.slider_quantize.value.return_value = 100
        self.slider_swing.value.return_value = 0
        self.slider_vel_val.value.return_value = 100
        self.rad_merged.isChecked.return_value = False
        for chk in ("chk_cc", "chk_gm", "chk_wav32"):
            getattr(self, chk).isChecked.return_value = True
        for attr in (
            "btn_drums",
            "btn_bass",
            "btn_other",
            "btn_vocals",
            "btn_harmonic",
        ):
            setattr(self, attr, MagicMock(**{"isChecked.return_value": False}))


class TestSavePresetDialog:
    @pytest.mark.parametrize("entered", ["Default", "default", "   ", ""])
    def test_rejected_names_are_not_saved(self, presets_file, entered):
        dialog = _Dialog()
        with (
            patch(
                f"{DIALOG_MODULE}.get_limited_text",
                return_value=(entered, True),
            ),
            patch(f"{DIALOG_MODULE}.publish") as publish_mock,
        ):
            dialog.save_preset_dialog()
        assert list(dialog.presets) == ["Default"]
        assert dialog.presets["Default"] == DEFAULT_PRESET
        assert not presets_file.exists()
        publish_mock.assert_called_once()

    def test_valid_name_is_saved_trimmed(self, presets_file):
        dialog = _Dialog()
        with patch(
            f"{DIALOG_MODULE}.get_limited_text",
            return_value=("  Warm Lead ", True),
        ):
            dialog.save_preset_dialog()
        assert "Warm Lead" in dialog.presets
        assert "  Warm Lead " not in dialog.presets
        saved = json.loads(presets_file.read_text(encoding="utf-8"))
        assert "Warm Lead" in saved
        assert saved["Warm Lead"]["octave"] == "Original"

    def test_cancel_does_nothing(self, presets_file):
        dialog = _Dialog()
        with patch(
            f"{DIALOG_MODULE}.get_limited_text", return_value=("X", False)
        ):
            dialog.save_preset_dialog()
        assert list(dialog.presets) == ["Default"]
