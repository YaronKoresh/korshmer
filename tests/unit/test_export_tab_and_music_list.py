import os

import pytest

from korshmer.core import window_initializer
from korshmer.mixins.pages import main as pages_main
from korshmer.mixins.presets_action import PresetsMixin
from korshmer.services.file_management import FileManagementService
from korshmer.services.preset_service import (
    DEFAULT_PRESET,
    DEFAULT_SEPARATION,
    PresetService,
)

WORKING = (
    "editor_temp.mid",
    "editor_preview.wav",
    "full_mix_temp.mid",
    "full_mix_preview.wav",
    "temp_preview.wav",
    "temp_spectrum.png",
    "temp_trimmed.wav",
    "song.temp.wav",
    "vocals_ab12.wav.part",
    "notes.tmp",
    "~lock.mid",
)
MUSIC = ("idea.mid", "Rec_1794.wav", "Editor_1794.mid", "temple_bells.wav")


def test_working_files_never_show_in_my_music(tmp_path):
    for name in (*WORKING, *MUSIC):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "papit").mkdir()
    shown = {
        name for name, _ in FileManagementService.list_output_files(tmp_path)
    }
    assert shown == {*MUSIC, "papit"}
    for name in WORKING:
        assert FileManagementService.is_working_file(name), name
        assert FileManagementService.is_working_file(name.upper()), name
    for name in MUSIC:
        assert not FileManagementService.is_working_file(name), name


def test_stems_default_to_normal_wav_and_balanced_separation():
    assert DEFAULT_PRESET["wav32"] is False
    assert PresetService.get_default()["wav32"] is False
    assert DEFAULT_SEPARATION == "Balanced"
    assert pages_main.DEFAULT_SEPARATION == "Balanced"


class _Box:
    def __init__(self, value=None):
        self.value = value

    def setChecked(self, value):
        self.value = value

    def setCurrentText(self, value):
        self.value = value

    def setValue(self, value):
        self.value = value


class _Host(PresetsMixin):
    def __init__(self):
        for name in (
            "combo_octave combo_root combo_scale combo_grid slider_quantize "
            "slider_swing combo_velocity slider_vel_val rad_merged "
            "rad_separate chk_cc chk_gm chk_wav32 btn_drums btn_bass "
            "btn_other btn_vocals btn_harmonic"
        ).split():
            setattr(self, name, _Box())
        self.chk_wav32 = _Box(True)
        self.combo_separation = _Box("Best")


def test_reset_defaults_turns_off_wav32_and_picks_balanced():
    host = _Host()
    host.apply_preset_default()
    assert host.chk_wav32.value is False
    assert host.combo_separation.value == "Balanced"
    assert host.rad_separate.value is True
    host.chk_wav32.setChecked(True)
    host.apply_preset_data({})
    assert host.chk_wav32.value is False


def test_radio_buttons_have_a_visible_ring_in_both_states():
    sheet_source = open(window_initializer.__file__, encoding="utf-8").read()
    off = sheet_source.index("QRadioButton::indicator {{")
    on = sheet_source.index("QRadioButton::indicator:checked {{")
    unchecked = sheet_source[off:on]
    checked = sheet_source[on : sheet_source.index("}}", on)]
    assert "border: 1px solid #5F6368" in unchecked
    assert "background-color: white" in unchecked
    assert "solid {MAT_PRIMARY}" in checked
    assert (
        os.path.basename(window_initializer.__file__) == "window_initializer.py"
    )


@pytest.mark.parametrize("name", ["editor_temp.mid", "editor_preview.wav"])
def test_editor_playback_files_are_written_to_the_temp_folder(name):
    import korshmer.__main__ as app_main

    source = open(app_main.__file__, encoding="utf-8").read()
    assert f'os.path.join(TEMP_DIR, "{name}")' in source
    assert f'os.path.join(DATA_DIR, "{name}")' not in source
