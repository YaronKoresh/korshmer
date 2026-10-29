import os
import platform

import pytest

from korshmer.core import fluidsynth as fs_module
from korshmer.mixins.actions import core as core_actions
from korshmer.ui.utils import file_operations


def _set_platform(monkeypatch, module, name, system):
    monkeypatch.setattr(module.sys, "platform", name)
    monkeypatch.setattr(platform, "system", lambda: system)


@pytest.mark.parametrize(
    "name", ["CON", "con", "NUL", "COM1", "LPT9", "AUX.txt", "prn.wav"]
)
def test_sanitize_filename_prefixes_reserved_names_on_windows(
    monkeypatch, name
):
    monkeypatch.setattr(file_operations.sys, "platform", "win32")
    assert file_operations.sanitize_filename(name) == f"file_{name}"


@pytest.mark.parametrize("platform_name", ["linux", "darwin"])
def test_sanitize_filename_keeps_reserved_names_off_windows(
    monkeypatch, platform_name
):
    monkeypatch.setattr(file_operations.sys, "platform", platform_name)
    assert file_operations.sanitize_filename("CON") == "CON"
    assert file_operations.sanitize_filename("COM1.txt") == "COM1.txt"


@pytest.mark.parametrize("platform_name", ["win32", "linux", "darwin"])
def test_sanitize_filename_replaces_unsafe_characters_everywhere(
    monkeypatch, platform_name
):
    monkeypatch.setattr(file_operations.sys, "platform", platform_name)
    assert file_operations.sanitize_filename('a<b>c:d"e|f?g*h') == (
        "a_b_c_d_e_f_g_h"
    )


def test_validate_path_warns_on_long_windows_path_only(monkeypatch, caplog):
    long_path = "/" + "a" * 270
    monkeypatch.setattr(file_operations.sys, "platform", "win32")
    with caplog.at_level("WARNING"):
        file_operations.validate_path(long_path)
    assert "MAX_PATH" in caplog.text

    caplog.clear()
    monkeypatch.setattr(file_operations.sys, "platform", "linux")
    with caplog.at_level("WARNING"):
        file_operations.validate_path(long_path)
    assert "MAX_PATH" not in caplog.text


def _bare_fluidsynth():
    instance = fs_module.FluidSynth.__new__(fs_module.FluidSynth)
    instance.audio_driver = None
    instance.synth = None
    instance.settings = None
    return instance


def _load_library_names(monkeypatch, system):
    tried = []

    def fake_cdll(name):
        tried.append(name)
        raise OSError("not found")

    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(fs_module, "CDLL", fake_cdll)
    monkeypatch.setattr(fs_module, "find_library", lambda _name: None)
    instance = _bare_fluidsynth()
    with pytest.raises(ImportError):
        instance._load_library(None)
    return tried


@pytest.mark.parametrize(
    "system, expected_first, expected_ext",
    [
        ("Windows", "libfluidsynth-3.dll", ".dll"),
        ("Linux", "libfluidsynth.so.3", ".so"),
        ("Darwin", "libfluidsynth.3.dylib", ".dylib"),
    ],
)
def test_fluidsynth_load_library_tries_platform_names(
    monkeypatch, system, expected_first, expected_ext
):
    tried = _load_library_names(monkeypatch, system)
    assert tried[0] == expected_first
    assert any(expected_ext in name for name in tried)


def test_fluidsynth_load_library_unknown_system_uses_linux_default(
    monkeypatch,
):
    tried = _load_library_names(monkeypatch, "FreeBSD")
    assert tried[0] == "libfluidsynth.so"


def test_fluidsynth_add_to_path_registers_dll_directory_on_windows(
    monkeypatch,
):
    added = []
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(os, "add_dll_directory", added.append, raising=False)
    monkeypatch.setenv("PATH", "base")
    instance = _bare_fluidsynth()
    instance._add_to_path("/some/dir")
    assert added == ["/some/dir"]
    assert "/some/dir" in os.environ["PATH"]


def test_fluidsynth_add_to_path_skips_dll_directory_off_windows(monkeypatch):
    added = []
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(os, "add_dll_directory", added.append, raising=False)
    monkeypatch.setenv("PATH", "base")
    instance = _bare_fluidsynth()
    instance._add_to_path("/some/dir")
    assert added == []


class _Host:
    def get_selected_midi(self):
        return "/tmp/song.mid"


def _run_piano_roll_img(monkeypatch, platform_name):
    calls = {"startfile": [], "subprocess": []}
    monkeypatch.setattr(core_actions.sys, "platform", platform_name)
    monkeypatch.setattr(
        core_actions.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: ("/tmp/out.png", "")),
    )
    monkeypatch.setattr(
        core_actions, "generate_piano_roll_image", lambda p, s: s
    )
    monkeypatch.setattr(
        core_actions.QMessageBox,
        "information",
        staticmethod(lambda *a, **k: None),
    )
    monkeypatch.setattr(
        core_actions.os,
        "startfile",
        calls["startfile"].append,
        raising=False,
    )
    monkeypatch.setattr(
        core_actions.subprocess,
        "call",
        lambda cmd: calls["subprocess"].append(cmd),
    )
    core_actions.CoreActionsMixin.run_piano_roll_img(_Host())
    return calls


def test_piano_roll_image_opens_with_startfile_on_windows(monkeypatch):
    calls = _run_piano_roll_img(monkeypatch, "win32")
    assert calls["startfile"] == ["/tmp/out.png"]
    assert calls["subprocess"] == []


def test_piano_roll_image_opens_with_open_command_on_macos(monkeypatch):
    calls = _run_piano_roll_img(monkeypatch, "darwin")
    assert calls["subprocess"] == [["open", "/tmp/out.png"]]
    assert calls["startfile"] == []


def test_piano_roll_image_is_not_opened_on_linux(monkeypatch):
    calls = _run_piano_roll_img(monkeypatch, "linux")
    assert calls == {"startfile": [], "subprocess": []}
