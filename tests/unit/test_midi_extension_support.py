from pathlib import Path

import pytest

from korshmer.services.file_management import FileManagementService
from korshmer.ui.viewmodels.spectrogram_vm import SpectrogramViewModel
from korshmer.utils.filesystem import MIDI_EXTENSIONS, is_midi_path
from korshmer.visualization.spectrum import calculate_waveform_features


@pytest.mark.parametrize(
    "name",
    ["song.mid", "song.MID", "song.midi", "Song.MiDi", "a.b.midi"],
)
def test_midi_names_are_recognised(name):
    assert is_midi_path(name) is True
    assert is_midi_path(Path(name)) is True


@pytest.mark.parametrize(
    "value",
    ["song.wav", "song.mid.txt", "midi", "", "song.mi", None, 5, b"x.mid"],
)
def test_other_values_are_not_midi(value):
    assert is_midi_path(value) is False


def test_extension_list_holds_both_spellings():
    assert MIDI_EXTENSIONS == (".mid", ".midi")


def test_file_service_accepts_midi_extension():
    service = FileManagementService
    assert service.is_valid_midi("song.midi")
    assert service.is_valid_midi("SONG.MIDI")
    assert service.is_valid_file("song.midi")
    assert not service.is_valid_audio("song.midi")


def test_output_list_shows_midi_extension_files(tmp_path):
    (tmp_path / "a.midi").write_bytes(b"MThd")
    (tmp_path / "b.mid").write_bytes(b"MThd")
    (tmp_path / "c.wav").write_bytes(b"x")
    (tmp_path / "d.bin").write_bytes(b"x")

    names = {
        name for name, _ in FileManagementService.list_output_files(tmp_path)
    }

    assert names == {"a.midi", "b.mid", "c.wav"}


def test_spectrogram_skips_midi_extension():
    view_model = SpectrogramViewModel()
    assert view_model.set_loading("song.midi") is False
    assert view_model.loading is False
    assert view_model.set_loading("song.wav") is True


def test_waveform_features_skip_midi_extension():
    assert calculate_waveform_features("song.midi") == (None, 0)
