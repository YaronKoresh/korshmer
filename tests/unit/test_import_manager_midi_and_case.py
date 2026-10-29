import wave

import pytest

from korshmer.project_management.import_manager import ImportManager

MIDI_BYTES = (
    b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0"
    b"MTrk\x00\x00\x00\x04\x00\xff\x2f\x00"
)


def _wav(path):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * 800)


@pytest.fixture
def manager(tmp_path):
    return ImportManager(tmp_path / "base")


@pytest.fixture
def pack(tmp_path):
    folder = tmp_path / "pack"
    folder.mkdir()
    return folder


def test_midi_file_is_imported_into_the_midi_folder(manager, pack):
    source = pack / "tune.mid"
    source.write_bytes(MIDI_BYTES)

    imported = manager.import_audio_files([source])

    assert imported == [manager.imports_dir / "midi" / "tune.mid"]
    assert imported[0].read_bytes() == MIDI_BYTES


def test_midi_extension_midi_and_uppercase_are_imported(manager, pack):
    first = pack / "a.midi"
    second = pack / "B.MID"
    first.write_bytes(MIDI_BYTES)
    second.write_bytes(MIDI_BYTES)

    imported = manager.import_audio_files([first, second])

    assert sorted(p.name for p in imported) == ["B.MID", "a.midi"]


def test_fake_midi_file_is_skipped(manager, pack):
    fake = pack / "fake.mid"
    fake.write_bytes(b"this is not a midi file")

    assert manager.import_audio_files([fake]) == []
    assert not (manager.imports_dir / "midi" / "fake.mid").exists()


def test_batch_import_takes_uppercase_extensions(manager, pack):
    _wav(pack / "kick.wav")
    _wav(pack / "SNARE.WAV")
    (pack / "tune.MID").write_bytes(MIDI_BYTES)
    (pack / "notes.txt").write_text("not audio")

    imported = manager.batch_import_from_directory(pack)

    assert sorted(p.name for p in imported) == [
        "SNARE.WAV",
        "kick.wav",
        "tune.MID",
    ]


def test_batch_import_recurses_and_can_stay_flat(manager, pack):
    _wav(pack / "top.wav")
    nested = pack / "nested"
    nested.mkdir()
    _wav(nested / "deep.wav")

    flat = manager.batch_import_from_directory(pack, recursive=False)
    assert [p.name for p in flat] == ["top.wav"]

    deep = manager.batch_import_from_directory(pack, recursive=True)
    assert sorted(p.name for p in deep) == ["deep.wav", "top_1.wav"]


def test_batch_import_accepts_extensions_without_dot_and_any_case(
    manager, pack
):
    _wav(pack / "kick.wav")
    (pack / "tune.mid").write_bytes(MIDI_BYTES)

    imported = manager.batch_import_from_directory(
        pack, file_extensions=["WAV"]
    )

    assert [p.name for p in imported] == ["kick.wav"]


def test_batch_import_ignores_directories_named_like_audio(manager, pack):
    (pack / "folder.wav").mkdir()
    _wav(pack / "real.wav")

    imported = manager.batch_import_from_directory(pack)

    assert [p.name for p in imported] == ["real.wav"]


def test_riff_wrapped_midi_file_is_imported(manager, pack):
    source = pack / "wrapped.mid"
    source.write_bytes(b"RIFF\x20\x00\x00\x00RMIDdata" + MIDI_BYTES)

    assert manager.import_audio_files([source]) == [
        manager.imports_dir / "midi" / "wrapped.mid"
    ]
