import json
import logging
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from korshmer.audio_file import AudioFile
from korshmer.network.link_sync import LinkSync
from korshmer.project_management.export_manager import (
    ExportManager,
    validate_export_preset,
)
from korshmer.utils import logging_config

SR = 44100
TAGS = {"title": "Song", "artist": "Band", "album": "Disc", "genre": "Pop"}
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not installed"
)


def _tone(seconds=1.0):
    t = np.arange(int(seconds * SR)) / SR
    return (0.4 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


@pytest.fixture
def source_wav(tmp_path):
    path = tmp_path / "source.wav"
    sf.write(path, np.stack([_tone(), _tone()], axis=1), SR, subtype="PCM_16")
    return path


def _export(tmp_path, source, name, config, suffix):
    manager = ExportManager(tmp_path / "base")
    manager.save_preset(
        name, {"sample_rate": SR, "normalize": "none", **config}
    )
    output = tmp_path / f"{name}.{suffix}"
    assert manager._export_file(source, output, name, dict(TAGS))
    return output


@pytest.mark.parametrize("subtype", ["PCM_S8", "PCM_16", "PCM_24", "PCM_32"])
def test_au_files_are_imported(tmp_path, subtype):
    path = tmp_path / "tone.au"
    sf.write(path, _tone(), SR, format="AU", subtype=subtype)
    loaded = AudioFile.load(str(path))
    assert loaded.sr == SR
    assert loaded.duration == pytest.approx(1.0, abs=0.01)


@needs_ffmpeg
@pytest.mark.parametrize("name", ["tone.m4a", "tone.aac"])
def test_aac_files_are_imported_through_ffmpeg(tmp_path, source_wav, name):
    path = tmp_path / name
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(source_wav), str(path)],
        check=True,
    )
    loaded = AudioFile.load(str(path))
    assert loaded.sr == SR
    assert loaded.channels == 2
    assert loaded.duration == pytest.approx(1.0, abs=0.1)


def test_a_file_no_decoder_reads_still_fails_cleanly(tmp_path):
    path = tmp_path / "broken.m4a"
    path.write_bytes(b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 64)
    with pytest.raises(ValueError, match="Failed to load audio file"):
        AudioFile.load(str(path))


def test_wav_export_carries_riff_info_tags(tmp_path, source_wav):
    output = _export(
        tmp_path, source_wav, "wav16", {"format": "wav", "bit_depth": 16}, "wav"
    )
    with sf.SoundFile(output) as handle:
        assert (handle.title, handle.artist) == ("Song", "Band")
        assert (handle.album, handle.genre) == ("Disc", "Pop")
        assert handle.frames == SR
    data, _ = sf.read(output)
    original, _ = sf.read(source_wav)
    assert np.array_equal(data, original)


def test_ogg_export_carries_vorbis_comments_and_follows_quality(
    tmp_path, source_wav
):
    from mutagen.oggvorbis import OggVorbis

    low = _export(
        tmp_path, source_wav, "low", {"format": "ogg", "quality": -1}, "ogg"
    )
    high = _export(
        tmp_path, source_wav, "high", {"format": "ogg", "quality": 10}, "ogg"
    )
    assert high.stat().st_size > 1.5 * low.stat().st_size
    tags = OggVorbis(str(high))
    assert tags["title"] == ["Song"]
    assert tags["artist"] == ["Band"]


def test_flac_export_accepts_a_compression_level(tmp_path, source_wav):
    output = _export(
        tmp_path,
        source_wav,
        "flac",
        {"format": "flac", "bit_depth": 24, "compression": 8},
        "flac",
    )
    assert sf.info(output).subtype == "PCM_24"


@needs_ffmpeg
def test_mp3_export_supports_variable_bitrate(tmp_path, source_wav):
    from mutagen.mp3 import MP3, BitrateMode

    fixed = _export(
        tmp_path, source_wav, "cbr", {"format": "mp3", "bitrate": 64}, "mp3"
    )
    variable = _export(
        tmp_path,
        source_wav,
        "vbr",
        {"format": "mp3", "vbr": True, "quality": 4},
        "mp3",
    )
    assert MP3(str(fixed)).info.bitrate_mode == BitrateMode.CBR
    assert MP3(str(variable)).info.bitrate_mode == BitrateMode.VBR


@needs_ffmpeg
def test_aac_export_writes_a_tagged_m4a_file(tmp_path, source_wav):
    from mutagen.mp4 import MP4

    output = _export(
        tmp_path, source_wav, "aac", {"format": "m4a", "bitrate": 128}, "m4a"
    )
    audio = MP4(str(output))
    assert audio.info.length == pytest.approx(1.0, abs=0.1)
    assert audio["\xa9nam"] == ["Song"]
    assert audio["\xa9ART"] == ["Band"]


@pytest.mark.parametrize(
    "config",
    [
        {"format": "mp3", "bitrate": 32},
        {"format": "mp3", "bitrate": "high"},
        {"format": "ogg", "quality": 11},
        {"format": "mp3", "vbr": True, "quality": -1},
        {"format": "flac", "compression": 9},
    ],
)
def test_encoder_settings_outside_the_documented_range_are_rejected(config):
    with pytest.raises(ValueError, match="Invalid"):
        validate_export_preset(config)


class _FakeLink:
    def __init__(self, bpm, _loop):
        self.tempo = bpm
        self.enabled = False
        self.num_peers = 0

    def set_tempo_callback(self, callback):
        self.callback = callback


def test_link_tempo_arrives_as_a_two_decimal_bpm():
    received = []
    holder = {}

    def factory(bpm, loop):
        holder["link"] = _FakeLink(bpm, loop)
        return holder["link"]

    sync = LinkSync(received.append, link_factory=factory)
    try:
        assert sync.start(120.0)
        holder["link"].callback(98.69995706551867)
        holder["link"].callback(123.4)
    finally:
        sync.stop()
    assert received == [98.7, 123.4]


@pytest.fixture
def restored_root():
    root = logging.getLogger()
    saved, level = list(root.handlers), root.level
    yield root
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in saved:
        root.addHandler(handler)
    root.setLevel(level)


def test_console_is_json_even_after_a_library_logged_first(
    monkeypatch, capsys, restored_root
):
    for handler in list(restored_root.handlers):
        restored_root.removeHandler(handler)
    implicit = logging.StreamHandler()
    implicit.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    restored_root.addHandler(implicit)
    monkeypatch.setenv("KORSHMER_LOG_FORMAT", "json")
    logging_config.setup_logging()
    logging.getLogger("korshmer.demo").warning("careful")
    captured = capsys.readouterr()
    lines = captured.out.strip().splitlines()
    assert implicit not in restored_root.handlers
    assert "careful" not in captured.err
    assert json.loads(lines[-1])["message"] == "careful"
    assert all(isinstance(json.loads(line), dict) for line in lines)
