import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from korshmer import constants
from korshmer.audio_processing.save_audio import save_audio
from korshmer.plugins.builtin import SmartMasteringPlugin

RATE = 44100


class PassThroughMastering:
    def process(self, y, sr):
        return y * 0.5, sr


@pytest.fixture
def tone():
    t = np.arange(RATE // 10) / RATE
    return 0.5 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def song(tmp_path, tone):
    path = tmp_path / "song.wav"
    sf.write(str(path), np.column_stack([tone, tone]), RATE, subtype="FLOAT")
    return path


@pytest.fixture
def quick_mastering(monkeypatch):
    monkeypatch.setattr(
        "korshmer.audio_processing.SmartMastering", PassThroughMastering
    )


@pytest.mark.parametrize(
    "first_import",
    [
        "import korshmer.audio_processing.save_audio",
        "from korshmer.audio_processing.save_audio import save_audio",
        "import korshmer.audio_processing",
    ],
)
def test_the_package_always_hands_out_the_save_function_not_its_module(
    first_import,
):
    script = (
        f"{first_import}\n"
        "from korshmer.audio_processing import save_audio\n"
        "import inspect\n"
        "raise SystemExit(0 if inspect.isfunction(save_audio) else 3)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr


def test_smart_mastering_writes_the_wav_file_it_says_it_wrote(
    tmp_path, song, quick_mastering
):
    requested = tmp_path / "mastered.wav"

    written = SmartMasteringPlugin().process(str(song), str(requested))

    info = sf.info(str(requested))
    assert written == str(requested)
    assert not (tmp_path / "mastered.mp3").exists()
    assert (info.format, info.channels, info.frames) == ("WAV", 2, RATE // 10)


def test_smart_mastering_writes_an_mp3_when_an_mp3_is_asked_for(
    tmp_path, song, quick_mastering
):
    requested = tmp_path / "mastered.mp3"

    written = SmartMasteringPlugin().process(str(song), str(requested))

    assert written == str(requested)
    assert sf.info(str(requested)).format == "MP3"


def test_saving_reports_the_file_that_was_really_written(tmp_path, tone):
    written = save_audio(str(tmp_path / "take.mid"), tone, file_format="WAV")

    assert written == str(tmp_path / "take.wav")
    assert (tmp_path / "take.wav").exists()


@pytest.mark.parametrize("bits", [16, 32, 64])
def test_an_ordinary_wav_is_a_plain_wav_any_tool_can_open(tmp_path, tone, bits):
    path = tmp_path / "take.wav"

    save_audio(str(path), tone, file_format="WAV", bits=bits)

    assert path.read_bytes()[:4] == b"RIFF"
    assert sf.info(str(path)).format == "WAV"


def test_audio_beyond_the_wav_size_limit_falls_back_to_rf64(
    tmp_path, tone, monkeypatch
):
    monkeypatch.setattr(constants, "WAV_MAX_DATA_BYTES", 1000)
    path = tmp_path / "huge.wav"

    save_audio(str(path), tone, file_format="WAV", bits=32)

    assert sf.info(str(path)).format == "RF64"
