import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from korshmer.audio.loader import check_audio_file_type, usable_audio_path
from korshmer.services.file_management import FileManagementService

SR = 44100

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not installed"
)


def tone():
    t = np.arange(SR) / SR
    return (0.4 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


@pytest.mark.parametrize(
    "name", ["song.m4a", "song.aac", "song.au", "SONG.M4A", "Song.Aac"]
)
def test_aac_and_au_files_are_accepted_by_the_file_list(name):
    assert FileManagementService.is_valid_audio(name)
    assert FileManagementService.is_valid_file(name)


@pytest.mark.parametrize("name", ["song.wma", "song.mp4", "song.txt", "m4a"])
def test_formats_the_decoder_cannot_promise_stay_out_as_a_safeguard(name):
    assert not FileManagementService.is_valid_audio(name)


@needs_ffmpeg
@pytest.mark.parametrize("name", ["tone.m4a", "tone.aac"])
def test_an_accepted_aac_file_becomes_playable_audio(tmp_path, name):
    source = tmp_path / "source.wav"
    sf.write(source, tone(), SR)
    path = tmp_path / name
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(source), path],
        check=True,
    )
    assert FileManagementService.is_valid_audio(str(path))
    assert check_audio_file_type(str(path))
    info = sf.info(usable_audio_path(str(path), tmp_path / "converted"))
    assert info.samplerate == SR
    assert info.duration == pytest.approx(1.0, abs=0.1)


def test_an_accepted_au_file_is_read_directly(tmp_path):
    path = tmp_path / "tone.au"
    sf.write(path, tone(), SR, format="AU", subtype="PCM_16")
    assert FileManagementService.is_valid_audio(str(path))
    assert check_audio_file_type(str(path))
    assert usable_audio_path(str(path), tmp_path / "converted") == str(path)
