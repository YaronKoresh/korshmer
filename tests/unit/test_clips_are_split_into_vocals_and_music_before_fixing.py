import numpy as np
import pytest
import soundfile as sf

from korshmer.audio_processing import clip_stems

RATE = 44100


def _tone(seconds, freq, level):
    time = np.arange(int(seconds * RATE)) / RATE
    return level * np.sin(2 * np.pi * freq * time)


def _write(path, left, right=None):
    right = left if right is None else right
    sf.write(path, np.stack([left, right], axis=1).astype(np.float32), RATE)
    return str(path)


@pytest.fixture
def known_voice(monkeypatch):
    voice = {}

    def separate(mix, progress=None, should_stop=None):
        return np.stack([voice["part"], voice["part"]])[:, : mix.shape[1]]

    monkeypatch.setattr(clip_stems, "vocals_of", separate)
    return voice


def test_a_song_clip_is_split_into_a_vocals_file_and_a_music_file(
    tmp_path, known_voice
):
    voice = _tone(3.0, 440.0, 0.2)
    music = _tone(3.0, 110.0, 0.3)
    known_voice["part"] = voice
    path = _write(tmp_path / "take.wav", voice + music)

    parts = clip_stems.separate_clip_file(path, folder=tmp_path)

    kinds = [kind for kind, _path in parts]
    assert kinds == [clip_stems.VOCALS, clip_stems.MUSIC]
    vocals, rate = sf.read(parts[0][1])
    instruments, _rate = sf.read(parts[1][1])
    assert rate == RATE
    assert np.allclose(vocals[:, 0], voice, atol=1e-4)
    assert np.allclose(instruments[:, 0], music, atol=1e-4)


def test_a_clip_with_only_singing_keeps_only_the_vocals(tmp_path, known_voice):
    voice = _tone(3.0, 440.0, 0.2)
    known_voice["part"] = voice
    path = _write(tmp_path / "sung.wav", voice + _tone(3.0, 110.0, 0.0002))

    parts = clip_stems.separate_clip_file(path, folder=tmp_path)

    assert [kind for kind, _path in parts] == [clip_stems.VOCALS]


def test_a_clip_with_only_instruments_keeps_only_the_music(
    tmp_path, known_voice
):
    music = _tone(3.0, 110.0, 0.3)
    known_voice["part"] = _tone(3.0, 440.0, 0.0003)
    path = _write(tmp_path / "riff.wav", music)

    parts = clip_stems.separate_clip_file(path, folder=tmp_path)

    assert [kind for kind, _path in parts] == [clip_stems.MUSIC]


def test_a_silent_clip_has_no_parts(tmp_path, known_voice):
    known_voice["part"] = np.zeros(RATE * 2)
    path = _write(tmp_path / "quiet.wav", np.zeros(RATE * 2))

    assert clip_stems.separate_clip_file(path, folder=tmp_path) == []


def test_without_a_vocal_model_the_clip_cannot_be_split(tmp_path, monkeypatch):
    monkeypatch.setattr(clip_stems, "vocals_of", lambda mix, **kw: None)
    path = _write(tmp_path / "take.wav", _tone(2.0, 440.0, 0.2))

    with pytest.raises(clip_stems.ClipSeparationUnavailable):
        clip_stems.separate_clip_file(path, folder=tmp_path)


def test_a_mono_clip_at_another_rate_is_split_at_the_separation_rate(
    tmp_path, known_voice
):
    rate = 22050
    time = np.arange(rate * 2) / rate
    sf.write(
        tmp_path / "mono.wav",
        (0.2 * np.sin(2 * np.pi * 440 * time)).astype(np.float32),
        rate,
    )
    known_voice["part"] = _tone(2.0, 440.0, 0.2)

    parts = clip_stems.separate_clip_file(
        str(tmp_path / "mono.wav"), folder=tmp_path
    )

    info = sf.info(parts[0][1])
    assert info.samplerate == RATE
    assert info.frames == pytest.approx(RATE * 2, abs=4)
