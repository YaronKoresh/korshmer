import os
import sys

import numpy as np
import pytest
import soundfile as sf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

if not QApplication.instance():
    app = QApplication(sys.argv)

from korshmer.audio_processing import clip_polish, clip_stems
from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline

RATE = 44100


def _tone(seconds, freq, level):
    time = np.arange(int(seconds * RATE)) / RATE
    return level * np.sin(2 * np.pi * freq * time)


@pytest.fixture
def editor(tmp_path, monkeypatch):
    monkeypatch.setattr(clip_polish, "DATA_DIR", tmp_path / "data")
    arranger = ArrangerTimeline()
    yield arranger
    _settle(arranger)
    arranger.wait_for_waveforms(20000)
    arranger.close()


@pytest.fixture
def voice_model(monkeypatch):
    voice = {"part": None, "calls": 0}

    def separate(mix, progress=None, should_stop=None):
        voice["calls"] += 1
        part = voice["part"]
        return np.stack([part, part])[:, : mix.shape[1]]

    monkeypatch.setattr(clip_stems, "vocals_of", separate)
    return voice


def _clip(editor, tmp_path, audio, name="take", kind="audio"):
    path = tmp_path / f"{name}.wav"
    stereo = np.stack([audio, audio], axis=1).astype(np.float32)
    sf.write(path, stereo, RATE)
    return editor.add_audio_clip(str(path), 1.0, name=name.title(), kind=kind)


def _settle(editor):
    for _ in range(50):
        editor.wait_for_clip_edits(20000)
        QApplication.processEvents()
        if not editor.clip_edit_running():
            QApplication.processEvents()
            if not editor.clip_edit_running():
                return


def test_fixing_a_song_clip_turns_it_into_a_vocals_clip_and_a_music_clip(
    editor, tmp_path, voice_model
):
    voice = _tone(3.0, 440.0, 0.2)
    voice_model["part"] = voice
    _clip(editor, tmp_path, voice + _tone(3.0, 110.0, 0.3))

    editor.remove_clip_noise(editor.model.clips[0].id)
    _settle(editor)

    clips = editor.model.clips
    assert [clip.kind for clip in clips] == ["vocals", "music"]
    assert [clip.name for clip in clips] == ["Take (vocals)", "Take (music)"]
    assert all(clip.start == pytest.approx(1.0) for clip in clips)


def test_each_part_is_cleaned_with_the_chain_for_its_kind(
    editor, tmp_path, voice_model, monkeypatch
):
    voice = _tone(3.0, 440.0, 0.2)
    voice_model["part"] = voice
    cleaned = []
    real = clip_polish.remove_noise_file

    def recording(path, folder=None, kind=clip_stems.VOCALS):
        cleaned.append(kind)
        return real(path, folder, kind)

    monkeypatch.setattr(clip_polish, "remove_noise_file", recording)
    _clip(editor, tmp_path, voice + _tone(3.0, 110.0, 0.3))

    editor.remove_clip_noise(editor.model.clips[0].id)
    _settle(editor)

    assert sorted(cleaned) == ["music", "vocals"]


def test_one_undo_brings_back_the_clip_as_it_was_before_it_was_split(
    editor, tmp_path, voice_model
):
    voice = _tone(3.0, 440.0, 0.2)
    voice_model["part"] = voice
    original = _clip(editor, tmp_path, voice + _tone(3.0, 110.0, 0.3))
    path = original.path

    editor.remove_clip_noise(original.id)
    _settle(editor)
    editor.undo()

    assert [(clip.path, clip.kind) for clip in editor.model.clips] == [
        (path, "audio")
    ]


def test_a_clip_with_only_singing_stays_one_clip_marked_as_vocals(
    editor, tmp_path, voice_model
):
    voice = _tone(3.0, 440.0, 0.2)
    voice_model["part"] = voice
    _clip(editor, tmp_path, voice, name="sung")

    editor.remove_clip_noise(editor.model.clips[0].id)
    _settle(editor)

    assert [clip.kind for clip in editor.model.clips] == ["vocals"]
    assert [clip.name for clip in editor.model.clips] == ["Sung"]


def test_a_clip_already_known_as_vocals_is_not_split_again(
    editor, tmp_path, voice_model
):
    voice_model["part"] = _tone(3.0, 440.0, 0.2)
    _clip(editor, tmp_path, _tone(3.0, 440.0, 0.2), kind="vocals")

    editor.remove_clip_noise(editor.model.clips[0].id)
    _settle(editor)

    assert voice_model["calls"] == 0
    assert [clip.kind for clip in editor.model.clips] == ["vocals"]


def test_splitting_and_cleaning_tell_the_musician_nothing_extra(
    editor, tmp_path, voice_model
):
    voice = _tone(3.0, 440.0, 0.2)
    voice_model["part"] = voice
    said = []
    editor.clip_edit_status.connect(said.append)
    _clip(editor, tmp_path, voice + _tone(3.0, 110.0, 0.3))

    editor.remove_clip_noise(editor.model.clips[0].id)
    _settle(editor)

    assert [clip.kind for clip in editor.model.clips] == ["vocals", "music"]
    assert not any(
        word in text.lower()
        for text in said
        for word in ("split", "model", "separation")
    )


def test_when_the_vocal_model_cannot_run_the_clip_is_simply_cleaned_whole(
    editor, tmp_path, monkeypatch
):
    monkeypatch.setattr(clip_stems, "vocals_of", lambda mix, **kw: None)
    said = []
    editor.clip_edit_status.connect(said.append)
    _clip(editor, tmp_path, _tone(3.0, 440.0, 0.2))

    editor.remove_clip_noise(editor.model.clips[0].id)
    _settle(editor)

    assert len(editor.model.clips) == 1
    assert "_clean" in os.path.basename(editor.model.clips[0].path)
    assert not any("split" in text.lower() for text in said)


def test_a_vocal_model_that_is_not_on_the_computer_yet_is_fetched(monkeypatch):
    from korshmer.audio_processing import mdx_vocals, separation_engine

    asked = []

    def ensure(*args, **kwargs):
        asked.append(kwargs.get("download", True))

    def offline(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(mdx_vocals, "ensure_mdx_model", ensure)
    monkeypatch.setattr(separation_engine, "_specialist_vocals", offline)

    found = separation_engine.vocals_of(np.zeros((2, RATE), dtype=np.float32))

    assert found is None
    assert asked == [True]


def test_the_vocals_and_the_music_of_a_take_stay_together_on_the_beat(
    editor, tmp_path, voice_model
):
    editor.model.tempo = 120.0
    time = np.arange(int(4.0 * RATE)) / RATE
    hits = (np.mod(time + 0.13, 0.5) < 0.03) * np.sin(2 * np.pi * 150 * time)
    voice = _tone(4.0, 440.0, 0.1)
    voice_model["part"] = voice
    _clip(editor, tmp_path, voice + 0.5 * hits)

    editor.fix_clip_automatically(editor.model.clips[0].id)
    _settle(editor)

    starts = {clip.kind: clip.start for clip in editor.model.clips}
    assert set(starts) == {"vocals", "music"}
    assert starts["vocals"] == pytest.approx(starts["music"], abs=1e-6)


def test_one_undo_also_forgets_that_a_sung_clip_was_marked_as_vocals(
    editor, tmp_path, voice_model
):
    voice = _tone(3.0, 440.0, 0.2)
    voice_model["part"] = voice
    original = _clip(editor, tmp_path, voice, name="sung")
    path = original.path

    editor.remove_clip_noise(original.id)
    _settle(editor)
    editor.undo()

    assert [(clip.path, clip.kind) for clip in editor.model.clips] == [
        (path, "audio")
    ]
