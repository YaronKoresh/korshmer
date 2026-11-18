import shutil
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from korshmer.audio import loader
from korshmer.audio_processing.stem_session import mix_render_with_clips
from korshmer.exceptions import AudioFormatError
from korshmer.ui.managers.context_menu import ContextMenuManager
from korshmer.ui.viewmodels.editor_vm import EditorModel
from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline

RATE = 44100
BEAT_SECONDS = 0.5


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


def _rising_tones(path, seconds=6):
    time = np.arange(RATE * seconds) / RATE
    pitch = 200.0 + 100.0 * np.floor(time)
    sf.write(str(path), 0.5 * np.sin(2 * np.pi * pitch * time), RATE)
    return str(path)


def _clicks(path, late_seconds, noise=0.0, level=0.4, seconds=12):
    audio = np.zeros(RATE * seconds, dtype=np.float32)
    burst = np.hanning(2000) * np.sin(np.arange(2000) * 0.2) * level
    for beat in range(2, int(seconds / BEAT_SECONDS) - 2):
        first = int((beat * BEAT_SECONDS + late_seconds) * RATE)
        audio[first : first + 2000] += burst
    if noise:
        audio += np.random.default_rng(3).normal(0, noise, audio.shape)
    sf.write(str(path), audio, RATE)
    return str(path)


def _voice_like(path, level=0.05, noise=0.003, seconds=6):
    time = np.arange(RATE * seconds) / RATE
    speaking = (np.floor(time * 2) % 2 == 0).astype(np.float32)
    tone = sum(
        np.sin(2 * np.pi * f * time) / n
        for n, f in enumerate((220, 440, 880, 1760), 1)
    )
    audio = level * speaking * tone
    audio += np.random.default_rng(5).normal(0, noise, audio.shape)
    sf.write(str(path), audio.astype(np.float32), RATE)
    return str(path)


def _heard_pitch(samples):
    spectrum = np.abs(np.fft.rfft(samples))
    return float(np.fft.rfftfreq(len(samples), 1 / RATE)[np.argmax(spectrum)])


def _timeline(qapp, tempo=120.0):
    timeline = ArrangerTimeline()
    timeline.resize(1100, 600)
    timeline.show()
    qapp.processEvents()
    timeline.btn_snap.setChecked(False)
    timeline.btn_even_bpm.setChecked(False)
    timeline.model.tempo = tempo
    return timeline


def _inside_edge(timeline, clip, edge):
    reach = 2.0 / timeline.piano_roll._clip_items[clip.id].pixels_per_second
    return clip.start + reach if edge == "start" else clip.end - reach


def _drag(timeline, clip, from_seconds, to_seconds):
    roll = timeline.piano_roll
    item = roll._clip_items[clip.id]
    roll.ensureVisible(item.sceneBoundingRect(), 40, 0)
    QApplication.processEvents()
    height = item.sceneBoundingRect().center().y()
    scale = item.pixels_per_second

    def spot(seconds):
        return roll.mapFromScene(QPointF(seconds * scale, height))

    first, last = spot(from_seconds), spot(to_seconds)
    surface = roll.viewport()
    QTest.mousePress(surface, Qt.LeftButton, Qt.NoModifier, first)
    for step in range(1, 6):
        QTest.mouseMove(surface, first + (last - first) * step / 5)
    QTest.mouseRelease(surface, Qt.LeftButton, Qt.NoModifier, last)


def test_cutting_the_start_of_an_audio_clip_keeps_the_rest_in_place():
    model = EditorModel()
    clip = model.add_clip("voice.wav", start=1.0, duration=6.0)

    assert model.trim_clip(clip.id, 3.0, 7.0)

    assert (clip.start, clip.duration, clip.offset) == (3.0, 4.0, 2.0)
    assert clip.length == 6.0
    assert clip.is_trimmed()


def test_cutting_the_end_of_an_audio_clip_shortens_only_the_end():
    model = EditorModel()
    clip = model.add_clip("voice.wav", start=1.0, duration=6.0)

    model.trim_clip(clip.id, 1.0, 4.5)

    assert (clip.start, clip.duration, clip.offset) == (1.0, 3.5, 0.0)


def test_a_cut_audio_clip_can_be_stretched_back_but_not_past_its_recording():
    model = EditorModel()
    clip = model.add_clip("voice.wav", start=1.0, duration=6.0)
    model.trim_clip(clip.id, 3.0, 5.0)

    model.trim_clip(clip.id, -4.0, 40.0)

    assert (clip.start, clip.duration, clip.offset) == (1.0, 6.0, 0.0)
    assert not clip.is_trimmed()


def test_a_cut_audio_clip_never_becomes_empty():
    model = EditorModel()
    clip = model.add_clip("voice.wav", start=0.0, duration=2.0)

    model.trim_clip(clip.id, 1.5, 1.5)

    assert clip.duration > 0.0


def test_cuts_of_an_audio_clip_survive_saving_and_reopening_a_song():
    model = EditorModel()
    clip = model.add_clip("voice.wav", start=1.0, duration=6.0)
    model.trim_clip(clip.id, 2.5, 5.0)

    reopened = EditorModel.from_dict(model.to_dict()).clips[0]

    assert (reopened.start, reopened.duration) == (2.5, 2.5)
    assert (reopened.offset, reopened.length) == (1.5, 6.0)


def test_undo_brings_back_the_part_cut_from_an_audio_clip():
    model = EditorModel()
    clip = model.add_clip("voice.wav", start=1.0, duration=6.0)
    model.snapshot()
    model.trim_clip(clip.id, 3.0, 7.0)

    model.undo()

    restored = model.clips[0]
    assert (restored.start, restored.duration, restored.offset) == (
        1.0,
        6.0,
        0.0,
    )


def test_the_mixed_song_plays_only_the_kept_part_of_a_cut_audio_clip(tmp_path):
    model = EditorModel()
    clip = model.add_clip(
        _rising_tones(tmp_path / "tones.wav"), start=1.0, duration=6.0
    )
    model.trim_clip(clip.id, 3.0, 5.0)

    mixed, _rate = sf.read(
        mix_render_with_clips(
            None, model.active_clips(), str(tmp_path / "mix.wav")
        )
    )
    mono = mixed.mean(axis=1) if mixed.ndim > 1 else mixed
    heard = np.flatnonzero(np.abs(mono) > 1e-3)

    assert heard[0] / RATE == pytest.approx(3.0, abs=0.02)
    assert heard[-1] / RATE == pytest.approx(5.0, abs=0.02)
    middle = mono[int(3.25 * RATE) : int(3.75 * RATE)]
    assert _heard_pitch(middle) == pytest.approx(400.0, abs=4.0)


def test_dragging_the_left_edge_of_an_audio_clip_cuts_its_start(qapp, tmp_path):
    timeline = _timeline(qapp)
    clip = timeline.add_audio_clip(_rising_tones(tmp_path / "tones.wav"), 1.0)
    timeline.wait_for_waveforms()

    _drag(timeline, clip, _inside_edge(timeline, clip, "start"), 3.0)

    assert clip.start == pytest.approx(3.0, abs=0.05)
    assert clip.offset == pytest.approx(2.0, abs=0.05)
    assert clip.end == pytest.approx(7.0, abs=0.05)


def test_dragging_the_right_edge_of_an_audio_clip_cuts_its_end(qapp, tmp_path):
    timeline = _timeline(qapp)
    clip = timeline.add_audio_clip(_rising_tones(tmp_path / "tones.wav"), 1.0)
    timeline.wait_for_waveforms()

    _drag(timeline, clip, _inside_edge(timeline, clip, "end"), 4.0)

    assert clip.start == pytest.approx(1.0)
    assert clip.end == pytest.approx(4.0, abs=0.05)
    assert clip.offset == 0.0


def test_a_closed_main_window_does_not_take_mouse_presses_meant_for_another_window(
    qapp, tmp_path, monkeypatch
):
    from PySide6.QtWidgets import QMessageBox

    from korshmer.__main__ import MainWindow

    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )
    MainWindow().close()
    timeline = _timeline(qapp)
    clip = timeline.add_audio_clip(_rising_tones(tmp_path / "tones.wav"), 1.0)
    timeline.wait_for_waveforms()

    _drag(timeline, clip, _inside_edge(timeline, clip, "end"), 4.0)

    assert clip.end == pytest.approx(4.0, abs=0.05)


def _spot(timeline, clip, seconds):
    roll = timeline.piano_roll
    item = roll._clip_items[clip.id]
    roll.ensureVisible(item.sceneBoundingRect(), 40, 0)
    QApplication.processEvents()
    height = item.sceneBoundingRect().center().y()
    return roll.mapFromScene(QPointF(seconds * item.pixels_per_second, height))


def test_two_quick_presses_on_the_edge_of_an_audio_clip_do_not_silence_it(
    qapp, tmp_path
):
    timeline = _timeline(qapp)
    clip = timeline.add_audio_clip(_rising_tones(tmp_path / "tones.wav"), 1.0)
    timeline.wait_for_waveforms()
    edge = _spot(timeline, clip, _inside_edge(timeline, clip, "end"))

    QTest.mouseDClick(
        timeline.piano_roll.viewport(), Qt.LeftButton, Qt.NoModifier, edge
    )

    assert not timeline.model.clip_by_id(clip.id).muted


def test_two_quick_presses_on_the_middle_of_an_audio_clip_still_silence_it(
    qapp, tmp_path
):
    timeline = _timeline(qapp)
    clip = timeline.add_audio_clip(_rising_tones(tmp_path / "tones.wav"), 1.0)
    timeline.wait_for_waveforms()
    middle = _spot(timeline, clip, (clip.start + clip.end) / 2)

    QTest.mouseDClick(
        timeline.piano_roll.viewport(), Qt.LeftButton, Qt.NoModifier, middle
    )

    assert timeline.model.clip_by_id(clip.id).muted


def test_dragging_the_middle_of_a_cut_audio_clip_moves_it_without_changing_the_cut(
    qapp, tmp_path
):
    timeline = _timeline(qapp)
    clip = timeline.add_audio_clip(_rising_tones(tmp_path / "tones.wav"), 1.0)
    timeline.wait_for_waveforms()
    _drag(timeline, clip, _inside_edge(timeline, clip, "start"), 3.0)
    offset, duration = clip.offset, clip.duration

    _drag(
        timeline,
        clip,
        clip.start + duration / 2,
        clip.start + duration / 2 + 1.0,
    )
    timeline.piano_roll.sync_model_from_scene()

    assert clip.start == pytest.approx(4.0, abs=0.1)
    assert (clip.offset, clip.duration) == pytest.approx((offset, duration))


def _recording_in_an_unreadable_ogg_flavour(folder):
    converter = shutil.which("ffmpeg")
    if converter is None:
        pytest.skip("FFmpeg is needed to make this kind of OGG recording")
    plain = _rising_tones(folder / "plain.wav", seconds=2)
    recording = folder / "recording.ogg"
    subprocess.run(
        [
            converter,
            "-v",
            "error",
            "-y",
            "-i",
            plain,
            "-c:a",
            "flac",
            "-f",
            "ogg",
            str(recording),
        ],
        check=True,
    )
    with pytest.raises(sf.LibsndfileError):
        sf.info(str(recording))
    return str(recording)


def test_an_ogg_recording_the_audio_reader_rejects_is_converted_and_added_as_a_clip(
    qapp, tmp_path, monkeypatch
):
    monkeypatch.setattr(loader, "DATA_DIR", tmp_path / "data")
    recording = _recording_in_an_unreadable_ogg_flavour(tmp_path)
    timeline = _timeline(qapp)

    clip = timeline.add_audio_clip(recording, 0.0)
    timeline.wait_for_waveforms()
    qapp.processEvents()

    assert clip.path.endswith(".wav")
    assert sf.info(clip.path).duration == pytest.approx(2.0, abs=0.05)
    assert clip.duration == pytest.approx(2.0, abs=0.05)


def test_a_converted_recording_is_reused_instead_of_converted_again(tmp_path):
    recording = _recording_in_an_unreadable_ogg_flavour(tmp_path)
    first = loader.usable_audio_path(recording, tmp_path / "converted")
    written = sf.info(first).frames

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(loader.shutil, "which", lambda _name: None)
        second = loader.usable_audio_path(recording, tmp_path / "converted")

    assert second == first
    assert sf.info(second).frames == written


def test_a_readable_audio_file_is_used_as_it_is(tmp_path):
    plain = _rising_tones(tmp_path / "plain.wav", seconds=1)

    assert loader.usable_audio_path(plain, tmp_path / "converted") == plain
    assert not (tmp_path / "converted").exists()


def test_a_damaged_recording_is_refused_with_advice(tmp_path):
    damaged = tmp_path / "damaged.ogg"
    damaged.write_bytes(b"not a recording at all")

    with pytest.raises(AudioFormatError) as refusal:
        loader.usable_audio_path(str(damaged), tmp_path / "converted")

    assert "damaged.ogg" in str(refusal.value)
    assert refusal.value.recovery_hint
    assert not list((tmp_path / "converted").glob("*"))


def test_an_unreadable_recording_without_a_converter_installed_says_what_to_install(
    tmp_path, monkeypatch
):
    damaged = tmp_path / "recording.ogg"
    damaged.write_bytes(b"not a recording at all")
    monkeypatch.setattr(loader.shutil, "which", lambda _name: None)

    with pytest.raises(AudioFormatError) as refusal:
        loader.usable_audio_path(str(damaged), tmp_path / "converted")

    assert "FFmpeg" in str(refusal.value)
    assert "FFmpeg" in refusal.value.recovery_hint


@pytest.mark.parametrize("ending", [".ogg", ".oga", ".opus"])
def test_recordings_with_ogg_family_endings_can_be_dropped_on_the_editor(
    ending,
):
    from korshmer.ui.widgets.midi.piano_roll import AUDIO_EXTENSIONS

    assert ending in AUDIO_EXTENSIONS


def _clip_menu_labels(timeline, clip):
    item = timeline.piano_roll._clip_items[clip.id]
    spot = timeline.piano_roll.mapFromScene(item.sceneBoundingRect().center())
    actions = timeline.clip_menu_actions(spot)
    menu = ContextMenuManager().create_piano_roll_menu(None, actions)
    return menu, [action.text() for action in menu.actions()]


def _choose(timeline, clip, label):
    menu, _labels = _clip_menu_labels(timeline, clip)
    next(
        action for action in menu.actions() if action.text() == label
    ).trigger()
    timeline.wait_for_clip_edits()
    QApplication.processEvents()


@pytest.fixture
def edited_clips_folder(tmp_path, monkeypatch):
    from korshmer.audio_processing import clip_polish

    monkeypatch.setattr(clip_polish, "DATA_DIR", tmp_path / "data")
    return tmp_path / "data"


def test_right_clicking_an_audio_clip_offers_the_automatic_fixes_first(
    qapp, tmp_path
):
    timeline = _timeline(qapp)
    clip = timeline.add_audio_clip(_clicks(tmp_path / "clicks.wav", 0.0), 0.0)

    _menu, labels = _clip_menu_labels(timeline, clip)

    assert labels[:4] == [
        "Align Clip to the Beat",
        "Remove Background Noise",
        "Improve Level and Quality",
        "Fix Clip Automatically",
    ]
    assert "Select All" in labels


def test_right_clicking_away_from_any_audio_clip_offers_no_clip_fixes(
    qapp, tmp_path
):
    timeline = _timeline(qapp)
    timeline.add_audio_clip(_clicks(tmp_path / "clicks.wav", 0.0), 0.0)

    assert timeline.clip_menu_actions(None) == []
    menu = ContextMenuManager().create_piano_roll_menu(None)
    labels = [action.text() for action in menu.actions()]
    assert "Align Clip to the Beat" not in labels


def test_aligning_a_late_audio_clip_moves_its_pulse_onto_the_beat(
    qapp, tmp_path, edited_clips_folder
):
    timeline = _timeline(qapp)
    clip = timeline.add_audio_clip(_clicks(tmp_path / "clicks.wav", 0.12), 1.0)
    path = clip.path

    _choose(timeline, clip, "Align Clip to the Beat")

    clip = timeline.model.clip_by_id(clip.id)
    assert clip.start == pytest.approx(0.88, abs=0.03)
    assert clip.path == path

    timeline.undo()
    assert timeline.model.clip_by_id(clip.id).start == pytest.approx(1.0)


def _aligned(qapp, recording, start):
    timeline = _timeline(qapp)
    clip = timeline.add_audio_clip(recording, start)
    said = []
    timeline.clip_edit_status.connect(said.append)
    _choose(timeline, clip, "Align Clip to the Beat")
    return timeline, timeline.model.clip_by_id(clip.id), said[-1]


def _speech_after_a_pause(path, pause_seconds=1.13):
    rng = np.random.default_rng(11)
    audio = np.zeros(RATE * 8, dtype=np.float32)
    position = pause_seconds
    while position < 7.0:
        length = float(rng.uniform(0.12, 0.45))
        time = np.arange(int(length * RATE)) / RATE
        word = 0.3 * np.sin(2 * np.pi * rng.uniform(140, 260) * time)
        first = int(position * RATE)
        audio[first : first + word.shape[0]] += word * np.hanning(word.shape[0])
        position += length + float(rng.uniform(0.05, 0.6))
    sf.write(str(path), audio, RATE)
    return str(path)


def _offbeat_rhythm(path, late_seconds):
    audio = np.zeros(RATE * 12, dtype=np.float32)
    burst = np.hanning(2000) * np.sin(np.arange(2000) * 0.2) * 0.4
    for half_beat in range(4, 44, 3):
        first = int((half_beat * BEAT_SECONDS / 2 + late_seconds) * RATE)
        audio[first : first + 2000] += burst
    sf.write(str(path), audio, RATE)
    return str(path)


def test_aligning_speech_without_a_steady_pulse_puts_its_first_sound_on_a_beat(
    qapp, tmp_path, edited_clips_folder
):
    recording = _speech_after_a_pause(tmp_path / "speech.wav")

    timeline, clip, said = _aligned(qapp, recording, 2.0)

    first_sound = clip.start + 1.13
    assert first_sound / BEAT_SECONDS == pytest.approx(
        round(first_sound / BEAT_SECONDS), abs=0.06
    )
    assert abs(clip.start - 2.0) <= BEAT_SECONDS / 2 + 0.01
    assert clip.start != pytest.approx(2.0, abs=0.01)
    assert "first sound" in said

    timeline.undo()
    assert timeline.model.clip_by_id(clip.id).start == pytest.approx(2.0)


def test_aligning_a_rhythm_that_falls_between_the_beats_moves_it_onto_the_half_beats(
    qapp, tmp_path, edited_clips_folder
):
    recording = _offbeat_rhythm(tmp_path / "rhythm.wav", 0.07)

    _timeline_used, clip, said = _aligned(qapp, recording, 1.0)

    assert clip.start == pytest.approx(0.93, abs=0.03)
    assert "onto the beat" in said


def test_aligning_an_audio_clip_that_is_already_in_time_says_so_and_adds_no_undo_step(
    qapp, tmp_path, edited_clips_folder
):
    recording = _clicks(tmp_path / "clicks.wav", 0.0)
    timeline = _timeline(qapp)
    clip = timeline.add_audio_clip(recording, 1.0)
    timeline.wait_for_waveforms()
    said = []
    timeline.clip_edit_status.connect(said.append)
    _choose(timeline, clip, "Align Clip to the Beat")
    settled = timeline.model.clip_by_id(clip.id).start

    _choose(timeline, clip, "Align Clip to the Beat")
    timeline.undo()

    assert "already on the beat" in said[-1]
    assert settled != 1.0
    assert timeline.model.clip_by_id(clip.id).start == pytest.approx(1.0)


def test_aligning_a_late_audio_clip_at_the_very_start_of_the_song_moves_it_to_the_next_beat(
    qapp, tmp_path, edited_clips_folder
):
    recording = _clicks(tmp_path / "clicks.wav", 0.12)

    _timeline_used, clip, said = _aligned(qapp, recording, 0.0)

    assert clip.start == pytest.approx(0.38, abs=0.03)
    assert "onto the beat" in said


def test_aligning_a_silent_audio_clip_leaves_it_alone_and_says_it_is_silent(
    qapp, tmp_path, edited_clips_folder
):
    silence = tmp_path / "silence.wav"
    sf.write(str(silence), np.zeros(RATE * 3, dtype=np.float32), RATE)

    _timeline_used, clip, said = _aligned(qapp, str(silence), 1.1)

    assert clip.start == pytest.approx(1.1)
    assert "silent" in said


def test_aligning_steady_noise_starts_it_on_a_beat(
    qapp, tmp_path, edited_clips_folder
):
    hiss = tmp_path / "hiss.wav"
    sf.write(str(hiss), np.random.default_rng(1).normal(0, 0.1, RATE * 6), RATE)

    _timeline_used, clip, said = _aligned(qapp, str(hiss), 1.1)

    assert clip.start == pytest.approx(1.0, abs=0.02)
    assert "first sound" in said


def test_removing_background_noise_quiets_the_pauses_and_keeps_the_original_file(
    qapp, tmp_path, edited_clips_folder
):
    timeline = _timeline(qapp)
    original = _voice_like(tmp_path / "voice.wav", level=0.3, noise=0.01)
    before, _rate = sf.read(original)
    clip = timeline.add_audio_clip(original, 2.0)

    _choose(timeline, clip, "Remove Background Noise")

    clip = timeline.model.clip_by_id(clip.id)
    cleaned, _rate = sf.read(clip.path)
    pause = slice(int(0.6 * RATE), int(0.9 * RATE))
    speech = slice(int(0.1 * RATE), int(0.4 * RATE))
    assert clip.path != original
    assert np.abs(cleaned[pause]).mean() < 0.6 * np.abs(before[pause]).mean()
    assert np.abs(cleaned[speech]).mean() > 0.7 * np.abs(before[speech]).mean()
    assert clip.start == 2.0
    assert np.array_equal(sf.read(original)[0], before)

    timeline.undo()
    assert timeline.model.clip_by_id(clip.id).path == original


def test_improving_a_quiet_audio_clip_makes_it_louder_without_distortion(
    qapp, tmp_path, edited_clips_folder
):
    timeline = _timeline(qapp)
    original = _voice_like(tmp_path / "voice.wav", level=0.02, noise=0.0005)
    before, _rate = sf.read(original)
    clip = timeline.add_audio_clip(original, 0.0, gain=1.8)

    _choose(timeline, clip, "Improve Level and Quality")

    clip = timeline.model.clip_by_id(clip.id)
    improved, _rate = sf.read(clip.path)
    assert np.sqrt(np.mean(improved**2)) > 2.0 * np.sqrt(np.mean(before**2))
    assert np.abs(improved).max() <= 1.0
    assert clip.gain == 1.0
    assert len(improved) == len(before)


def test_fixing_an_audio_clip_automatically_cleans_improves_and_aligns_it_in_one_undo_step(
    qapp, tmp_path, edited_clips_folder
):
    timeline = _timeline(qapp)
    original = _clicks(tmp_path / "clicks.wav", 0.12, noise=0.002, level=0.1)
    clip = timeline.add_audio_clip(original, 1.0)
    timeline.model.trim_clip(clip.id, 2.0, 9.0)
    timeline.piano_roll.load_model()

    _choose(timeline, clip, "Fix Clip Automatically")

    clip = timeline.model.clip_by_id(clip.id)
    assert clip.path != original
    assert clip.start == pytest.approx(1.88, abs=0.04)
    assert (clip.offset, clip.duration) == pytest.approx((1.0, 7.0))

    timeline.undo()
    restored = timeline.model.clip_by_id(clip.id)
    assert (restored.path, restored.start) == (original, 2.0)


def test_a_fix_that_cannot_read_the_clip_reports_the_problem_and_changes_nothing(
    qapp, tmp_path, edited_clips_folder, monkeypatch
):
    from PySide6.QtWidgets import QMessageBox

    shown = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args: shown.append(args[2])
    )
    timeline = _timeline(qapp)
    original = _clicks(tmp_path / "clicks.wav", 0.0)
    clip = timeline.add_audio_clip(original, 1.0)
    timeline.wait_for_waveforms()
    (tmp_path / "clicks.wav").unlink()

    _choose(timeline, clip, "Remove Background Noise")

    clip = timeline.model.clip_by_id(clip.id)
    assert (clip.path, clip.start) == (original, 1.0)
    assert shown and "clicks" in shown[0]
