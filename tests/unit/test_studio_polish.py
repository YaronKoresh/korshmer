import os
import sys
import types

import numpy as np
import pretty_midi
import pytest
import soundfile as sf
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

if not QApplication.instance():
    app = QApplication(sys.argv)

from korshmer.audio_processing import stem_session
from korshmer.audio_processing.live_levels import (
    SILENCE_DB,
    Levels,
    loudness,
    measure_levels,
)
from korshmer.audio_processing.stem_session import (
    CLEAN_VOCALS_NAME,
    StemSession,
    cleaned_vocals_path,
)
from korshmer.midi.tempo_map import TempoMap
from korshmer.mixins.actions import core as core_actions
from korshmer.mixins.actions.core import AB_NO_SOURCE, CoreActionsMixin
from korshmer.ui.managers import tutorial
from korshmer.ui.managers.tutorial import TutorialManager, TutorialStep
from korshmer.ui.tool_index import ToolEntry, search_tools
from korshmer.ui.viewmodels.editor_vm import EditorModel
from korshmer.ui.widgets.arranger import timeline
from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline
from korshmer.ui.widgets.arranger.track_panel import (
    ClipRow,
    TrackPanel,
    TrackRow,
)
from korshmer.ui.widgets.audio.live_meters import LiveMeters, format_reading

SR = 22050


def _tone(seconds=2.0, level=0.5, rate=SR):
    t = np.arange(int(seconds * rate)) / rate
    return (level * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def _session(tmp_path):
    stems = tmp_path / "song" / "Stems"
    stems.mkdir(parents=True)
    vocals = stems / "vocals.wav"
    sf.write(vocals, _tone(), SR)
    return StemSession(
        str(tmp_path / "song.mp3"),
        str(tmp_path / "song"),
        stems={"vocals": str(vocals)},
    )


def test_the_clean_vocal_lives_in_the_stems_folder(tmp_path):
    session = _session(tmp_path)
    path = cleaned_vocals_path(session)
    assert path == os.path.join(session.output_dir, "Stems", CLEAN_VOCALS_NAME)
    assert os.path.isfile(path)
    assert not os.path.exists(os.path.join(session.output_dir, "Editor"))
    assert stem_session.find_stem_files(
        os.path.join(session.output_dir, "Stems")
    ) == {"vocals": session.stems["vocals"]}


def test_an_old_editor_folder_is_folded_into_stems(tmp_path):
    session = _session(tmp_path)
    old_dir = tmp_path / "song" / "Editor"
    old_dir.mkdir()
    sf.write(old_dir / CLEAN_VOCALS_NAME, _tone(level=0.25), SR)
    os.utime(old_dir / CLEAN_VOCALS_NAME, None)
    path = cleaned_vocals_path(session)
    assert os.path.dirname(path).endswith("Stems")
    assert not old_dir.exists()
    assert np.abs(sf.read(path)[0]).max() == pytest.approx(0.25, abs=0.01)


def test_an_evened_clip_is_saved_beside_a_stem(tmp_path, monkeypatch):
    monkeypatch.setattr(timeline, "DATA_DIR", tmp_path / "data")
    session = _session(tmp_path)
    arranger = ArrangerTimeline()
    tempo_map = TempoMap((0.0, 1.0), (100.0, 110.0))
    stem_clip = types.SimpleNamespace(path=session.stems["vocals"], start=0.0)
    path, _ = arranger._aligned_clip(stem_clip, tempo_map, 105.0)
    assert os.path.dirname(path) == os.path.dirname(session.stems["vocals"])
    assert os.path.basename(path).startswith("vocals_even_")
    loose = tmp_path / "loose.wav"
    sf.write(loose, _tone(), SR)
    other = types.SimpleNamespace(path=str(loose), start=0.0)
    path, _ = arranger._aligned_clip(other, tempo_map, 105.0)
    assert os.path.dirname(path).endswith(timeline.ALIGNED_CLIPS_DIR)
    assert arranger.aligned_song_audio(str(loose)) == str(loose)


def _model():
    model = EditorModel(tempo=120.0)
    track = model.add_track("Bass", program=33)
    model.add_note(track.id, 40, 0.0, 0.5, 80)
    model.add_note(track.id, 43, 0.5, 0.5, 100)
    return model, track


def test_a_track_level_scales_its_exported_velocities():
    model, track = _model()
    model.set_track_gain(track.id, 0.5)
    notes = model.to_pretty_midi().instruments[0].notes
    assert [n.velocity for n in notes] == [40, 50]
    assert [n.velocity for n in model.notes] == [80, 100]
    model.set_track_gain(track.id, 9.0)
    assert track.gain == 2.0
    assert (
        max(n.velocity for n in model.to_pretty_midi().instruments[0].notes)
        == 127
    )
    model.set_track_gain(track.id, "loud")
    assert track.gain == 1.0
    model.set_track_gain(track.id, 0.75)
    restored = EditorModel().load_state(model.to_dict())
    assert next(iter(restored.tracks.values())).gain == 0.75
    assert (
        EditorModel().load_state({"tracks": [{"id": 1}]}).tracks[1].gain == 1.0
    )


def test_levels_can_be_typed_or_stepped_from_the_keyboard(tmp_path):
    model, track = _model()
    wav = tmp_path / "clip.wav"
    sf.write(wav, _tone(), SR)
    clip = model.add_clip(str(wav), name="Vocals", duration=2.0)
    panel = TrackPanel(model)
    panel.refresh()
    row = panel.findChildren(TrackRow)[0]
    assert row.level_box.value() == 100
    assert row.level_box.suffix() == "%"
    row.level_box.setValue(65)
    assert track.gain == pytest.approx(0.65)
    row.level_box.stepUp()
    assert track.gain == pytest.approx(0.70)

    clip_row = panel.findChildren(ClipRow)[0]
    clip_row.level_box.setValue(37)
    assert clip.gain == pytest.approx(0.37)
    assert clip_row.gain_slider.value() == 37
    clip_row.gain_slider.setValue(80)
    assert clip_row.level_box.value() == 80
    assert clip.gain == pytest.approx(0.80)


class _Player:
    def __init__(self):
        self.where = 4321
        self.played = 0

    def position(self):
        return self.where

    def playbackState(self):
        return 1

    def setPosition(self, value):
        self.where = value

    def play(self):
        self.played += 1


class _Button:
    def __init__(self):
        self.checked = False

    def isChecked(self):
        return self.checked

    def setChecked(self, value):
        self.checked = value

    def blockSignals(self, _value):
        return False


class _AbHost(CoreActionsMixin):
    def __init__(self):
        self.control_bar = types.SimpleNamespace(
            player=_Player(), btn_ab=_Button()
        )
        self.output_list = types.SimpleNamespace(currentItem=lambda: None)
        self.file_list_paths = []


@pytest.fixture
def ab(tmp_path, monkeypatch):
    events = []
    shown = []
    monkeypatch.setattr(
        core_actions,
        "publish",
        lambda event, **data: events.append((event, data)),
    )
    monkeypatch.setattr(
        QMessageBox, "information", lambda *args: shown.append(args[2])
    )
    monkeypatch.setattr(core_actions, "TEMP_DIR", tmp_path / "temp")
    host = _AbHost()
    host.events, host.shown = events, shown
    return host


def _loads(host):
    return [
        data["path"]
        for event, data in host.events
        if event == core_actions.Event.AUDIO_LOAD_REQUESTED
    ]


def test_ab_without_a_song_says_what_to_do(ab):
    ab.control_bar.btn_ab.setChecked(True)
    ab.toggle_ab_compare()
    assert ab.shown == [AB_NO_SOURCE]
    assert ab.control_bar.btn_ab.isChecked() is False
    assert _loads(ab) == []


def test_ab_switches_between_the_loaded_song_and_the_editor(ab, tmp_path):
    session = _session(tmp_path)
    source = tmp_path / "song.mp3"
    source.write_bytes(b"x")
    ab.current_session = session
    ab.control_bar.btn_ab.setChecked(True)
    ab.toggle_ab_compare()
    assert _loads(ab) == [str(source)]
    assert ab.control_bar.player.where == 4321
    assert ab.control_bar.player.played == 1

    cut = tmp_path / "song" / "music_only.wav"
    sf.write(cut, _tone(), SR)
    session.music_cut = {"source": "music_only.wav", "timeline": []}
    ab.toggle_ab_compare()
    assert _loads(ab)[-1] == str(cut)

    ab.control_bar.btn_ab.setChecked(False)
    ab.toggle_ab_compare()
    assert len(_loads(ab)) == 2
    preview = tmp_path / "temp" / "editor_preview.wav"
    preview.parent.mkdir()
    sf.write(preview, _tone(), SR)
    ab.toggle_ab_compare()
    assert _loads(ab)[-1] == str(preview)


def test_live_levels_match_known_signals():
    stereo = np.stack([_tone(3.0, 0.5, 48000)] * 2, axis=1)
    levels = measure_levels(stereo, 48000, len(stereo))
    assert levels.peak_db == pytest.approx(-6.02, abs=0.05)
    assert levels.rms_db == pytest.approx(-9.03, abs=0.05)
    assert levels.crest_db == pytest.approx(3.01, abs=0.05)
    assert levels.correlation == pytest.approx(1.0)
    assert abs(levels.momentary_lufs - levels.short_term_lufs) < 0.3
    assert not levels.clipping
    assert loudness(_tone(3.0, 0.5, 48000), 48000) < levels.short_term_lufs
    silent = measure_levels(np.zeros((48000, 2)), 48000, 48000)
    assert silent.peak_db == silent.momentary_lufs == SILENCE_DB
    assert measure_levels(None, 48000, 0) == Levels()
    assert measure_levels(stereo, 48000, 0) == Levels()
    hot = measure_levels(np.ones((48000, 2)), 48000, 48000)
    assert hot.clipping
    wide = np.stack([_tone(1.0), -_tone(1.0)], axis=1)
    assert measure_levels(wide, SR, len(wide)).correlation == pytest.approx(
        -1.0
    )


def test_loudness_agrees_with_the_reference_meter():
    pyln = pytest.importorskip("pyloudnorm")
    noise = 0.1 * np.random.default_rng(0).standard_normal((44100 * 3, 2))
    expected = pyln.Meter(44100).integrated_loudness(noise)
    assert loudness(noise, 44100) == pytest.approx(expected, abs=0.2)


def test_meters_only_follow_audio_that_is_playing():
    meters = LiveMeters()
    stereo = np.stack([_tone(1.0)] * 2, axis=1)
    engine = types.SimpleNamespace(
        data=stereo, fs=SR, idx=len(stereo), playbackState=lambda: 0
    )
    meters.set_engine(engine)
    meters.timer.stop()
    meters.refresh()
    assert meters.values["peak_db"].text() == "-inf"
    engine.playbackState = lambda: 1
    meters.refresh()
    assert meters.values["peak_db"].text() == "-6.0"
    assert meters.values["correlation"].text() == "+1.00"
    assert 0 < meters.bar.value() <= 100
    assert format_reading("rms_db", SILENCE_DB) == "-inf"
    assert set(meters.values) >= {"momentary_lufs", "short_term_lufs", "rms_db"}


class _Window(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(900, 600)
        self.flag = False
        self.target = QWidget(self)
        self.target.setGeometry(40, 50, 120, 30)


def test_a_tutorial_step_waits_until_the_user_did_it():
    window = _Window()
    window.show()
    manager = TutorialManager(window)
    manager.tutorials["t"] = [
        TutorialStep("Intro", "Read this."),
        TutorialStep(
            "Do it",
            "Then continue.",
            target_widget_name="target",
            task="set the flag.",
            begin=lambda w: w.flag,
            check=lambda w, was: w.flag != was,
        ),
    ]
    manager.start_tutorial("t")
    overlay = manager.overlay
    assert overlay.next_btn.isEnabled()
    overlay.next_btn.click()
    assert manager.current_step_index == 1
    assert not overlay.next_btn.isEnabled()
    assert "set the flag" in overlay.task_label.text()
    assert overlay.highlight_rect.contains(window.target.geometry())
    assert not overlay.mask().contains(window.target.geometry().center())
    manager._next_step()
    manager._watch_step()
    assert manager.current_step_index == 1
    window.flag = True
    manager._watch_step()
    assert overlay.next_btn.isEnabled()
    assert overlay.task_label.text().startswith("✓")
    overlay.next_btn.click()
    assert not manager.is_tutorial_active()
    manager.timer.stop()


def test_the_tutorial_panel_grows_to_fit_its_text_and_uses_app_colours():
    window = _Window()
    window.show()
    manager = TutorialManager(window)
    manager.tutorials["t"] = [TutorialStep("Long", "Many words here. " * 40)]
    manager.start_tutorial("t")
    overlay = manager.overlay
    label = overlay.desc_label
    assert label.height() >= label.heightForWidth(label.width())
    assert overlay.info_panel.height() > 200
    assert "#00FFFF" not in tutorial.PANEL_STYLE.upper()
    assert tutorial.MAT_PRIMARY in tutorial.PANEL_STYLE
    intro = manager.tutorials["intro"]
    assert sum(step.interactive for step in intro) >= 4
    assert all(step.task for step in intro if step.interactive)
    manager._close_tutorial()


def test_search_drops_weak_help_matches_when_a_name_matches():
    entries = [
        ToolEntry("A", "Edit", "Transpose", "", 0),
        ToolEntry("A", "Edit", "Loop", "works on any track", 0),
    ]
    assert [e.name for e in search_tools(entries, "tra")] == ["Transpose"]
    assert [e.name for e in search_tools(entries, "works")] == ["Loop"]
    assert Qt.AccessibleTextRole is not None
    assert pretty_midi.Note(1, 1, 0, 1).velocity == 1
