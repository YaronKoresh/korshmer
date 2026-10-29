import sys

import numpy as np
import pretty_midi
import pytest
import soundfile as sf
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

if not QApplication.instance():
    app = QApplication(sys.argv)

try:
    from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline
except Exception:
    ArrangerTimeline = None

from korshmer.ui.widgets.arranger.track_panel import TrackPanel
from korshmer.ui.widgets.midi.audio_clip_item import (
    CLIP_LANE_HEIGHT,
    AudioClipItem,
)
from korshmer.ui.widgets.midi.piano_roll import (
    InteractivePianoRoll,
    PianoKeyItem,
)


def build_midi():
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    keys = pretty_midi.Instrument(program=4, is_drum=False, name="Keys")
    keys.notes.append(pretty_midi.Note(90, 60, 0.0, 0.5))
    keys.notes.append(pretty_midi.Note(80, 64, 1.0, 1.5))
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    drums.notes.append(pretty_midi.Note(100, 36, 0.0, 0.1))
    pm.instruments.extend([keys, drums])
    return pm


def write_wav(path, seconds=1.0, rate=22050):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    sf.write(str(path), 0.5 * np.sin(2 * np.pi * 220 * t), rate)
    return str(path)


def note_items(roll):
    return [i for i in roll.scene.items() if isinstance(i, PianoKeyItem)]


def clip_items(roll):
    return [i for i in roll.scene.items() if isinstance(i, AudioClipItem)]


def key_event(key, modifiers=Qt.NoModifier):
    return QKeyEvent(QKeyEvent.KeyPress, key, modifiers)


class FakeDropEvent:
    def __init__(self, paths):
        self._mime = QMimeData()
        self._mime.setUrls([QUrl.fromLocalFile(p) for p in paths])
        self.accepted = False

    def mimeData(self):
        return self._mime

    def acceptProposedAction(self):
        self.accepted = True


class TestRootNote:
    def test_root_note_index_and_name(self):
        roll = InteractivePianoRoll()
        assert roll.root_note == 0
        assert roll.root_note_name == "C"
        roll.set_scale_view("A", "Minor")
        assert roll.root_note == 9
        assert roll.root_note_name == "A"
        assert isinstance(roll.root_note, int)
        assert isinstance(roll.root_note_name, str)


class TestLanes:
    def test_load_notes_creates_colored_lanes(self):
        roll = InteractivePianoRoll()
        roll.load_notes(build_midi())
        tracks = list(roll.model.tracks.values())
        assert [t.name for t in tracks] == ["Keys", "Drums"]
        assert len({t.color for t in tracks}) == 2
        items = note_items(roll)
        assert len(items) == 3
        colors = {t.id: t.color.lower() for t in tracks}
        for item in items:
            assert item.color.name().lower() == colors[item.track_id]
            assert item.note_id is not None
        assert roll.current_track_id == tracks[0].id
        assert roll.has_content()

    def test_load_notes_none_clears(self):
        roll = InteractivePianoRoll()
        roll.load_notes(build_midi())
        roll.load_notes(None)
        assert roll.model.notes == []
        assert note_items(roll) == []
        assert roll.playhead.scene() is roll.scene

    def test_hidden_and_muted_tracks(self):
        roll = InteractivePianoRoll()
        roll.load_notes(build_midi())
        keys, drums = list(roll.model.tracks.values())
        roll.model.set_track_visible(drums.id, False)
        roll.model.set_track_mute(keys.id, True)
        roll.load_model()
        items = note_items(roll)
        assert len(items) == 2
        assert all(item.dim_alpha < 1.0 for item in items)

    def test_legacy_items_are_adopted(self):
        roll = InteractivePianoRoll()
        roll.scene.addItem(PianoKeyItem(60, 100, 50, 80, 50, 20))
        assert roll.has_content()
        roll.sync_model_from_scene()
        assert len(roll.model.notes) == 1
        assert roll.model.notes[0].pitch == 60
        assert abs(roll.model.notes[0].start - 1.0) < 1e-9
        assert roll.model.tracks[roll.model.notes[0].track_id].name == "Lead"


class TestClipLane:
    def test_scene_rect_extends_only_with_clips(self):
        roll = InteractivePianoRoll()
        assert roll.scene.sceneRect().top() == 0
        assert roll.scene.sceneRect().height() == 128 * 20
        clip = roll.model.add_clip(
            "/tmp/vocals.wav", kind="vocals", duration=2.0
        )
        roll.add_clip(clip)
        rect = roll.scene.sceneRect()
        assert rect.top() == 0
        assert rect.height() == 128 * 20
        assert roll.has_clip_lane()
        assert roll.lane_visible()
        lane_top, lane_height, padding = roll._lane_geometry()
        assert abs(lane_height - CLIP_LANE_HEIGHT) < 1e-6
        item = clip_items(roll)[0]
        assert abs(item.rect().width() - 200) < 1e-6
        assert abs(item.y() - (lane_top + padding)) < 1e-6
        item.setSelected(True)
        assert roll.delete_selected() == 1
        assert roll.model.clips == []
        assert not roll.lane_visible()
        assert not roll.has_clip_lane()

    def test_clip_moves_horizontally_and_syncs(self):
        roll = InteractivePianoRoll()
        clip = roll.model.add_clip("/tmp/v.wav", kind="vocals", duration=1.0)
        item = roll.add_clip(clip)
        item.setPos(137.0, 500.0)
        lane_top, _lane_height, padding = roll._lane_geometry()
        assert abs(item.y() - (lane_top + padding)) < 1e-6
        assert abs(item.x() - 137.5) < 1e-6
        assert roll.sync_model_from_scene()
        assert abs(clip.start - 1.375) < 1e-9
        roll.set_snap(False)
        item.setPos(133.0, 0)
        assert abs(item.x() - 133.0) < 1e-6

    def test_double_click_toggle_mute_and_undo(self):
        roll = InteractivePianoRoll()
        clip = roll.model.add_clip("/tmp/v.wav", kind="vocals", duration=1.0)
        roll.add_clip(clip)
        changes = []
        roll.clips_changed.connect(lambda: changes.append(True))
        roll._toggle_clip_mute(clip.id)
        assert clip.muted and changes
        assert roll.undo()
        assert not roll.model.clips[0].muted

    def test_set_tempo_rescales_notes_but_not_clips(self):
        roll = InteractivePianoRoll()
        track = roll.model.add_track("Lead")
        note = roll.model.add_note(track.id, 60, 1.0, 0.5)
        clip = roll.model.add_clip("/tmp/v.wav", duration=1.0, start=1.0)
        roll.load_model()
        roll.set_tempo(60.0)
        assert abs(roll.model.notes_by_id()[note.id].start - 2.0) < 1e-9
        assert clip.start == 1.0
        assert abs(clip_items(roll)[0].x() - 100.0) < 1e-6
        assert abs(note_items(roll)[0].x() - 200.0) < 1e-6


class TestEditingOperations:
    def _roll_with_notes(self):
        roll = InteractivePianoRoll()
        roll.load_notes(build_midi())
        return roll

    def test_click_creates_note_in_current_track(self):
        roll = InteractivePianoRoll()
        note = roll._create_note_at(QPointF(100.0, (127 - 60) * 20 + 5))
        assert note.pitch == 60
        assert abs(note.start - 1.0) < 1e-9
        assert roll.model.tracks[note.track_id].name == "Lead"
        assert len(note_items(roll)) == 1
        assert note_items(roll)[0].isSelected()
        assert roll._create_note_at(QPointF(10.0, -20.0)) is None
        assert roll.undo()
        assert roll.model.notes == []

    def test_select_all_and_delete(self):
        roll = self._roll_with_notes()
        assert roll.select_all() == 3
        assert roll.delete_selected() == 3
        assert roll.model.notes == []
        assert note_items(roll) == []
        assert roll.undo()
        assert len(roll.model.notes) == 3
        assert len(note_items(roll)) == 3
        assert roll.redo()
        assert roll.model.notes == []

    def test_delete_key_handles_selection(self):
        roll = self._roll_with_notes()
        note_items(roll)[0].setSelected(True)
        roll.keyPressEvent(key_event(Qt.Key_Delete))
        assert len(roll.model.notes) == 2

    def test_duplicate_selected(self):
        roll = self._roll_with_notes()
        target = next(i for i in note_items(roll) if i.pitch == 60)
        target.setSelected(True)
        created = roll.duplicate_selected()
        assert len(created) == 1
        assert created[0].pitch == 60
        assert abs(created[0].start - 0.5) < 1e-9
        assert len(roll.model.notes) == 4
        assert [i.note_id for i in note_items(roll) if i.isSelected()] == [
            created[0].id
        ]

    def test_transpose_via_keys(self):
        roll = self._roll_with_notes()
        target = next(i for i in note_items(roll) if i.pitch == 64)
        target.setSelected(True)
        note_id = target.note_id
        roll.keyPressEvent(key_event(Qt.Key_Up))
        assert roll.model.notes_by_id()[note_id].pitch == 65
        roll.keyPressEvent(key_event(Qt.Key_Down, Qt.ShiftModifier))
        assert roll.model.notes_by_id()[note_id].pitch == 53
        assert roll.transpose_selected(0) == 0
        roll.scene.clearSelection()
        assert roll.transpose_selected(1) == 0

    def test_nudge_with_arrow_keys(self):
        roll = self._roll_with_notes()
        target = next(i for i in note_items(roll) if i.pitch == 64)
        target.setSelected(True)
        note_id = target.note_id
        roll.keyPressEvent(key_event(Qt.Key_Right))
        assert abs(roll.model.notes_by_id()[note_id].start - 1.125) < 1e-9
        roll.keyPressEvent(key_event(Qt.Key_Left))
        assert abs(roll.model.notes_by_id()[note_id].start - 1.0) < 1e-9

    def test_quantize_selected(self):
        roll = InteractivePianoRoll()
        track = roll.model.add_track("Lead")
        note = roll.model.add_note(track.id, 60, 0.13, 0.5)
        roll.load_model()
        assert roll.quantize_selected() == 1
        assert abs(roll.model.notes_by_id()[note.id].start - 0.125) < 1e-9
        note_items(roll)[0].setSelected(True)
        roll.model.notes_by_id()[note.id].start = 0.3
        roll.load_model(keep_selection=True)
        assert roll.quantize_selected(grid_divisions=1) == 1
        assert abs(roll.model.notes_by_id()[note.id].start - 0.5) < 1e-9

    def test_humanize_and_velocity(self):
        roll = self._roll_with_notes()
        assert roll.humanize_selected(0.01, 8, seed=3) == 3
        assert all(n.start >= 0 for n in roll.model.notes)
        note_items(roll)[0].setSelected(True)
        assert roll.set_selected_velocity(500) == 1
        assert max(n.velocity for n in roll.model.notes) == 127

    def test_copy_and_paste(self):
        roll = self._roll_with_notes()
        for item in note_items(roll):
            if item.track_id == roll.current_track_id:
                item.setSelected(True)
        assert roll.copy_selected() == 2
        assert roll.has_clipboard()
        created = roll.paste(at_time=4.0)
        assert [round(n.start, 3) for n in created] == [4.0, 5.0]
        assert len(roll.model.notes) == 5
        roll.keyPressEvent(key_event(Qt.Key_V, Qt.ControlModifier))
        assert len(roll.model.notes) == 7

    def test_mouse_move_commits_on_sync(self):
        roll = self._roll_with_notes()
        item = next(i for i in note_items(roll) if i.pitch == 60)
        item.setPos(50.0, (127 - 62) * 20)
        assert roll.sync_model_from_scene()
        note = roll.model.notes_by_id()[item.note_id]
        assert note.pitch == 62
        assert abs(note.start - 0.5) < 1e-9
        assert not roll.sync_model_from_scene()

    def test_select_track_notes_and_snap(self):
        roll = self._roll_with_notes()
        drums = [t for t in roll.model.tracks.values() if t.is_drum][0]
        assert roll.select_track_notes(drums.id) == 1
        roll.set_snap(False)
        assert all(not i.snap_enabled for i in note_items(roll))
        roll.set_grid_divisions(8)
        assert all(i.grid_divisions == 8 for i in note_items(roll))
        assert abs(roll.grid_step_seconds() - 0.0625) < 1e-9

    def test_ctrl_wheel_velocity_group(self):
        roll = self._roll_with_notes()
        item = note_items(roll)[0]
        before = item.velocity
        roll._wheel_velocity(item, 1)
        assert item.velocity == before + 5
        assert roll.model.notes_by_id()[item.note_id].velocity == before + 5

    def test_zoom_clamps(self):
        from korshmer.ui.widgets.midi.piano_roll import MAX_ZOOM, MIN_ZOOM

        roll = InteractivePianoRoll()
        roll.zoom(2.0, 1.0)
        assert abs(roll.transform().m11() - 2.0) < 1e-6

        roll.zoom(1e6, 1.0)
        assert abs(roll.transform().m11() - MAX_ZOOM) < 1e-6

        roll.zoom(1e-9, 1.0)
        assert abs(roll.transform().m11() - MIN_ZOOM) < 1e-6
        roll.set_playhead_time(3.0)
        assert roll.playhead.line().x1() == 300.0


class TestDropRouting:
    def test_drop_routes_by_extension(self):
        roll = InteractivePianoRoll()
        midi_paths = []
        audio_paths = []
        roll.midi_dropped.connect(midi_paths.append)
        roll.audio_dropped.connect(audio_paths.append)

        event = FakeDropEvent(["/tmp/song.MID"])
        roll.dropEvent(event)
        assert event.accepted and midi_paths == ["/tmp/song.MID"]

        event = FakeDropEvent(["/tmp/notes.txt", "/tmp/vocals.wav"])
        roll.dropEvent(event)
        assert event.accepted and audio_paths == ["/tmp/vocals.wav"]

        event = FakeDropEvent(["/tmp/readme.txt"])
        roll.dropEvent(event)
        assert not event.accepted
        assert len(midi_paths) == 1 and len(audio_paths) == 1


class TestTrackPanel:
    def test_panel_rows_and_actions(self):
        roll = InteractivePianoRoll()
        roll.load_notes(build_midi())
        panel = TrackPanel(roll.model)
        assert len(panel._rows) == 2
        changes = []
        panel.tracks_changed.connect(lambda: changes.append(True))
        active = []
        panel.active_track_changed.connect(active.append)
        keys, drums = list(roll.model.tracks.values())
        panel.set_active_track(drums.id)
        assert active == [drums.id]
        panel.set_track_mute(keys.id, True)
        assert roll.model.tracks[keys.id].muted
        panel.change_track_instrument(keys.id, 128)
        assert roll.model.tracks[keys.id].is_drum
        panel.change_track_instrument(keys.id, 7)
        assert roll.model.tracks[keys.id].program == 7
        assert not roll.model.tracks[keys.id].is_drum
        panel.cycle_track_color(keys.id)
        panel.rename_track(keys.id, "Piano")
        assert roll.model.tracks[keys.id].name == "Piano"
        new_track = panel.add_track()
        assert len(panel._rows) == 3
        panel.delete_track(new_track.id)
        assert len(panel._rows) == 2
        clip = roll.model.add_clip("/tmp/v.wav", kind="vocals", duration=1.0)
        assert panel.refresh()
        assert len(panel._clip_rows) == 2
        panel.set_clip_gain(clip.id, 1.5)
        panel.set_clip_mute(clip.id, True)
        panel.set_clip_solo(clip.id, True)
        assert clip.gain == 1.5 and clip.muted and clip.solo
        requested = []
        panel.select_track_notes_requested.connect(requested.append)
        panel.request_select_track_notes(drums.id)
        panel.request_select_track_notes(9999)
        assert requested == [drums.id]
        assert panel.active_track_id == drums.id
        panel.remove_clip(clip.id)
        assert roll.model.clips == []
        assert len(changes) >= 8
        assert roll.model.can_undo()
        assert "2 tracks" in panel.lbl_status.text()


@pytest.mark.skipif(
    ArrangerTimeline is None, reason="sounddevice backend unavailable"
)
class TestArrangerTimeline:
    def test_multitrack_midi_and_mute(self):
        arranger = ArrangerTimeline()
        arranger.load_midi_object(build_midi())
        assert len(arranger.track_panel._rows) == 2
        pm = arranger.get_midi_from_editor()
        assert [i.name for i in pm.instruments] == ["Keys", "Drums"]
        drums = [t for t in arranger.model.tracks.values() if t.is_drum][0]
        arranger.track_panel.set_track_mute(drums.id, True)
        assert [
            i.name for i in arranger.get_midi_from_editor().instruments
        ] == ["Keys"]
        assert (
            not arranger.lbl_placeholder.isVisible() or not arranger.isVisible()
        )
        arranger.clear_editor()
        assert arranger.model.notes == [] and arranger.model.tracks == {}
        assert arranger.undo()
        assert len(arranger.model.tracks) == 2

    def test_audio_clip_waveform_worker(self, tmp_path):
        path = write_wav(tmp_path / "vocals.wav", seconds=1.5)
        arranger = ArrangerTimeline()
        clip = arranger.add_audio_clip(path, 0.5, name="Vocals", kind="vocals")
        arranger.wait_for_waveforms()
        assert len(clip.waveform) == 512
        assert abs(clip.duration - 1.5) < 1e-3
        clips = arranger.get_audio_clips()
        assert clips[0]["name"] == "Vocals" and clips[0]["start"] == 0.5
        assert len(arranger.get_all_clips()) == 1
        assert len(clip_items(arranger.piano_roll)) == 1
        assert arranger.piano_roll.has_clip_lane()

    def test_project_roundtrip_and_settings(self, tmp_path):
        path = write_wav(tmp_path / "vocals.wav", seconds=0.5)
        arranger = ArrangerTimeline()
        arranger.load_midi_object(build_midi())
        arranger.add_audio_clip(path, 1.0, kind="vocals")
        arranger.wait_for_waveforms()
        arranger.set_grid("1/8")
        arranger.set_snap(False)
        assert arranger.piano_roll.grid_divisions == 2
        assert not arranger.piano_roll.snap_enabled
        assert not arranger.btn_snap.isChecked()
        assert arranger.current_grid_divisions() == 2
        data = arranger.to_project_dict()
        assert data["grid"] == "1/8" and data["snap"] is False
        assert len(data["clips"]) == 1 and len(data["notes"]) == 3

        restored = ArrangerTimeline()
        tempos = []
        restored.tempo_changed.connect(tempos.append)
        restored.load_project_dict(data)
        restored.wait_for_waveforms()
        assert [t.name for t in restored.model.tracks.values()] == [
            "Keys",
            "Drums",
        ]
        assert len(restored.model.notes) == 3
        assert restored.model.clips[0].path == path
        assert restored.combo_grid.currentText() == "1/8"
        assert not restored.piano_roll.snap_enabled
        assert tempos == [120.0]
        assert len(clip_items(restored.piano_roll)) == 1

    def test_signals_and_shortcuts_exist(self):
        arranger = ArrangerTimeline()
        fired = []
        arranger.export_mix_requested.connect(lambda: fired.append(True))
        arranger.btn_export_mix.click()
        assert fired == [True]
        clip_fired = []
        arranger.export_clip_requested.connect(lambda: clip_fired.append(True))
        arranger.btn_export_clip.click()
        assert clip_fired == [True]
        arranger.load_midi_object(build_midi())
        assert arranger.select_all() == 3
        assert arranger.copy_selected() == 3
        assert arranger.delete_selected() == 3
        assert len(arranger.paste()) == 3
        assert arranger.transpose_selected(1) == 3
        assert arranger.quantize_selected() == 3
        assert arranger.humanize_selected() == 3
        assert len(arranger.duplicate_selected()) == 3
        assert len(arranger.model.notes) == 6
        assert arranger.undo()
        assert len(arranger.model.notes) == 3
        assert arranger.redo()
        assert len(arranger.model.notes) == 6
