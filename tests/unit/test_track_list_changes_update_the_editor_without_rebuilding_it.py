import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

if not QApplication.instance():
    app = QApplication(sys.argv)

from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline
from korshmer.ui.widgets.midi.audio_clip_item import AudioClipItem
from korshmer.ui.widgets.midi.piano_roll import MUTED_TRACK_ALPHA, PianoKeyItem

NOTES_PER_TRACK = 6


@pytest.fixture
def editor():
    arranger = ArrangerTimeline()
    arranger.resize(1200, 700)
    model = arranger.model
    for number in range(4):
        track = model.add_track(f"Part {number + 1}", program=number)
        for step in range(NOTES_PER_TRACK):
            model.add_note(track.id, 60 + number, step * 0.5, 0.25)
    for number in range(2):
        model.add_clip(
            f"take{number}.wav", name=f"Take {number + 1}", duration=1.0
        )
    arranger.track_panel.refresh()
    arranger.piano_roll.load_model()
    arranger.show()
    QApplication.processEvents()
    yield arranger
    arranger.close()


def track_id(editor, index):
    return list(editor.model.tracks)[index]


def note_items(editor):
    return {
        item.note_id: item
        for item in editor.piano_roll.scene.items()
        if isinstance(item, PianoKeyItem)
    }


def clip_items(editor):
    return {
        item.clip_id: item
        for item in editor.piano_roll.scene.items()
        if isinstance(item, AudioClipItem)
    }


def items_of(editor, items, index):
    wanted = track_id(editor, index)
    return [item for item in items.values() if item.track_id == wanted]


def test_muting_a_track_keeps_every_note_already_drawn_in_the_editor(editor):
    before = note_items(editor)

    editor.track_panel.set_track_mute(track_id(editor, 1), True)

    after = note_items(editor)
    assert after.keys() == before.keys()
    assert all(after[key] is before[key] for key in before)


def test_a_muted_track_is_dimmed_and_the_others_are_not(editor):
    editor.track_panel.set_track_mute(track_id(editor, 1), True)

    items = note_items(editor)
    dimmed = {item.dim_alpha for item in items_of(editor, items, 1)}
    others = {item.dim_alpha for item in items_of(editor, items, 0)}
    assert dimmed == {MUTED_TRACK_ALPHA}
    assert others == {1.0}


def test_unmuting_a_track_brings_its_notes_back_to_full_colour(editor):
    editor.track_panel.set_track_mute(track_id(editor, 2), True)
    editor.track_panel.set_track_mute(track_id(editor, 2), False)

    items = note_items(editor)
    assert {item.dim_alpha for item in items_of(editor, items, 2)} == {1.0}


def test_hiding_a_track_takes_only_its_notes_off_the_editor(editor):
    before = note_items(editor)

    editor.track_panel.set_track_visible(track_id(editor, 0), False)

    after = note_items(editor)
    assert items_of(editor, after, 0) == []
    kept = [
        key for key in before if before[key].track_id != track_id(editor, 0)
    ]
    assert sorted(after) == sorted(kept)
    assert all(after[key] is before[key] for key in kept)


def test_showing_a_hidden_track_draws_its_notes_again(editor):
    editor.track_panel.set_track_visible(track_id(editor, 3), False)
    editor.track_panel.set_track_visible(track_id(editor, 3), True)

    assert len(items_of(editor, note_items(editor), 3)) == NOTES_PER_TRACK


def test_deleting_a_track_removes_only_its_notes_from_the_editor(editor):
    before = note_items(editor)
    gone = track_id(editor, 2)

    editor.track_panel.delete_track(gone)

    after = note_items(editor)
    assert all(item.track_id != gone for item in after.values())
    assert len(after) == 3 * NOTES_PER_TRACK
    assert all(after[key] is before[key] for key in after)


def test_recolouring_a_track_recolours_its_notes(editor):
    tid = track_id(editor, 1)

    editor.track_panel.cycle_track_color(tid)

    colour = QColor(editor.model.tracks[tid].color)
    assert {
        item.color.name() for item in items_of(editor, note_items(editor), 1)
    } == {colour.name()}


def test_renaming_a_track_renames_its_notes(editor):
    tid = track_id(editor, 0)

    editor.track_panel.rename_track(tid, "Lead")

    assert {
        item.track_name for item in items_of(editor, note_items(editor), 0)
    } == {"Lead"}


def test_removing_a_clip_takes_only_that_clip_off_the_editor(editor):
    before = clip_items(editor)
    first, second = list(before)

    editor.track_panel.remove_clip(first)

    after = clip_items(editor)
    assert list(after) == [second]
    assert after[second] is before[second]


def test_notes_picked_in_the_editor_stay_picked_after_a_track_change(editor):
    items = note_items(editor)
    picked = items_of(editor, items, 3)
    for item in picked:
        item.setSelected(True)

    editor.track_panel.set_track_solo(track_id(editor, 0), True)

    assert all(
        item.isSelected() for item in items_of(editor, note_items(editor), 3)
    )


def test_undoing_a_track_change_still_shows_the_song_as_it_was(editor):
    editor.track_panel.set_track_visible(track_id(editor, 1), False)
    editor.track_panel.set_track_mute(track_id(editor, 2), True)

    editor.undo()
    editor.undo()

    items = note_items(editor)
    assert len(items) == 4 * NOTES_PER_TRACK
    assert {item.dim_alpha for item in items.values()} == {1.0}


def test_muting_thirty_tracks_together_draws_no_note_again(monkeypatch):
    arranger = ArrangerTimeline()
    model = arranger.model
    for number in range(30):
        track = model.add_track(f"Part {number + 1}")
        model.add_note(track.id, 60, float(number), 0.5)
    arranger.track_panel.refresh()
    arranger.piano_roll.load_model()
    built = []
    real = PianoKeyItem.__init__

    def counting(self, *args, **kwargs):
        built.append(1)
        real(self, *args, **kwargs)

    monkeypatch.setattr(PianoKeyItem, "__init__", counting)

    arranger.track_panel.select_all_rows()
    arranger.track_panel.switch_selection("muted")

    assert all(track.muted for track in model.tracks.values())
    assert built == []
    arranger.close()


def test_changing_an_instrument_repaints_the_out_of_range_shading(editor):
    roll = editor.piano_roll
    repaints = []
    real = roll.viewport().update

    def counting(*args):
        repaints.append(args)
        return real(*args)

    roll.viewport().update = counting
    editor.track_panel.change_track_instrument(track_id(editor, 0), 128)

    assert editor.model.tracks[track_id(editor, 0)].is_drum
    assert repaints


def test_moving_a_clip_by_a_beat_keeps_every_note_already_drawn(editor):
    before = note_items(editor)
    clip = editor.model.clips[0]
    clip.start = 1.0
    editor.piano_roll.sync_with_model()

    editor.move_clip_by_beats(clip.id, 1.0)

    after = note_items(editor)
    assert all(after[key] is before[key] for key in before)
    assert after.keys() == before.keys()


def test_a_clip_moved_by_a_beat_is_drawn_at_its_new_place(editor):
    clip = editor.model.clips[1]
    clip.start = 1.0
    editor.piano_roll.sync_with_model()
    item = clip_items(editor)[clip.id]
    before = item.x()

    editor.move_clip_by_beats(clip.id, 1.0)

    assert clip_items(editor)[clip.id].x() > before


def test_deleting_picked_notes_keeps_the_other_notes_already_drawn(editor):
    items = note_items(editor)
    picked = items_of(editor, items, 1)
    for item in picked:
        item.setSelected(True)
    kept = {key: item for key, item in items.items() if item not in picked}

    editor.piano_roll.delete_selected()

    after = note_items(editor)
    assert after.keys() == kept.keys()
    assert all(after[key] is kept[key] for key in kept)
    assert len(editor.model.notes) == 3 * NOTES_PER_TRACK


def test_undoing_a_note_deletion_draws_the_notes_again(editor):
    for item in items_of(editor, note_items(editor), 2):
        item.setSelected(True)
    editor.piano_roll.delete_selected()

    editor.undo()

    assert len(note_items(editor)) == 4 * NOTES_PER_TRACK
