import sys

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

if not QApplication.instance():
    app = QApplication(sys.argv)

from korshmer.ui.viewmodels.editor_vm import EditorModel
from korshmer.ui.widgets.arranger.track_panel import (
    ClipRow,
    TrackPanel,
    TrackRow,
)

CTRL = Qt.ControlModifier
SHIFT = Qt.ShiftModifier


@pytest.fixture
def studio():
    model = EditorModel()
    for number in range(5):
        track = model.add_track(f"Part {number + 1}", program=number)
        model.add_note(track.id, 60 + number, 0.0, 0.5)
    for number in range(3):
        model.add_clip(
            f"take{number}.wav", name=f"Take {number + 1}", duration=1.0
        )
    panel = TrackPanel(model)
    panel.resize(760, 600)
    panel.show()
    QApplication.processEvents()
    yield panel, model
    panel.close()


def track_rows(panel):
    return panel.findChildren(TrackRow)


def clip_rows(panel):
    return panel.findChildren(ClipRow)


def click_row(row, modifiers=Qt.NoModifier):
    QTest.mouseClick(
        row, Qt.LeftButton, modifiers, QPoint(2, row.height() // 2)
    )


def counted(signal):
    seen = []
    signal.connect(lambda *args: seen.append(args))
    return seen


def counted_snapshots(model, monkeypatch):
    taken = []
    real = model.snapshot

    def snapshot(*args, **kwargs):
        taken.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(model, "snapshot", snapshot)
    return taken


def track_ids(panel, *indexes):
    rows = track_rows(panel)
    return [rows[index].track_id for index in indexes]


def test_control_click_adds_rows_to_the_selection_one_by_one(studio):
    panel, _model = studio
    rows = track_rows(panel)

    click_row(rows[0])
    click_row(rows[2], CTRL)
    click_row(rows[4], CTRL)

    assert panel.selected_track_ids() == track_ids(panel, 0, 2, 4)


def test_control_click_on_a_selected_row_takes_it_out_again(studio):
    panel, _model = studio
    rows = track_rows(panel)

    click_row(rows[0])
    click_row(rows[1], CTRL)
    click_row(rows[0], CTRL)

    assert panel.selected_track_ids() == track_ids(panel, 1)


def test_shift_click_selects_every_row_in_between_including_clips(studio):
    panel, _model = studio

    click_row(track_rows(panel)[3])
    click_row(clip_rows(panel)[1], SHIFT)

    assert panel.selected_track_ids() == track_ids(panel, 3, 4)
    assert panel.selected_clip_ids() == [
        row.clip_id for row in clip_rows(panel)[:2]
    ]


def test_a_plain_click_starts_a_new_selection(studio):
    panel, _model = studio
    rows = track_rows(panel)

    click_row(rows[0])
    click_row(rows[1], CTRL)
    click_row(rows[3])

    assert panel.selected_track_ids() == track_ids(panel, 3)


def test_control_click_on_the_name_selects_the_row_instead_of_editing(studio):
    panel, _model = studio
    rows = track_rows(panel)

    click_row(rows[0])
    name = rows[2].name_edit
    QTest.mouseClick(name, Qt.LeftButton, CTRL, QPoint(4, name.height() // 2))

    assert panel.selected_track_ids() == track_ids(panel, 0, 2)
    assert not name.hasFocus()


def test_selected_rows_are_shown_as_selected(studio):
    panel, _model = studio
    rows = track_rows(panel)

    click_row(rows[1])
    click_row(rows[2], CTRL)

    assert [row.is_selected() for row in rows] == [
        False,
        True,
        True,
        False,
        False,
    ]


def test_muting_one_selected_track_mutes_every_selected_track_and_clip(
    studio, monkeypatch
):
    panel, model = studio
    click_row(track_rows(panel)[0])
    click_row(track_rows(panel)[2], CTRL)
    click_row(clip_rows(panel)[1], CTRL)
    changes = counted(panel.tracks_changed)
    snapshots = counted_snapshots(model, monkeypatch)

    track_rows(panel)[2].btn_mute.click()

    assert [model.tracks[t].muted for t in model.tracks] == [
        True,
        False,
        True,
        False,
        False,
    ]
    assert [clip.muted for clip in model.clips] == [False, True, False]
    assert len(changes) == 1
    assert len(snapshots) == 1


def test_one_undo_brings_back_every_track_changed_together(studio):
    panel, model = studio
    for row in track_rows(panel):
        click_row(row, CTRL)

    track_rows(panel)[0].btn_mute.click()
    assert all(track.muted for track in model.tracks.values())

    model.undo()

    assert not any(track.muted for track in model.tracks.values())


def test_a_button_on_a_row_outside_the_selection_changes_only_that_row(studio):
    panel, model = studio
    click_row(track_rows(panel)[0])
    click_row(track_rows(panel)[1], CTRL)

    track_rows(panel)[4].btn_solo.click()

    assert [model.tracks[t].solo for t in model.tracks] == [
        False,
        False,
        False,
        False,
        True,
    ]


def test_solo_and_hide_reach_every_selected_track(studio):
    panel, model = studio
    click_row(track_rows(panel)[1])
    click_row(track_rows(panel)[3], SHIFT)

    track_rows(panel)[2].btn_solo.click()
    track_rows(panel)[2].btn_visible.click()

    solo = [model.tracks[t].solo for t in model.tracks]
    shown = [model.tracks[t].visible for t in model.tracks]
    assert solo == [False, True, True, True, False]
    assert shown == [True, False, False, False, True]


def test_the_delete_button_of_a_selected_row_removes_the_whole_selection(
    studio,
):
    panel, model = studio
    keep = track_ids(panel, 0, 4)
    click_row(track_rows(panel)[1])
    click_row(track_rows(panel)[3], SHIFT)
    click_row(clip_rows(panel)[0], CTRL)

    track_rows(panel)[2].btn_delete.click()

    assert list(model.tracks) == keep
    assert [clip.name for clip in model.clips] == ["Take 2", "Take 3"]
    assert panel.selected_track_ids() == []
    assert panel.selected_clip_ids() == []


def test_the_delete_key_removes_every_selected_row_in_one_undo_step(studio):
    panel, model = studio
    click_row(track_rows(panel)[0])
    click_row(track_rows(panel)[4], SHIFT)
    click_row(clip_rows(panel)[2], CTRL)

    QTest.keyClick(panel, Qt.Key_Delete)

    assert list(model.tracks) == []
    assert [clip.name for clip in model.clips] == ["Take 1", "Take 2"]
    model.undo()
    assert len(model.tracks) == 5
    assert len(model.clips) == 3


@pytest.mark.parametrize(
    ("key", "flag"), [(Qt.Key_M, "muted"), (Qt.Key_S, "solo")]
)
def test_letter_keys_switch_the_whole_selection_together(studio, key, flag):
    panel, model = studio
    click_row(track_rows(panel)[0])
    click_row(track_rows(panel)[1], CTRL)
    model.set_track_mute(track_rows(panel)[1].track_id, True)
    model.set_track_solo(track_rows(panel)[1].track_id, True)

    QTest.keyClick(panel, key)
    first = [getattr(model.tracks[t], flag) for t in track_ids(panel, 0, 1)]
    QTest.keyClick(panel, key)
    second = [getattr(model.tracks[t], flag) for t in track_ids(panel, 0, 1)]

    assert first == [True, True]
    assert second == [False, False]


def test_control_a_selects_every_track_and_clip_and_escape_clears_it(studio):
    panel, _model = studio
    click_row(track_rows(panel)[0])

    QTest.keyClick(panel, Qt.Key_A, CTRL)
    everything = (panel.selected_track_ids(), panel.selected_clip_ids())
    QTest.keyClick(panel, Qt.Key_Escape)

    assert len(everything[0]) == 5
    assert len(everything[1]) == 3
    assert panel.selected_track_ids() == []
    assert panel.selected_clip_ids() == []


def test_choosing_an_instrument_on_a_selected_track_sets_it_for_all(studio):
    panel, model = studio
    click_row(track_rows(panel)[0])
    click_row(track_rows(panel)[2], CTRL)

    track_rows(panel)[2].instrument_combo.setCurrentIndex(33)

    programs = [model.tracks[t].program for t in model.tracks]
    assert programs == [33, 1, 33, 3, 4]


def test_a_level_set_on_a_selected_track_is_set_on_every_selected_track(
    studio,
):
    panel, model = studio
    click_row(track_rows(panel)[1])
    click_row(track_rows(panel)[3], SHIFT)

    track_rows(panel)[2].level_box.setValue(50)

    gains = [round(model.tracks[t].gain, 2) for t in model.tracks]
    assert gains == [1.0, 0.5, 0.5, 0.5, 1.0]


def test_a_pan_set_on_a_selected_clip_is_set_on_every_selected_clip(studio):
    panel, model = studio
    click_row(clip_rows(panel)[0])
    click_row(clip_rows(panel)[2], CTRL)

    clip_rows(panel)[0].pan_slider.setValue(-100)

    assert [clip.pan for clip in model.clips] == [-1.0, 0.0, -1.0]


def test_the_right_click_menu_offers_actions_for_the_whole_selection(studio):
    panel, model = studio
    click_row(track_rows(panel)[0])
    click_row(track_rows(panel)[1], CTRL)
    click_row(clip_rows(panel)[0], CTRL)

    menu = panel.selection_menu()
    labels = [action.text() for action in menu.actions() if action.text()]
    remove = next(a for a in menu.actions() if a.text().startswith("Delete"))
    remove.trigger()

    assert "Mute selected" in labels
    assert "Solo selected" in labels
    assert "Hide selected tracks" in labels
    assert remove.text() == "Delete 2 tracks and 1 clip"
    assert len(model.tracks) == 3
    assert len(model.clips) == 2


def test_the_status_line_says_how_many_rows_are_selected(studio):
    panel, _model = studio
    click_row(track_rows(panel)[0])
    click_row(track_rows(panel)[1], CTRL)
    click_row(clip_rows(panel)[2], CTRL)

    assert "2 tracks and 1 clip selected" in panel.lbl_status.text()


def test_thirty_tracks_are_muted_with_a_single_redraw_of_the_editor(
    monkeypatch,
):
    model = EditorModel()
    for number in range(30):
        track = model.add_track(f"Part {number + 1}")
        model.add_note(track.id, 60, float(number), 0.5)
    panel = TrackPanel(model)
    panel.resize(760, 900)
    panel.show()
    QApplication.processEvents()
    changes = counted(panel.tracks_changed)
    snapshots = counted_snapshots(model, monkeypatch)

    QTest.keyClick(panel, Qt.Key_A, CTRL)
    QTest.keyClick(panel, Qt.Key_M)

    assert all(track.muted for track in model.tracks.values())
    assert len(changes) == 1
    assert len(snapshots) == 1
    panel.close()
