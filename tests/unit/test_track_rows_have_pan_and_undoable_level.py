import sys

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QFocusEvent, QMouseEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QStyle, QStyleOptionSlider

if not QApplication.instance():
    app = QApplication(sys.argv)

from korshmer.ui.viewmodels.editor_vm import EditorModel
from korshmer.ui.widgets.arranger.track_panel import (
    ClipRow,
    TrackPanel,
    TrackRow,
)


@pytest.fixture
def studio():
    model = EditorModel()
    track = model.add_track("Keys", program=4)
    model.add_note(track.id, 60, 0.0, 0.5)
    clip = model.add_clip("take.wav", name="Take", duration=1.0)
    panel = TrackPanel(model)
    panel.resize(700, 300)
    panel.show()
    QApplication.processEvents()
    yield panel, model, track, clip
    panel.close()


def track_row(panel):
    return panel.findChildren(TrackRow)[0]


def clip_row(panel):
    return panel.findChildren(ClipRow)[0]


def counted(signal):
    seen = []
    signal.connect(lambda *args: seen.append(args))
    return seen


def handle_centre(slider):
    option = QStyleOptionSlider()
    slider.initStyleOption(option)
    return (
        slider.style()
        .subControlRect(
            QStyle.CC_Slider, option, QStyle.SC_SliderHandle, slider
        )
        .center()
    )


def mouse(slider, kind, point, button, held):
    where = QPointF(point)
    QApplication.sendEvent(
        slider,
        QMouseEvent(
            kind,
            where,
            QPointF(slider.mapToGlobal(point)),
            button,
            held,
            Qt.NoModifier,
        ),
    )


def drag(slider, pixels, steps=3, before_release=None):
    start = handle_centre(slider)
    mouse(slider, QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton)
    for step in range(1, steps + 1):
        point = QPoint(start.x() + pixels * step // steps, start.y())
        mouse(slider, QEvent.MouseMove, point, Qt.NoButton, Qt.LeftButton)
        if before_release and step == 1:
            before_release()
    mouse(slider, QEvent.MouseButtonRelease, point, Qt.LeftButton, Qt.NoButton)


def leave(widget):
    QApplication.sendEvent(widget, QFocusEvent(QEvent.FocusOut))


def wheel_up(widget):
    centre = widget.rect().center()
    QApplication.sendEvent(
        widget,
        QWheelEvent(
            QPointF(centre),
            QPointF(widget.mapToGlobal(centre)),
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.NoButton,
            Qt.NoModifier,
            Qt.NoScrollPhase,
            False,
        ),
    )


def test_dragging_the_track_pan_control_pans_the_track(studio):
    panel, _model, track, _clip = studio
    drag(track_row(panel).pan_slider, -12)
    assert track.pan < 0.0


def test_dragging_the_clip_pan_control_pans_the_clip(studio):
    panel, _model, _track, clip = studio
    drag(clip_row(panel).pan_slider, 12)
    assert clip.pan > 0.0


def test_the_pan_control_says_how_far_left_or_right_it_is(studio):
    panel, _model, _track, _clip = studio
    slider = track_row(panel).pan_slider
    assert "centre" in slider.toolTip().lower()
    slider.setValue(-30)
    assert "30% left" in slider.toolTip()
    slider.setValue(75)
    assert "75% right" in slider.toolTip()


def test_the_pan_controls_are_named_for_screen_readers(studio):
    panel, _model, _track, _clip = studio
    assert track_row(panel).pan_slider.accessibleName() == "Pan of Keys"
    assert clip_row(panel).pan_slider.accessibleName() == "Pan of Take"


def test_one_drag_of_the_pan_is_one_undo_step(studio):
    panel, model, track, _clip = studio
    drag(track_row(panel).pan_slider, -15, steps=5)
    assert track.pan < 0.0
    assert model.undo()
    assert model.tracks[track.id].pan == 0.0
    assert not model.can_undo()


def test_one_drag_of_the_clip_level_is_one_undo_step(studio):
    panel, model, _track, clip = studio
    drag(clip_row(panel).gain_slider, 30, steps=5)
    assert clip.gain != 1.0
    assert model.undo()
    assert model.clip_by_id(clip.id).gain == 1.0
    assert not model.can_undo()


def test_two_separate_drags_are_two_undo_steps(studio):
    panel, model, track, _clip = studio
    slider = track_row(panel).pan_slider
    drag(slider, -8)
    after_first = track.pan
    drag(slider, -8)
    assert track.pan < after_first < 0.0
    assert model.undo()
    assert model.tracks[track.id].pan == after_first
    assert model.undo()
    assert model.tracks[track.id].pan == 0.0


def test_undoing_in_the_middle_of_a_drag_keeps_the_rest_undoable(studio):
    panel, model, track, _clip = studio
    drag(track_row(panel).pan_slider, -15, before_release=model.undo)
    assert model.tracks[track.id].pan < 0.0
    assert model.undo()
    assert model.tracks[track.id].pan == 0.0


def test_double_clicking_the_pan_control_centres_it_in_one_undo_step(studio):
    panel, model, track, _clip = studio
    slider = track_row(panel).pan_slider
    slider.setValue(60)
    QTest.mouseDClick(slider, Qt.LeftButton, Qt.NoModifier, QPoint(3, 5))
    assert slider.value() == 0
    assert track.pan == 0.0
    assert model.undo()
    assert model.tracks[track.id].pan == pytest.approx(0.6)


def test_a_double_click_made_of_two_real_clicks_is_still_one_undo_step(studio):
    panel, model, track, _clip = studio
    slider = track_row(panel).pan_slider
    slider.setValue(60)
    spot = QPoint(3, 5)
    left, none = Qt.LeftButton, Qt.NoButton
    mouse(slider, QEvent.MouseButtonPress, spot, left, left)
    mouse(slider, QEvent.MouseButtonRelease, spot, left, none)
    mouse(slider, QEvent.MouseButtonDblClick, spot, left, left)
    mouse(slider, QEvent.MouseButtonRelease, spot, left, none)
    assert track.pan == 0.0
    assert model.undo()
    assert model.tracks[track.id].pan == pytest.approx(0.6)


def test_arrow_keys_on_the_pan_control_add_up_to_one_undo_step(studio):
    panel, model, track, _clip = studio
    slider = track_row(panel).pan_slider
    for _press in range(3):
        QTest.keyClick(slider, Qt.Key_Right)
    assert track.pan == pytest.approx(0.15)
    assert model.undo()
    assert model.tracks[track.id].pan == 0.0
    assert not model.can_undo()


def test_coming_back_to_a_control_later_starts_a_new_undo_step(studio):
    panel, model, track, _clip = studio
    slider = track_row(panel).pan_slider
    QTest.keyClick(slider, Qt.Key_Right)
    leave(slider)
    QTest.keyClick(slider, Qt.Key_Right)
    assert model.undo()
    assert model.tracks[track.id].pan == pytest.approx(0.05)
    assert model.undo()
    assert model.tracks[track.id].pan == 0.0


def test_turning_the_wheel_on_the_clip_level_is_one_undo_step(studio):
    panel, model, _track, clip = studio
    slider = clip_row(panel).gain_slider
    wheel_up(slider)
    wheel_up(slider)
    assert clip.gain > 1.0
    assert model.undo()
    assert model.clip_by_id(clip.id).gain == 1.0
    assert not model.can_undo()


def test_typing_and_stepping_a_track_level_is_one_undo_step(studio):
    panel, model, track, _clip = studio
    box = track_row(panel).level_box
    box.selectAll()
    QTest.keyClicks(box, "65")
    QTest.keyClick(box, Qt.Key_Return)
    QTest.keyClick(box, Qt.Key_Up)
    assert track.gain == pytest.approx(0.70)
    assert model.undo()
    assert model.tracks[track.id].gain == 1.0
    assert not model.can_undo()


def test_keys_that_change_nothing_add_no_undo_step(studio):
    panel, model, _track, _clip = studio
    slider = track_row(panel).pan_slider
    slider.setValue(100)
    QTest.keyClick(slider, Qt.Key_Right)
    QTest.keyClick(slider, Qt.Key_Right)
    assert model.undo()
    assert not model.can_undo()


def test_level_and_pan_changes_do_not_ask_for_the_notes_to_be_redrawn(studio):
    panel, _model, track, clip = studio
    rebuilds = counted(panel.tracks_changed)
    mixes = counted(panel.mix_changed)
    track_row(panel).level_box.setValue(50)
    track_row(panel).pan_slider.setValue(50)
    clip_row(panel).gain_slider.setValue(50)
    clip_row(panel).pan_slider.setValue(50)
    assert (track.gain, track.pan, clip.gain, clip.pan) == (0.5,) * 4
    assert rebuilds == []
    assert len(mixes) == 4


def test_undoing_a_level_change_shows_the_old_level_again(studio):
    panel, model, _track, _clip = studio
    row = track_row(panel)
    row.level_box.setValue(65)
    row.pan_slider.setValue(50)
    model.undo()
    model.undo()
    panel.refresh()
    assert row.level_box.value() == 100
    assert row.pan_slider.value() == 0


def test_muting_a_track_keeps_the_same_rows_on_screen(studio):
    panel, _model, track, _clip = studio
    row = track_row(panel)
    row_of_clip = clip_row(panel)
    QTest.mouseClick(row.btn_mute, Qt.LeftButton)
    assert track.muted
    assert panel.findChildren(TrackRow) == [row]
    assert panel.findChildren(ClipRow) == [row_of_clip]


def test_a_change_made_elsewhere_updates_the_row_without_replacing_it(studio):
    panel, model, track, clip = studio
    row = track_row(panel)
    row_of_clip = clip_row(panel)
    model.rename_track(track.id, "Organ")
    model.set_track_program(track.id, 16)
    model.set_track_solo(track.id, True)
    model.set_clip_mute(clip.id, True)
    model.set_clip_gain(clip.id, 0.5)
    assert panel.refresh()
    assert panel.findChildren(TrackRow) == [row]
    assert row.name_edit.text() == "Organ"
    assert row.instrument_combo.currentIndex() == 16
    assert row.btn_solo.isChecked()
    assert row_of_clip.btn_mute.isChecked()
    assert row_of_clip.gain_slider.value() == 50


def test_adding_a_track_builds_a_row_and_a_pan_control_for_it(studio):
    panel, _model, _track, _clip = studio
    QTest.mouseClick(panel.btn_add_track, Qt.LeftButton)
    added = max(panel.model.tracks)
    row = next(
        row for row in panel.findChildren(TrackRow) if row.track_id == added
    )
    assert row.pan_slider.value() == 0
    assert panel.active_track_id == added
