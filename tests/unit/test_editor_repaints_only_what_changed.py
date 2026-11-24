import numpy as np
import pretty_midi
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QFont, QPolygonF
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from korshmer.midi.tempo_map import TempoMap
from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline
from korshmer.ui.widgets.midi import (
    audio_clip_item,
    grid_painter,
    piano_roll_renderer,
)
from korshmer.ui.widgets.midi.piano_roll import (
    InteractivePianoRoll,
    PianoKeyItem,
)

SCALES = {
    "plain": (1.0, 1.0),
    "zoomed out": (0.3, 1.0),
    "zoomed in": (4.0, 1.5),
    "rows at their smallest": (1.0, 0.001),
    "song narrower than the window": (0.05, 1.0),
}


class WatchedRoll(InteractivePianoRoll):
    def __init__(self):
        super().__init__()
        self.background_widths = []

    def drawBackground(self, painter, rect):
        self.background_widths.append(rect.width() * self.transform().m11())
        super().drawBackground(painter, rect)


def song(notes=8):
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    piano = pretty_midi.Instrument(program=0, name="Piano")
    for index in range(notes):
        piano.notes.append(
            pretty_midi.Note(
                velocity=90,
                pitch=60 + index % 5,
                start=index * 0.5,
                end=index * 0.5 + 0.4,
            )
        )
    midi.instruments.append(piano)
    return midi


def settle():
    for _ in range(3):
        QApplication.processEvents()


def add_clip_with_waveform(roll, tmp_path, points=64):
    clip = roll.model.add_clip(
        str(tmp_path / "take.wav"), 0.5, name="Take", duration=4.0
    )
    clip.waveform = [((index * 7) % 10) / 10.0 for index in range(points)]
    roll.add_clip(clip)
    return clip


def picture(widget):
    window = widget.window()
    return window.screen().grabWindow(window.winId()).toImage()


def differing_pixels(first, second):
    assert first.size() == second.size()
    one = np.frombuffer(first.constBits(), dtype=np.uint32)
    two = np.frombuffer(second.constBits(), dtype=np.uint32)
    return int(np.count_nonzero(one != two))


def pixels_a_full_repaint_would_change(roll):
    settle()
    seen = picture(roll).copy()
    roll.viewport().update()
    settle()
    return differing_pixels(seen, picture(roll))


def repaint(roll, times=3):
    for _ in range(times):
        roll.viewport().update()
        QApplication.processEvents()


def note_items(roll):
    found = [
        item for item in roll.scene.items() if isinstance(item, PianoKeyItem)
    ]
    return sorted(found, key=lambda item: item.x())


def spot_on(roll, item):
    return roll.mapFromScene(item.sceneBoundingRect().center())


def counted(monkeypatch, module, name, base):
    built = []

    class Counted(base):
        def __init__(self, *args):
            built.append(args)
            super().__init__(*args)

    monkeypatch.setattr(module, name, Counted)
    return built


@pytest.fixture
def shown_roll(qapp):
    roll = WatchedRoll()
    roll.resize(900, 500)
    roll.load_notes(song())
    roll.show()
    settle()
    yield roll
    roll.close()


@pytest.fixture
def scaled_roll(qapp, tmp_path, request):
    roll = InteractivePianoRoll()
    roll.resize(700, 420)
    roll.load_notes(song())
    add_clip_with_waveform(roll, tmp_path)
    roll.set_scale_view("C", "Major")
    roll.show()
    settle()
    across, upright = SCALES[request.param]
    roll.zoom(across, upright)
    roll.focus_notes()
    roll.horizontalScrollBar().setValue(0)
    settle()
    yield roll
    roll.close()


def test_the_playhead_moving_does_not_recount_every_note(qapp):
    arranger = ArrangerTimeline()
    arranger.load_midi_object(song())
    settle()
    counts = []
    summary = arranger.model.summary_text

    def counting():
        counts.append(1)
        return summary()

    arranger.model.summary_text = counting
    arranger.piano_roll.is_monitoring = True

    for step in range(1, 6):
        arranger.piano_roll.update_playhead(step * 16)
        settle()

    assert counts == []


def test_loading_a_song_still_takes_the_drop_hint_away(qapp):
    arranger = ArrangerTimeline()
    arranger.show()
    settle()
    assert arranger.lbl_placeholder.isVisible()

    arranger.load_midi_object(song())
    settle()

    assert not arranger.lbl_placeholder.isVisible()
    arranger.close()


def test_placing_a_note_with_a_click_still_updates_the_track_summary(qapp):
    arranger = ArrangerTimeline()
    arranger.resize(1100, 600)
    arranger.show()
    settle()
    before = arranger.track_panel.lbl_status.text()

    QTest.mouseClick(
        arranger.piano_roll.viewport(), Qt.LeftButton, pos=QPoint(200, 150)
    )
    settle()

    assert len(arranger.model.notes) == 1
    assert arranger.track_panel.lbl_status.text() != before
    assert not arranger.lbl_placeholder.isVisible()
    arranger.close()


def test_a_playhead_step_repaints_a_narrow_strip_not_the_whole_editor(
    shown_roll,
):
    shown_roll.is_monitoring = True
    shown_roll.update_playhead(1000)
    settle()
    shown_roll.background_widths.clear()

    shown_roll.update_playhead(1016)
    settle()

    assert shown_roll.background_widths
    assert max(shown_roll.background_widths) < 40


def test_scrolling_still_repaints_the_whole_editor_so_labels_stay_in_place(
    shown_roll,
):
    shown_roll.background_widths.clear()

    bar = shown_roll.horizontalScrollBar()
    bar.setValue(bar.value() + 37)
    settle()

    assert max(shown_roll.background_widths) >= (
        shown_roll.viewport().width() - 2
    )


def test_a_playhead_step_only_works_out_the_grid_lines_next_to_the_playhead(
    qapp,
):
    roll = InteractivePianoRoll()
    roll.resize(900, 500)
    roll.load_notes(song())
    roll.model.tempo_map = TempoMap((0.0, 4.0), (120.0, 90.0))
    roll.load_model()
    roll.show()
    settle()
    spans = []
    mapped_lines = roll.vm.mapped_lines

    def watching(left, right, divisions=1):
        spans.append(right - left)
        return mapped_lines(left, right, divisions)

    roll.vm.mapped_lines = watching
    roll.is_monitoring = True
    roll.update_playhead(1000)
    settle()
    spans.clear()

    roll.update_playhead(1016)
    settle()

    assert spans
    assert max(spans) < 200
    roll.close()


@pytest.mark.parametrize("scaled_roll", list(SCALES), indirect=True)
def test_selecting_moving_and_deselecting_a_note_still_leaves_a_clean_picture(
    scaled_roll,
):
    roll = scaled_roll
    view = roll.viewport()
    note = note_items(roll)[2]
    start = spot_on(roll, note)
    before = note.pos()

    QTest.mouseClick(view, Qt.LeftButton, pos=start)
    settle()
    assert note.isSelected()
    QTest.mousePress(view, Qt.LeftButton, pos=start)
    for step in range(1, 5):
        QTest.mouseMove(view, start + QPoint(18 * step, 14 * step))
        QApplication.processEvents()
    QTest.mouseRelease(view, Qt.LeftButton, pos=start + QPoint(72, 56))
    settle()
    assert note.pos() != before
    QTest.keyClick(roll, Qt.Key_Escape)
    settle()
    assert not note.isSelected()

    assert pixels_a_full_repaint_would_change(roll) == 0


@pytest.mark.parametrize("scaled_roll", list(SCALES), indirect=True)
def test_a_playhead_sweep_still_leaves_a_clean_picture(scaled_roll):
    roll = scaled_roll
    roll.follow_playhead = False
    roll.is_monitoring = True
    across_px = roll.viewport().width()
    label_column = [5 * step for step in range(1, 8)]
    rest = list(range(60, across_px, across_px // 6))

    for device_x in label_column + rest:
        scene_x = roll.mapToScene(device_x, 0).x()
        roll.update_playhead(1000.0 * scene_x / roll.pixels_per_second)
        QApplication.processEvents()

    assert pixels_a_full_repaint_would_change(roll) == 0


def test_repainting_the_grid_builds_no_new_fonts_or_colours(
    shown_roll, monkeypatch
):
    repaint(shown_roll, 1)
    fonts = counted(monkeypatch, grid_painter, "QFont", QFont)
    colours = counted(monkeypatch, grid_painter, "QColor", QColor)

    repaint(shown_roll)

    assert fonts == []
    assert colours == []


def test_repainting_the_notes_builds_no_new_fonts(shown_roll, monkeypatch):
    repaint(shown_roll, 1)
    fonts = counted(monkeypatch, piano_roll_renderer, "QFont", QFont)

    repaint(shown_roll)

    assert fonts == []


def test_repainting_audio_clips_builds_no_new_fonts(
    shown_roll, monkeypatch, tmp_path
):
    add_clip_with_waveform(shown_roll, tmp_path)
    repaint(shown_roll, 1)
    fonts = counted(monkeypatch, audio_clip_item, "QFont", QFont)

    repaint(shown_roll)

    assert fonts == []


def test_repainting_a_clip_does_not_rebuild_its_waveform_shape(
    shown_roll, monkeypatch, tmp_path
):
    add_clip_with_waveform(shown_roll, tmp_path)
    repaint(shown_roll, 1)
    shapes = counted(monkeypatch, audio_clip_item, "QPolygonF", QPolygonF)

    repaint(shown_roll)

    assert shapes == []


def test_trimming_a_clip_still_redraws_its_waveform_shape(
    shown_roll, monkeypatch, tmp_path
):
    clip = add_clip_with_waveform(shown_roll, tmp_path)
    repaint(shown_roll, 1)
    shapes = counted(monkeypatch, audio_clip_item, "QPolygonF", QPolygonF)

    assert shown_roll.trim_clip(clip.id, 1.0, 3.0)
    repaint(shown_roll, 1)

    assert shapes


def test_a_newly_measured_waveform_still_replaces_the_old_shape(
    shown_roll, monkeypatch, tmp_path
):
    clip = add_clip_with_waveform(shown_roll, tmp_path)
    repaint(shown_roll, 1)
    shapes = counted(monkeypatch, audio_clip_item, "QPolygonF", QPolygonF)

    shown_roll.model.set_clip_waveform(clip.id, [0.5] * 16)
    shown_roll.refresh_clip(clip.id)
    repaint(shown_roll, 1)

    assert shapes
