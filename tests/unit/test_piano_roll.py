import sys

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

if not QApplication.instance():
    app = QApplication(sys.argv)

from korshmer.ui.widgets.midi.piano_roll import (
    InteractivePianoRoll,
    PianoKeyItem,
)


class TestPianoRollInitialization:
    def test_default_initialization(self):
        piano_roll = InteractivePianoRoll()

        assert piano_roll.grid_size == 20
        assert piano_roll.pixels_per_second == 100.0
        assert piano_roll.tempo == 120.0
        assert piano_roll.divisions_per_octave == 12
        assert piano_roll.total_divisions == 128
        assert piano_roll.root_note == 0

    def test_scene_rect_initialization(self):
        piano_roll = InteractivePianoRoll()
        scene_rect = piano_roll.scene.sceneRect()

        assert scene_rect.width() == 5000
        assert scene_rect.height() == 128 * 20


class TestMicrotonalSupport:
    def test_set_divisions_per_octave_quartertones(self):
        piano_roll = InteractivePianoRoll()
        piano_roll.set_divisions_per_octave(24)

        assert piano_roll.divisions_per_octave == 24
        assert piano_roll.total_divisions == 256

    def test_set_divisions_per_octave_eighth_tones(self):
        piano_roll = InteractivePianoRoll()
        piano_roll.set_divisions_per_octave(48)

        assert piano_roll.divisions_per_octave == 48
        assert piano_roll.total_divisions == 512

    def test_set_divisions_invalid_value_defaults_to_12(self):
        piano_roll = InteractivePianoRoll()
        piano_roll.set_divisions_per_octave(17)

        assert piano_roll.divisions_per_octave == 12
        assert piano_roll.total_divisions == 128

    def test_scene_rect_updates_with_divisions(self):
        piano_roll = InteractivePianoRoll()
        piano_roll.set_divisions_per_octave(24)

        scene_rect = piano_roll.scene.sceneRect()
        assert scene_rect.height() == 256 * 20


class TestTempoChanges:
    def test_set_tempo_valid_value(self):
        piano_roll = InteractivePianoRoll()
        piano_roll.set_tempo(140.0)

        assert piano_roll.tempo == 140.0

    def test_set_tempo_clamps_to_min(self):
        piano_roll = InteractivePianoRoll()
        piano_roll.set_tempo(10.0)

        assert piano_roll.tempo == 20.0

    def test_set_tempo_clamps_to_max(self):
        piano_roll = InteractivePianoRoll()
        piano_roll.set_tempo(500.0)

        assert piano_roll.tempo == 300.0

    def test_set_tempo_ignores_zero(self):
        piano_roll = InteractivePianoRoll()
        initial_tempo = piano_roll.tempo
        piano_roll.set_tempo(0.0)

        assert piano_roll.tempo == initial_tempo

    def test_set_tempo_ignores_negative(self):
        piano_roll = InteractivePianoRoll()
        initial_tempo = piano_roll.tempo
        piano_roll.set_tempo(-50.0)

        assert piano_roll.tempo == initial_tempo

    def test_get_beat_width_calculation(self):
        piano_roll = InteractivePianoRoll()
        piano_roll.tempo = 120.0
        piano_roll.pixels_per_second = 100.0

        beat_width = piano_roll.get_beat_width()
        expected_width = (60.0 / 120.0) * 100.0

        assert abs(beat_width - expected_width) < 0.01

    def test_get_beat_width_different_tempo(self):
        piano_roll = InteractivePianoRoll()
        piano_roll.tempo = 60.0
        piano_roll.pixels_per_second = 100.0

        beat_width = piano_roll.get_beat_width()
        expected_width = (60.0 / 60.0) * 100.0

        assert abs(beat_width - expected_width) < 0.01


class TestPianoKeyItem:
    def test_piano_key_item_initialization(self):
        item = PianoKeyItem(
            pitch=60, x_pos=0, width=100, velocity=100, grid_x=50, grid_y=20
        )

        assert item.pitch == 60
        assert item.velocity == 100
        assert item.grid_x == 50
        assert item.grid_y == 20
        assert item.rect().width() == 100
        assert item.rect().height() == 20

    def test_piano_key_item_y_position(self):
        grid_y = 20
        pitch = 60
        item = PianoKeyItem(
            pitch=pitch,
            x_pos=0,
            width=100,
            velocity=100,
            grid_x=50,
            grid_y=grid_y,
        )

        expected_y = (127 - pitch) * grid_y
        assert abs(item.y() - expected_y) < 0.01

    def test_piano_key_item_snap_grid(self):
        item = PianoKeyItem(
            pitch=60, x_pos=0, width=100, velocity=100, grid_x=50, grid_y=20
        )

        test_point = QPointF(47, 33)
        snapped = item.snapGrid(test_point)

        assert snapped.x() == 50
        assert snapped.y() == 40

    def test_piano_key_item_snap_grid_negative_clamped(self):
        item = PianoKeyItem(
            pitch=60, x_pos=0, width=100, velocity=100, grid_x=50, grid_y=20
        )

        test_point = QPointF(-10, -10)
        snapped = item.snapGrid(test_point)

        assert snapped.x() == 0
        assert snapped.y() == 0


class TestScaleView:
    def test_set_scale_view_major(self):
        piano_roll = InteractivePianoRoll()
        piano_roll.set_scale_view("C", "Major")

        assert piano_roll.root_note == 0
        assert piano_roll.scale_intervals == [0, 2, 4, 5, 7, 9, 11]

    def test_set_scale_view_minor(self):
        piano_roll = InteractivePianoRoll()
        piano_roll.set_scale_view("A", "Minor")

        assert piano_roll.root_note == 9
        assert piano_roll.scale_intervals == [0, 2, 3, 5, 7, 8, 10]

    def test_set_scale_view_auto_defaults_to_chromatic(self):
        piano_roll = InteractivePianoRoll()
        piano_roll.set_scale_view("C", "Auto (Detect)")

        assert piano_roll.scale_intervals == [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
        ]

    def test_set_scale_view_auto_root_defaults_to_c(self):
        piano_roll = InteractivePianoRoll()
        piano_roll.set_scale_view("Auto", "Major")

        assert piano_roll.root_note == 0


class TestPlayheadUpdate:
    def test_update_playhead_disabled_when_not_monitoring(self):
        piano_roll = InteractivePianoRoll()
        piano_roll.is_monitoring = False

        initial_x = piano_roll.playhead.line().x1()
        piano_roll.update_playhead(1000)

        assert piano_roll.playhead.line().x1() == initial_x

    def test_update_playhead_enabled_when_monitoring(self):
        piano_roll = InteractivePianoRoll()
        piano_roll.is_monitoring = True

        piano_roll.update_playhead(1000)
        expected_x = 1.0 * piano_roll.pixels_per_second

        assert abs(piano_roll.playhead.line().x1() - expected_x) < 0.01


class TestVelocityBounds:
    def test_piano_key_item_clamps_high_velocity(self):
        item = PianoKeyItem(
            pitch=60, x_pos=0, width=100, velocity=200, grid_x=50, grid_y=20
        )
        assert item.velocity == 127

    def test_piano_key_item_clamps_invalid_velocity(self):
        item = PianoKeyItem(
            pitch=60,
            x_pos=0,
            width=100,
            velocity=float("nan"),
            grid_x=50,
            grid_y=20,
        )
        assert isinstance(item.velocity, int)
        assert 1 <= item.velocity <= 127

    def test_default_velocity_setter_clamps(self):
        piano_roll = InteractivePianoRoll()
        piano_roll.default_velocity = 500
        assert piano_roll.default_velocity == 127
        piano_roll.default_velocity = -10
        assert piano_roll.default_velocity == 1


class TestNoteRenderer:
    def test_tooltip_format(self):
        from korshmer.ui.widgets.midi.piano_roll_renderer import NoteRenderer

        text = NoteRenderer.note_tooltip(60, 100, "Lead")
        assert "vel 100" in text
        assert "Lead" in text
        no_track = NoteRenderer.note_tooltip(60, 100, "")
        assert "·" in no_track and no_track.count("·") == 1

    def test_note_colors_scale_with_velocity(self):
        from PySide6.QtGui import QColor

        from korshmer.ui.widgets.midi.piano_roll_renderer import NoteRenderer

        base = QColor("#1A73E8")
        low_fill, _ = NoteRenderer.note_colors(base, 1, 1.0)
        high_fill, _ = NoteRenderer.note_colors(base, 127, 1.0)
        assert high_fill.alphaF() >= low_fill.alphaF()

    def test_note_geometry_matches_legacy(self):
        from korshmer.ui.widgets.midi.piano_roll_renderer import (
            MIN_NOTE_WIDTH_PX,
            NoteRenderer,
        )

        class N:
            start = 1.0
            duration = 0.5
            pitch = 60

        x, width, y = NoteRenderer.note_geometry(N(), 100.0, 20, 12, 128)
        assert x == 100.0
        assert width == max(MIN_NOTE_WIDTH_PX, 50.0)
        assert y == (127 - 60) * 20

    def test_note_label_out_of_range_is_none(self):
        from korshmer.ui.widgets.midi.piano_roll_renderer import NoteRenderer

        assert NoteRenderer.note_label(-10000, 20) is None
