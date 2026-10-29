import numpy as np

from korshmer.midi_processing.quantization import (
    AdvancedQuantizer,
    GrooveTemplate,
    Note,
    QuantizeMode,
)


class TestNote:
    def test_note_creation(self):
        note = Note(start_time=0.0, end_time=1.0, pitch=60, velocity=100)
        assert note.start_time == 0.0
        assert note.end_time == 1.0
        assert note.pitch == 60
        assert note.velocity == 100

    def test_note_duration(self):
        note = Note(start_time=0.5, end_time=1.5, pitch=60, velocity=100)
        assert note.duration == 1.0

    def test_note_zero_duration(self):
        note = Note(start_time=1.0, end_time=1.0, pitch=60, velocity=100)
        assert note.duration == 0.0


class TestGrooveTemplate:
    def test_groove_template_creation(self):
        template = GrooveTemplate(
            name="Test Groove",
            timing_offsets={0.0: 0.0, 1.0: 0.1},
            velocity_curve={0.0: 1.0},
        )
        assert template.name == "Test Groove"
        assert template.swing_amount == 0.0

    def test_apply_timing_exact_match(self):
        template = GrooveTemplate(
            name="Test", timing_offsets={0.0: 0.0, 1.0: 0.05}, velocity_curve={}
        )
        result = template.apply_timing(1.0)
        assert result == 1.05

    def test_apply_timing_interpolation(self):
        template = GrooveTemplate(
            name="Test", timing_offsets={0.0: 0.0, 1.0: 0.1}, velocity_curve={}
        )
        result = template.apply_timing(0.5)
        assert 0.54 < result < 0.56

    def test_apply_timing_outside_range(self):
        template = GrooveTemplate(
            name="Test", timing_offsets={0.0: 0.0, 1.0: 0.1}, velocity_curve={}
        )
        result = template.apply_timing(2.0)
        assert result == 2.0

    def test_apply_velocity_exact_match(self):
        template = GrooveTemplate(
            name="Test", timing_offsets={}, velocity_curve={0.0: 1.5}
        )
        result = template.apply_velocity(0.0, 80)
        assert result == 120

    def test_apply_velocity_clipping(self):
        template = GrooveTemplate(
            name="Test", timing_offsets={}, velocity_curve={0.0: 2.0}
        )
        result = template.apply_velocity(0.0, 100)
        assert result == 127

    def test_apply_velocity_no_curve(self):
        template = GrooveTemplate(
            name="Test", timing_offsets={}, velocity_curve={}
        )
        result = template.apply_velocity(0.5, 80)
        assert result == 80


class TestAdvancedQuantizerInit:
    def test_init_default_parameters(self):
        quantizer = AdvancedQuantizer()
        assert quantizer.bpm == 120.0
        assert quantizer.time_signature == (4, 4)
        assert quantizer.quantize_strength == 1.0
        assert quantizer.groove_template is None

    def test_init_custom_bpm(self):
        quantizer = AdvancedQuantizer(bpm=140.0)
        assert quantizer.bpm == 140.0

    def test_init_custom_time_signature(self):
        quantizer = AdvancedQuantizer(time_signature=(3, 4))
        assert quantizer.time_signature == (3, 4)


class TestBasicQuantization:
    def setup_method(self):
        self.quantizer = AdvancedQuantizer(bpm=120.0)

    def test_quantize_on_grid(self):
        notes = [Note(start_time=0.0, end_time=0.5, pitch=60, velocity=100)]
        result = self.quantizer.quantize_notes(notes, grid_division=0.25)

        assert len(result) == 1
        assert result[0].start_time == 0.0
        assert result[0].pitch == 60

    def test_quantize_off_grid(self):

        notes = [Note(start_time=0.52, end_time=1.02, pitch=60, velocity=100)]
        result = self.quantizer.quantize_notes(notes, grid_division=0.25)

        assert len(result) == 1

        beat_duration = 60.0 / 120.0
        grid_size = beat_duration * 0.25
        expected = round(0.52 / grid_size) * grid_size
        assert abs(result[0].start_time - expected) < 0.01

    def test_quantize_mode_nearest(self):
        notes = [Note(start_time=0.24, end_time=0.74, pitch=60, velocity=100)]
        result = self.quantizer.quantize_notes(
            notes, grid_division=0.5, mode=QuantizeMode.NEAREST
        )

        beat_duration = 60.0 / 120.0
        grid_size = beat_duration * 0.5
        expected = round(0.24 / grid_size) * grid_size
        assert abs(result[0].start_time - expected) < 0.01

    def test_quantize_mode_floor(self):
        notes = [Note(start_time=0.24, end_time=0.74, pitch=60, velocity=100)]
        result = self.quantizer.quantize_notes(
            notes, grid_division=0.5, mode=QuantizeMode.FLOOR
        )

        beat_duration = 60.0 / 120.0
        grid_size = beat_duration * 0.5
        expected = np.floor(0.24 / grid_size) * grid_size
        assert abs(result[0].start_time - expected) < 0.01

    def test_quantize_mode_ceil(self):
        notes = [Note(start_time=0.24, end_time=0.74, pitch=60, velocity=100)]
        result = self.quantizer.quantize_notes(
            notes, grid_division=0.5, mode=QuantizeMode.CEIL
        )

        beat_duration = 60.0 / 120.0
        grid_size = beat_duration * 0.5
        expected = np.ceil(0.24 / grid_size) * grid_size
        assert abs(result[0].start_time - expected) < 0.01

    def test_quantize_preserves_pitch(self):
        notes = [Note(start_time=0.1, end_time=0.6, pitch=72, velocity=100)]
        result = self.quantizer.quantize_notes(notes)
        assert result[0].pitch == 72

    def test_quantize_preserves_velocity(self):
        notes = [Note(start_time=0.1, end_time=0.6, pitch=60, velocity=80)]
        result = self.quantizer.quantize_notes(notes)
        assert result[0].velocity == 80

    def test_quantize_multiple_notes(self):
        notes = [
            Note(start_time=0.1, end_time=0.6, pitch=60, velocity=100),
            Note(start_time=0.6, end_time=1.1, pitch=64, velocity=90),
            Note(start_time=1.2, end_time=1.7, pitch=67, velocity=85),
        ]
        result = self.quantizer.quantize_notes(notes, grid_division=0.25)

        assert len(result) == 3

        for note in result:
            assert note.pitch in [60, 64, 67]

    def test_quantize_minimum_duration(self):

        notes = [Note(start_time=0.49, end_time=0.51, pitch=60, velocity=100)]
        result = self.quantizer.quantize_notes(notes, grid_division=0.25)

        assert result[0].end_time > result[0].start_time
        assert result[0].duration >= 0.01


class TestQuantizeStrength:
    def test_full_strength(self):
        quantizer = AdvancedQuantizer(bpm=120.0)
        quantizer.quantize_strength = 1.0

        notes = [Note(start_time=0.1, end_time=0.6, pitch=60, velocity=100)]
        result = quantizer.quantize_notes(notes, grid_division=0.25)

        beat_duration = 60.0 / 120.0
        grid_size = beat_duration * 0.25
        expected = round(0.1 / grid_size) * grid_size
        assert abs(result[0].start_time - expected) < 0.01

    def test_half_strength(self):
        quantizer = AdvancedQuantizer(bpm=120.0)
        quantizer.quantize_strength = 0.5

        notes = [Note(start_time=0.1, end_time=0.6, pitch=60, velocity=100)]
        result = quantizer.quantize_notes(notes, grid_division=0.25)

        assert 0.0 < result[0].start_time < 0.125

    def test_zero_strength(self):
        quantizer = AdvancedQuantizer(bpm=120.0)
        quantizer.quantize_strength = 0.0

        notes = [Note(start_time=0.1, end_time=0.6, pitch=60, velocity=100)]
        result = quantizer.quantize_notes(notes, grid_division=0.25)

        assert abs(result[0].start_time - 0.1) < 0.001


class TestHumanization:
    def test_humanize_adds_variation(self):
        quantizer = AdvancedQuantizer(bpm=120.0)

        notes = [Note(start_time=0.0, end_time=0.5, pitch=60, velocity=100)]

        results = []
        np.random.seed(42)
        for _ in range(10):
            result = quantizer.quantize_notes(
                notes, grid_division=0.25, humanize=True
            )
            results.append(result[0].start_time)

        assert len(set(results)) > 1

    def test_humanize_stays_close_to_grid(self):
        quantizer = AdvancedQuantizer(bpm=120.0)
        np.random.seed(42)

        notes = [Note(start_time=0.0, end_time=0.5, pitch=60, velocity=100)]
        result = quantizer.quantize_notes(
            notes, grid_division=0.25, humanize=True
        )

        beat_duration = 60.0 / 120.0
        grid_size = beat_duration * 0.25
        max_deviation = grid_size * 0.1

        assert abs(result[0].start_time - 0.0) < max_deviation * 2


class TestSwing:
    def test_swing_delays_offbeats(self):
        quantizer = AdvancedQuantizer(bpm=120.0)

        notes = [
            Note(start_time=0.0, end_time=0.2, pitch=60, velocity=100),
            Note(start_time=0.25, end_time=0.45, pitch=60, velocity=100),
        ]

        result = quantizer.apply_swing(notes, swing_amount=0.5)

        assert result[0].start_time == 0.0
        assert result[1].start_time > 0.25

    def test_swing_amount_zero(self):
        quantizer = AdvancedQuantizer(bpm=120.0)

        notes = [Note(start_time=0.25, end_time=0.45, pitch=60, velocity=100)]
        result = quantizer.apply_swing(notes, swing_amount=0.0)

        assert result[0].start_time == 0.25

    def test_swing_preserves_pitch(self):
        quantizer = AdvancedQuantizer(bpm=120.0)

        notes = [Note(start_time=0.25, end_time=0.45, pitch=72, velocity=80)]
        result = quantizer.apply_swing(notes, swing_amount=0.5)

        assert result[0].pitch == 72
        assert result[0].velocity == 80


class TestSelectiveQuantize:
    def test_selective_quantize_threshold(self):
        quantizer = AdvancedQuantizer(bpm=120.0)

        notes = [
            Note(start_time=0.0, end_time=0.5, pitch=60, velocity=100),
            Note(start_time=0.3, end_time=0.8, pitch=64, velocity=100),
        ]

        result = quantizer.selective_quantize(
            notes, grid_division=0.25, threshold=0.05
        )

        assert result[0].start_time == 0.0


class TestGrooveIntegration:
    def test_apply_groove_template(self):
        quantizer = AdvancedQuantizer(bpm=120.0)

        template = GrooveTemplate(
            name="Test Groove",
            timing_offsets={0.0: 0.05, 1.0: 0.1},
            velocity_curve={},
        )
        quantizer.groove_template = template

        notes = [Note(start_time=0.0, end_time=0.5, pitch=60, velocity=100)]
        result = quantizer.quantize_notes(notes, grid_division=0.25)

        assert result[0].start_time > 0.0


class TestEdgeCases:
    def test_empty_notes_list(self):
        quantizer = AdvancedQuantizer(bpm=120.0)
        result = quantizer.quantize_notes([])
        assert result == []

    def test_very_high_bpm(self):
        quantizer = AdvancedQuantizer(bpm=300.0)
        notes = [Note(start_time=0.1, end_time=0.6, pitch=60, velocity=100)]
        result = quantizer.quantize_notes(notes)

        assert len(result) == 1
        assert not np.isnan(result[0].start_time)

    def test_very_low_bpm(self):
        quantizer = AdvancedQuantizer(bpm=40.0)
        notes = [Note(start_time=0.1, end_time=0.6, pitch=60, velocity=100)]
        result = quantizer.quantize_notes(notes)

        assert len(result) == 1
        assert not np.isnan(result[0].start_time)

    def test_notes_with_negative_time(self):
        quantizer = AdvancedQuantizer(bpm=120.0)
        notes = [Note(start_time=-0.1, end_time=0.4, pitch=60, velocity=100)]
        result = quantizer.quantize_notes(notes)

        assert len(result) == 1

    def test_fine_grid_division(self):
        quantizer = AdvancedQuantizer(bpm=120.0)
        notes = [
            Note(start_time=0.1234, end_time=0.6234, pitch=60, velocity=100)
        ]
        result = quantizer.quantize_notes(notes, grid_division=0.0625)

        assert len(result) == 1
        assert not np.isnan(result[0].start_time)

    def test_coarse_grid_division(self):
        quantizer = AdvancedQuantizer(bpm=120.0)
        notes = [Note(start_time=0.1, end_time=0.6, pitch=60, velocity=100)]
        result = quantizer.quantize_notes(notes, grid_division=4.0)

        assert len(result) == 1
        assert not np.isnan(result[0].start_time)


def test_humanize_never_moves_note_before_time_zero():
    import numpy as np

    from korshmer.midi_processing.quantization import AdvancedQuantizer, Note

    quantizer = AdvancedQuantizer(bpm=120)
    np.random.seed(1)
    for _ in range(200):
        result = quantizer.quantize_notes(
            [Note(0.0, 0.2, 60, 100)], humanize=True
        )[0]
        assert result.start_time >= 0.0
        assert result.end_time > result.start_time


def test_negative_groove_offset_never_moves_note_before_time_zero():
    from korshmer.midi_processing.quantization import (
        AdvancedQuantizer,
        GrooveTemplate,
        Note,
    )

    quantizer = AdvancedQuantizer(bpm=120)
    quantizer.groove_template = GrooveTemplate(
        name="early", timing_offsets={0.0: -0.05}, velocity_curve={}
    )
    result = quantizer.quantize_notes([Note(0.0, 0.2, 60, 100)])[0]
    assert result.start_time == 0.0
    assert result.end_time > result.start_time
