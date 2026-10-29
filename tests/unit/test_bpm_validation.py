import math

import pytest

from korshmer.midi_processing.pattern_generation import (
    Arpeggiator,
    StepSequencer,
)
from korshmer.midi_processing.quantization import (
    AdvancedQuantizer,
    GrooveExtractor,
)
from korshmer.ui.viewmodels.editor_vm import EditorModel
from korshmer.ui.viewmodels.piano_roll_vm import PianoRollViewModel
from korshmer.utils.tempo import (
    DEFAULT_BPM,
    MAX_BPM,
    MIN_BPM,
    validate_bpm,
)


class TestValidateBpm:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (-120, DEFAULT_BPM),
            (0, DEFAULT_BPM),
            (10, MIN_BPM),
            (20, 20.0),
            (120, 120.0),
            (300, 300.0),
            (301, MAX_BPM),
            (5000, MAX_BPM),
            (None, DEFAULT_BPM),
            ("abc", DEFAULT_BPM),
            ("150", 150.0),
        ],
    )
    def test_values(self, value, expected):
        assert validate_bpm(value) == expected

    def test_nan_and_inf_return_default(self):
        assert validate_bpm(float("nan")) == DEFAULT_BPM
        assert validate_bpm(float("inf")) == DEFAULT_BPM
        assert validate_bpm(float("-inf")) == DEFAULT_BPM

    def test_custom_default(self):
        assert validate_bpm(None, default=90.0) == 90.0

    def test_always_finite_and_in_range(self):
        for value in [-1, 0, float("nan"), float("inf"), None, "x", 999]:
            result = validate_bpm(value)
            assert math.isfinite(result)
            assert MIN_BPM <= result <= MAX_BPM


class TestProjectLoadTempo:
    def test_negative_tempo_becomes_default(self):
        model = EditorModel()
        model.load_state({"tempo": -50, "tracks": [], "notes": [], "clips": []})
        assert model.tempo == DEFAULT_BPM

    def test_excessive_tempo_clamped(self):
        model = EditorModel()
        model.load_state(
            {"tempo": 5000, "tracks": [], "notes": [], "clips": []}
        )
        assert model.tempo == MAX_BPM

    def test_infinite_tempo_becomes_default(self):
        model = EditorModel()
        model.load_state(
            {"tempo": float("inf"), "tracks": [], "notes": [], "clips": []}
        )
        assert model.tempo == DEFAULT_BPM

    def test_editor_model_init_validates(self):
        assert EditorModel(tempo=-10).tempo == DEFAULT_BPM
        assert EditorModel(tempo=9000).tempo == MAX_BPM

    def test_compute_tempo_change_rejects_invalid(self):
        vm = PianoRollViewModel()
        assert vm.compute_tempo_change(0) is None
        assert vm.compute_tempo_change(-5) is None
        assert vm.compute_tempo_change(float("nan")) is None
        assert vm.compute_tempo_change(None) is None


class TestMidiProcessingBpm:
    def test_groove_extractor_zero_bpm_no_division_error(self):
        extractor = GrooveExtractor(bpm=0)
        assert extractor.bpm == DEFAULT_BPM
        beat_duration = 60.0 / extractor.bpm
        assert beat_duration > 0

    def test_advanced_quantizer_negative_bpm(self):
        quant = AdvancedQuantizer(bpm=-1)
        assert quant.bpm == DEFAULT_BPM
        assert 60.0 / quant.bpm > 0

    def test_arpeggiator_and_step_sequencer(self):
        assert Arpeggiator(bpm=0).bpm == DEFAULT_BPM
        assert StepSequencer(bpm=-10).bpm == DEFAULT_BPM
        assert StepSequencer(bpm=9999).bpm == MAX_BPM
