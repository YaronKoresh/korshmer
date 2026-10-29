import math

import pytest

from korshmer.midi_processing.quantization import (
    AdvancedQuantizer,
    DottedNoteQuantizer,
    GrooveExtractor,
    Note,
    QuantizeMode,
)

BAD_GRIDS = [0, 0.0, -0.25, float("nan"), float("inf"), True, "0.25", None]


def _notes():
    return [Note(0.13, 0.4, 60, 90)]


@pytest.mark.parametrize("grid", BAD_GRIDS)
@pytest.mark.parametrize("mode", list(QuantizeMode))
def test_quantize_notes_rejects_invalid_grid(grid, mode):
    with pytest.raises(ValueError, match="Grid division"):
        AdvancedQuantizer().quantize_notes(_notes(), grid, mode)


@pytest.mark.parametrize("grid", BAD_GRIDS)
def test_other_entry_points_reject_invalid_grid(grid):
    with pytest.raises(ValueError, match="Grid division"):
        AdvancedQuantizer().selective_quantize(_notes(), grid)
    with pytest.raises(ValueError, match="Grid division"):
        AdvancedQuantizer().quantize_with_strength_curve(_notes(), grid)
    with pytest.raises(ValueError, match="Grid division"):
        DottedNoteQuantizer().quantize_to_dotted(_notes(), grid)
    with pytest.raises(ValueError, match="Grid division"):
        GrooveExtractor().extract_from_midi(_notes(), grid)


@pytest.mark.parametrize(
    "strength", [-0.1, 1.5, 5, float("nan"), float("inf"), True, "1"]
)
def test_strength_outside_zero_to_one_is_rejected(strength):
    quantizer = AdvancedQuantizer()
    with pytest.raises(ValueError, match="Quantize strength"):
        quantizer.quantize_strength = strength
    assert quantizer.quantize_strength == 1.0


def test_valid_input_still_quantizes():
    quantizer = AdvancedQuantizer(bpm=120)
    quantizer.quantize_strength = 0.5
    out = quantizer.quantize_notes(_notes(), 0.25)[0]
    assert out.start_time == pytest.approx(0.13 * 0.5 + 0.125 * 0.5)


def test_strength_curve_is_clamped_and_strength_restored():
    quantizer = AdvancedQuantizer()
    quantizer.quantize_strength = 0.3
    out = quantizer.quantize_with_strength_curve(
        _notes(), 0.25, lambda position: 5.0
    )[0]
    assert out.start_time == pytest.approx(0.125)
    assert quantizer.quantize_strength == 0.3


def test_strength_curve_nan_is_rejected_and_strength_restored():
    quantizer = AdvancedQuantizer()
    with pytest.raises(ValueError):
        quantizer.quantize_with_strength_curve(
            _notes(), 0.25, lambda position: math.nan
        )
    assert quantizer.quantize_strength == 1.0
