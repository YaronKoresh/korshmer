import pytest

from korshmer.midi_processing.quantization import AdvancedQuantizer, Note

BAD_FRACTIONS = [-0.1, 1.5, float("nan"), float("inf"), True, "0.5", None]


def _notes():
    return [Note(0.25, 0.4, 60, 80)]


@pytest.mark.parametrize("amount", BAD_FRACTIONS)
def test_apply_swing_rejects_invalid_amount(amount):
    with pytest.raises(ValueError, match="Swing amount"):
        AdvancedQuantizer().apply_swing(_notes(), amount)


@pytest.mark.parametrize("threshold", BAD_FRACTIONS)
def test_selective_quantize_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="Selective quantize threshold"):
        AdvancedQuantizer().selective_quantize(_notes(), 0.25, threshold)


@pytest.mark.parametrize("value", [0, 0.5, 1])
def test_boundary_values_are_accepted(value):
    quantizer = AdvancedQuantizer()
    assert len(quantizer.apply_swing(_notes(), value)) == 1
    assert len(quantizer.selective_quantize(_notes(), 0.25, value)) == 1
