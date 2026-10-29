import pytest

from korshmer.midi_processing.quantization import (
    AdvancedQuantizer,
    Note,
    QuantizeMode,
)


def _notes(first=0.35, count=3, step=0.5):
    return [
        Note(
            start_time=first + step * k,
            end_time=first + step * k + 0.25,
            pitch=60,
            velocity=80,
        )
        for k in range(count)
    ]


def test_default_offset_keeps_the_zero_grid():
    q = AdvancedQuantizer(120.0)

    starts = [n.start_time for n in q.quantize_notes(_notes(), 0.25)]

    assert starts == pytest.approx([0.375, 0.875, 1.375])


def test_offset_moves_the_grid_with_the_song():
    q = AdvancedQuantizer(120.0, grid_offset=0.35)

    starts = [n.start_time for n in q.quantize_notes(_notes(), 0.25)]

    assert starts == pytest.approx([0.35, 0.85, 1.35])


@pytest.mark.parametrize(
    "mode", [QuantizeMode.NEAREST, QuantizeMode.FLOOR, QuantizeMode.CEIL]
)
def test_fixed_modes_land_on_the_offset_grid(mode):
    q = AdvancedQuantizer(120.0, grid_offset=0.35)
    notes = _notes(first=0.4, count=1)

    start = q.quantize_notes(notes, 0.25, mode)[0].start_time

    steps = (start - 0.35) / 0.125
    assert steps == pytest.approx(round(steps), abs=1e-6)


def test_beat_position_counts_from_the_offset():
    q = AdvancedQuantizer(120.0, grid_offset=0.35)

    assert q._get_beat_position(0.35) == pytest.approx(0.0)
    assert q._get_beat_position(0.6) == pytest.approx(0.5)


def test_swing_finds_the_off_beat_after_the_offset():
    q = AdvancedQuantizer(120.0, grid_offset=0.35)
    note = Note(start_time=0.6, end_time=0.8, pitch=60, velocity=80)

    swung = q.apply_swing([note], 1.0)[0]

    assert swung.start_time > 0.6


def test_selective_quantize_uses_the_offset_grid():
    q = AdvancedQuantizer(120.0, grid_offset=0.35)
    on_grid = Note(start_time=0.35, end_time=0.5, pitch=60, velocity=80)

    assert q.selective_quantize([on_grid], 0.25, 0.1)[0] is on_grid


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), True, None])
def test_invalid_offset_is_rejected(bad):
    with pytest.raises(ValueError):
        AdvancedQuantizer(120.0, grid_offset=bad)
    q = AdvancedQuantizer(120.0)
    with pytest.raises(ValueError):
        q.grid_offset = bad
