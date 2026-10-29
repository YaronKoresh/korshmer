import pretty_midi
import pytest

from korshmer.midi_algorithms import (
    merge_same_instruments,
    smart_decompose_stem,
    smart_instrument_selector,
)


def _stem_midi(pitches, count=100):
    pm = pretty_midi.PrettyMIDI(initial_tempo=115)
    inst = pretty_midi.Instrument(0)
    for k in range(count):
        inst.notes.append(
            pretty_midi.Note(
                90, pitches[k % len(pitches)], k * 0.5, k * 0.5 + 0.4
            )
        )
    pm.instruments.append(inst)
    return pm


@pytest.mark.parametrize("stem", ["bass", "guitar", "piano", "other"])
def test_selector_returns_one_instrument_with_every_note(stem):
    pm = _stem_midi([48, 52, 55, 60])
    result = smart_instrument_selector(stem, pm)
    assert len(result.instruments) == 1
    assert len(result.instruments[0].notes) == 100


def test_selector_keeps_pitches_when_a_candidate_fits():
    pm = _stem_midi([40, 43, 45, 52])
    result = smart_instrument_selector("bass", pm)
    pitches = {n.pitch for n in result.instruments[0].notes}
    assert pitches == {40, 43, 45, 52}


def test_selector_transposes_by_whole_octaves_only():
    pm = _stem_midi([20, 24, 30])
    result = smart_instrument_selector("bass", pm)
    inst = result.instruments[0]
    assert {n.pitch for n in inst.notes} == {32, 36, 42}


def test_selector_transposes_high_stems_down_by_whole_octaves():
    pm = _stem_midi([81, 85, 89])
    result = smart_instrument_selector("bass", pm)
    pitches = sorted({n.pitch for n in result.instruments[0].notes})
    assert [
        (p - o) % 12 for p, o in zip(pitches, [81, 85, 89], strict=True)
    ] == [0, 0, 0]


def test_selector_falls_back_to_piano_when_nothing_fits():
    pm = _stem_midi([10, 120])
    result = smart_instrument_selector("bass", pm)
    assert [i.program for i in result.instruments] == [0]
    assert len(result.instruments[0].notes) == 100


def test_decomposed_stem_does_not_multiply_notes():
    pm = _stem_midi([40, 43, 45, 52])
    layers = smart_decompose_stem(pm, "bass", 115)
    instruments = [i for _, m in layers for i in m.instruments]
    assert len(instruments) == len(layers)
    assert sum(len(i.notes) for i in instruments) == 100


def test_merged_stem_keeps_the_source_note_count():
    pm = _stem_midi([40, 43, 45, 52])
    layers = smart_decompose_stem(pm, "bass", 115)
    out = pretty_midi.PrettyMIDI(initial_tempo=115)
    for _, m in layers:
        out.instruments.extend(m.instruments)
    merged = merge_same_instruments(out)
    assert sum(len(i.notes) for i in merged.instruments) == 100


def _bass_with_strays(strays):
    pm = _stem_midi([36, 40, 43, 45], count=100)
    inst = pm.instruments[0]
    for k, pitch in enumerate(strays):
        inst.notes.append(pretty_midi.Note(90, pitch, 60 + k, 60.2 + k))
    return pm


def test_a_stray_high_note_does_not_turn_the_bass_into_a_piano():
    result = smart_instrument_selector("bass", _bass_with_strays([90]))

    inst = result.instruments[0]
    assert inst.program != 0
    assert len(inst.notes) == 101
    assert all(28 <= n.pitch <= 67 for n in inst.notes)


def test_stray_notes_are_folded_by_whole_octaves():
    result = smart_instrument_selector("bass", _bass_with_strays([88, 12]))

    strays = sorted(
        n.pitch for n in result.instruments[0].notes if n.start >= 60
    )
    assert strays == [36, 64]


def test_the_main_line_keeps_its_pitches_with_strays_present():
    result = smart_instrument_selector("bass", _bass_with_strays([90]))

    main = {n.pitch for n in result.instruments[0].notes if n.start < 50}
    assert main == {36, 40, 43, 45}
