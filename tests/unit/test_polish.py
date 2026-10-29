import copy

import pretty_midi
import pytest

from korshmer.midi.polish import TARGETS, classify, polish_midi


def _chords():
    inst = pretty_midi.Instrument(program=0, name="Keys")

    for i, base in enumerate([60, 77, 55, 79]):
        for off in (0, 4, 7):
            inst.notes.append(
                pretty_midi.Note(90, base + off, i * 2.0, i * 2.0 + 1.8)
            )
    return inst


def _lead():
    inst = pretty_midi.Instrument(program=73, name="Lead")
    inst.notes += [
        pretty_midi.Note(90, 72, 0.0, 0.55),
        pretty_midi.Note(90, 74, 0.5, 0.9),
        pretty_midi.Note(90, 76, 1.0, 1.3),
        pretty_midi.Note(90, 76, 1.31, 1.6),
        pretty_midi.Note(90, 79, 1.7, 1.72),
        pretty_midi.Note(90, 77, 2.0, 2.4),
    ]
    return inst


def _drums():
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    for i in range(8):
        inst.notes.append(pretty_midi.Note(100, 36, i * 0.25, i * 0.25 + 0.02))
    return inst


def _song():
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    pm.instruments += [_chords(), _lead(), _drums()]
    return pm


def _notes(pm, name):
    inst = next(i for i in pm.instruments if i.name == name)
    return sorted(inst.notes, key=lambda n: (n.start, n.pitch))


def _chord_centers(pm):
    notes = _notes(pm, "Keys")
    groups = {}
    for n in notes:
        groups.setdefault(round(n.start, 3), []).append(n.pitch)
    return [sum(v) / len(v) for _, v in sorted(groups.items())]


def _total_jump(pm):
    c = _chord_centers(pm)
    return sum(abs(b - a) for a, b in zip(c, c[1:], strict=False))


def test_classifies_stacked_notes_as_chords_and_single_line_as_lead():
    assert classify(_chords()) == "chords"
    assert classify(_lead()) == "lead"
    assert classify(pretty_midi.Instrument(program=0)) == "lead"


def test_everything_smooths_chords_and_the_lead():
    pm, report = polish_midi(_song(), "Everything")
    assert _total_jump(pm) < _total_jump(_song())
    lead = _notes(pm, "Lead")
    for a, b in zip(lead, lead[1:], strict=False):
        assert a.end <= b.start + 1e-9
    assert report["chord_tracks"] == 1
    assert report["lead_tracks"] == 1
    assert report["chord_notes_moved"] > 0


def test_lead_glitches_are_removed_and_repeats_merged():
    pm, report = polish_midi(_song(), "Lead", min_ms=50)
    lead = _notes(pm, "Lead")
    assert 79 not in {n.pitch for n in lead}
    assert [n.pitch for n in lead].count(76) == 1
    assert report["short_removed"] == 1
    assert report["repeats_merged"] == 1


def test_small_gaps_in_the_lead_are_closed():
    pm, _ = polish_midi(_song(), "Lead")
    lead = _notes(pm, "Lead")
    first = lead[0]
    second = lead[1]
    assert abs(first.end - second.start) < 1e-9


def test_chord_pitch_classes_and_voice_count_are_kept():
    before = _song()
    pm, _ = polish_midi(before, "Chords")
    a = _notes(before, "Keys")
    b = _notes(pm, "Keys")
    assert len(a) == len(b)
    assert sorted(n.pitch % 12 for n in a) == sorted(n.pitch % 12 for n in b)


def test_drums_are_never_changed():
    before = _song()
    pm, _ = polish_midi(before, "Everything", min_ms=500)
    assert [(n.pitch, n.start, n.end) for n in _notes(pm, "Drums")] == [
        (n.pitch, n.start, n.end) for n in _notes(before, "Drums")
    ]


def test_target_chords_leaves_the_lead_alone():
    before = _song()
    pm, report = polish_midi(before, "Chords")
    assert [(n.pitch, n.start, n.end) for n in _notes(pm, "Lead")] == [
        (n.pitch, n.start, n.end) for n in _notes(before, "Lead")
    ]
    assert report["lead_tracks"] == 0


def test_target_lead_leaves_the_chords_alone():
    before = _song()
    pm, report = polish_midi(before, "Lead")
    assert [(n.pitch, n.start, n.end) for n in _notes(pm, "Keys")] == [
        (n.pitch, n.start, n.end) for n in _notes(before, "Keys")
    ]
    assert report["chord_tracks"] == 0


def test_the_input_is_not_modified():
    before = _song()
    snapshot = copy.deepcopy(
        [
            [(n.pitch, n.start, n.end) for n in i.notes]
            for i in before.instruments
        ]
    )
    polish_midi(before, "Everything")
    assert snapshot == [
        [(n.pitch, n.start, n.end) for n in i.notes] for i in before.instruments
    ]


def test_empty_file_and_unknown_target():
    pm, report = polish_midi(pretty_midi.PrettyMIDI(), "Everything")
    assert pm.instruments == []
    assert report["short_removed"] == 0
    with pytest.raises(ValueError, match="target"):
        polish_midi(_song(), "Nonsense")


def test_all_targets_run():
    for target in TARGETS:
        polish_midi(_song(), target)


def test_polish_is_idempotent_for_a_clean_file():
    once, _ = polish_midi(_song(), "Everything")
    twice, _ = polish_midi(once, "Everything")
    assert [
        (n.pitch, round(n.start, 6), round(n.end, 6))
        for i in once.instruments
        for n in i.notes
    ] == [
        (n.pitch, round(n.start, 6), round(n.end, 6))
        for i in twice.instruments
        for n in i.notes
    ]


def test_process_polish_writes_a_new_file(tmp_path):
    from korshmer.midi_processors import process_polish

    path = tmp_path / "song.mid"
    _song().write(str(path))
    out = process_polish(str(path), "Everything", 50)
    assert out != str(path)
    pm = pretty_midi.PrettyMIDI(out)
    assert len(pm.instruments) == 3
