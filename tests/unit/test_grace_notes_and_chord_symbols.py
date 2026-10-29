import pytest

from korshmer.midi_processing.score_notation import (
    ChordSymbolGenerator,
    GraceNoteHandler,
    Ornament,
    ScoreNote,
)


def _note(duration=1.0, start=2.0, velocity=80):
    return ScoreNote(
        pitch=60, duration=duration, start_time=start, velocity=velocity
    )


@pytest.mark.parametrize("method", ["add_appoggiatura", "add_acciaccatura"])
def test_grace_note_takes_its_time_from_the_main_note_without_a_gap(method):
    main = _note()
    end = main.start_time + main.duration
    grace, main = getattr(GraceNoteHandler(), method)(main, 62)
    assert grace.start_time == 2.0
    assert grace.start_time + grace.duration == pytest.approx(main.start_time)
    assert main.start_time + main.duration == pytest.approx(end)


@pytest.mark.parametrize("method", ["add_appoggiatura", "add_acciaccatura"])
def test_grace_note_at_the_start_of_the_piece_is_not_before_zero(method):
    grace, main = getattr(GraceNoteHandler(), method)(_note(start=0.0), 62)
    assert grace.start_time == 0.0
    assert main.start_time > 0.0


@pytest.mark.parametrize("method", ["add_appoggiatura", "add_acciaccatura"])
@pytest.mark.parametrize("velocity", [1, 5, 10, 15])
def test_grace_note_velocity_stays_above_zero(method, velocity):
    grace, _main = getattr(GraceNoteHandler(), method)(
        _note(velocity=velocity), 62
    )
    assert 1 <= grace.velocity <= 127


@pytest.mark.parametrize("method", ["add_appoggiatura", "add_acciaccatura"])
def test_grace_note_never_makes_the_main_note_negative(method):
    grace, main = getattr(GraceNoteHandler(), method)(_note(duration=0.02), 62)
    assert main.duration == pytest.approx(0.015)
    assert grace.duration == pytest.approx(0.005)


def test_grace_notes_keep_their_ornament_and_velocity_drop():
    handler = GraceNoteHandler()
    appog, _ = handler.add_appoggiatura(_note(), 62)
    acc, _ = handler.add_acciaccatura(_note(), 62)
    assert appog.ornament is Ornament.APPOGGIATURA
    assert acc.ornament is Ornament.ACCIACCATURA
    assert appog.velocity == 70
    assert acc.velocity == 65
    assert appog.duration == 0.125
    assert acc.duration == 0.03125


@pytest.mark.parametrize(
    ("notes", "expected"),
    [
        ([60, 64, 67], "C"),
        ([60, 63, 67], "Cm"),
        ([60, 63, 66], "Cdim"),
        ([60, 64, 68], "Caug"),
        ([60, 62, 67], "Csus2"),
        ([60, 65, 67], "Csus4"),
        ([60, 64, 67, 70], "C7"),
        ([60, 64, 67, 71], "Cmaj7"),
        ([60, 63, 67, 70], "Cm7"),
        ([60, 63, 66, 69], "Cdim7"),
        ([60, 63, 66, 70], "Cm7b5"),
        ([60, 64, 67, 69], "C6"),
        ([60, 64, 67, 70, 74], "C9"),
        ([64, 67, 72], "C"),
        ([48, 60, 64, 67, 79], "C"),
    ],
)
def test_chord_symbol_names_common_chords(notes, expected):
    assert ChordSymbolGenerator().generate_symbol(notes, 60) == expected


def test_chord_symbol_uses_the_root_name():
    assert ChordSymbolGenerator().generate_symbol([66, 70, 73], 66) == "F#"


def test_unknown_chord_keeps_the_question_mark_and_empty_is_empty():
    generator = ChordSymbolGenerator()
    assert generator.generate_symbol([60, 61, 62], 60) == "C?"
    assert generator.generate_symbol([], 60) == ""


@pytest.mark.parametrize("root", [-1, 128, 60.5, "60", None, True])
def test_chord_symbol_rejects_an_invalid_root(root):
    with pytest.raises(ValueError, match="root"):
        ChordSymbolGenerator().generate_symbol([60, 64, 67], root)


@pytest.mark.parametrize("bad", [-1, 128, 60.5, "x"])
def test_chord_symbol_rejects_an_invalid_note(bad):
    with pytest.raises(ValueError, match="note"):
        ChordSymbolGenerator().generate_symbol([60, bad], 60)
