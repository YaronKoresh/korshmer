import pretty_midi
import pytest

from korshmer import midi_algorithms, midi_processors
from korshmer.midi import save_midi
from korshmer.utils.track_layout import (
    DRUMS_NAME,
    FALLBACK_NAME,
    PROGRAM_NAMES,
    instrument_name,
    is_general_midi_name,
    merge_by_instrument,
)

MAX_NAME_LENGTH = 15


def _inst(program=0, is_drum=False, name="", notes=()):
    inst = pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)
    for pitch, start, end, velocity in notes:
        inst.notes.append(pretty_midi.Note(velocity, pitch, start, end))
    return inst


class TestInstrumentNames:
    def test_there_is_one_name_per_program(self):
        assert len(PROGRAM_NAMES) == 128

    def test_names_are_unique(self):
        assert len(set(PROGRAM_NAMES)) == 128
        assert DRUMS_NAME not in PROGRAM_NAMES

    def test_names_are_short_and_plain(self):
        for name in PROGRAM_NAMES:
            assert 0 < len(name) <= MAX_NAME_LENGTH
            assert name == name.strip()
            assert not any(c in name for c in "()0123456789"), name

    def test_no_general_midi_name_with_numbers_or_brackets_is_left(self):
        for program in range(128):
            classic = pretty_midi.program_to_instrument_name(program)
            if any(c in classic for c in "()0123456789"):
                assert instrument_name(program) != classic

    def test_well_known_programs(self):
        assert instrument_name(0) == "Grand Piano"
        assert instrument_name(33) == "Bass Guitar"
        assert instrument_name(48) == "Strings"
        assert instrument_name(80) == "Square Lead"
        assert instrument_name(89) == "Warm Pad"

    def test_drums_ignore_the_program(self):
        assert instrument_name(0, is_drum=True) == DRUMS_NAME
        assert instrument_name(40, is_drum=True) == DRUMS_NAME

    @pytest.mark.parametrize("program", [-1, 128, 500, None, "x"])
    def test_unknown_programs_get_the_fallback_name(self, program):
        assert instrument_name(program) == FALLBACK_NAME


class TestGeneralMidiNames:
    @pytest.mark.parametrize(
        "name",
        [
            "Acoustic Grand Piano",
            "electric bass (finger)",
            "Synth Strings 1",
            "Acoustic Grand Piano (Low)",
            "Bass - Electric Bass (Finger)",
        ],
    )
    def test_classic_names_are_recognised(self, name):
        assert is_general_midi_name(name)

    @pytest.mark.parametrize(
        "name",
        ["", "Violin", "Lead Vocal", "Left hand", "Grand Piano", "Cello"],
    )
    def test_plain_and_user_names_are_not(self, name):
        assert not is_general_midi_name(name)


class TestMergeByInstrument:
    def test_same_program_becomes_one_track(self):
        a = _inst(4, name="Schema 1", notes=[(60, 0.0, 0.5, 90)])
        b = _inst(4, name="Schema 2", notes=[(64, 0.25, 0.75, 80)])
        c = _inst(4, name="Chords", notes=[(67, 1.0, 1.5, 70)])

        tracks = merge_by_instrument([a, b, c])

        assert len(tracks) == 1
        assert tracks[0].program == 4
        assert not tracks[0].is_drum
        assert tracks[0].name == "Mellow Keys"
        assert [n.pitch for n in tracks[0].notes] == [60, 64, 67]

    def test_different_programs_stay_apart_in_first_seen_order(self):
        low = _inst(33, notes=[(40, 0.0, 0.5, 90)])
        high = _inst(80, notes=[(72, 0.0, 0.5, 90)])
        again = _inst(33, notes=[(43, 1.0, 1.5, 90)])

        tracks = merge_by_instrument([low, high, again])

        assert [t.program for t in tracks] == [33, 80]
        assert [t.name for t in tracks] == ["Bass Guitar", "Square Lead"]
        assert len(tracks[0].notes) == 2

    def test_all_drums_become_one_track_with_the_busiest_kit(self):
        quiet = _inst(8, True, notes=[(36, 0.0, 0.1, 100)])
        busy = _inst(
            16,
            True,
            notes=[
                (38, 0.5, 0.6, 100),
                (42, 0.75, 0.8, 90),
                (42, 1.0, 1.1, 90),
            ],
        )

        tracks = merge_by_instrument([quiet, busy])

        assert len(tracks) == 1
        assert tracks[0].is_drum
        assert tracks[0].program == 16
        assert tracks[0].name == DRUMS_NAME
        assert len(tracks[0].notes) == 4

    def test_drums_and_a_piano_are_different_instruments(self):
        piano = _inst(0, notes=[(60, 0.0, 0.5, 90)])
        kit = _inst(0, True, notes=[(36, 0.0, 0.1, 90)])

        tracks = merge_by_instrument([kit, piano])

        assert [t.is_drum for t in tracks] == [True, False]
        assert [t.name for t in tracks] == [DRUMS_NAME, "Grand Piano"]

    def test_empty_instruments_are_dropped(self):
        tracks = merge_by_instrument(
            [_inst(0), _inst(1, notes=[(60, 0.0, 0.5, 90)])]
        )

        assert [t.program for t in tracks] == [1]

    def test_the_sources_are_left_alone(self):
        a = _inst(0, name="Melody", notes=[(60, 0.0, 1.0, 90)])
        b = _inst(0, name="Arp", notes=[(60, 0.5, 1.5, 90)])

        tracks = merge_by_instrument([a, b])

        assert a.name == "Melody" and b.name == "Arp"
        assert a.notes[0].end == 1.0
        assert tracks[0].notes[0] is not a.notes[0]

    def test_same_pitch_overlaps_are_cut_short_not_dropped(self):
        a = _inst(0, notes=[(60, 0.0, 1.0, 90)])
        b = _inst(0, notes=[(60, 0.5, 1.5, 90)])

        notes = merge_by_instrument([a, b])[0].notes

        assert [(n.start, n.end) for n in notes] == [(0.0, 0.5), (0.5, 1.5)]

    def test_notes_starting_together_on_one_pitch_keep_the_louder_one(self):
        a = _inst(0, notes=[(60, 0.0, 0.5, 60)])
        b = _inst(0, notes=[(60, 0.0, 1.0, 100)])

        notes = merge_by_instrument([a, b])[0].notes

        assert len(notes) == 1
        assert notes[0].velocity == 100
        assert notes[0].end == 1.0

    def test_notes_are_sorted_by_time(self):
        a = _inst(0, notes=[(60, 2.0, 2.5, 90)])
        b = _inst(0, notes=[(62, 0.0, 0.5, 90), (64, 1.0, 1.5, 90)])

        notes = merge_by_instrument([a, b])[0].notes

        assert [n.start for n in notes] == [0.0, 1.0, 2.0]

    def test_each_controller_follows_one_source(self):
        a = _inst(0, notes=[(60, 0.0, 0.5, 90)])
        a.control_changes.append(pretty_midi.ControlChange(11, 100, 0.0))
        b = _inst(0, notes=[(62, 0.0, 0.5, 90)])
        b.control_changes.append(pretty_midi.ControlChange(11, 20, 0.1))
        b.control_changes.append(pretty_midi.ControlChange(64, 127, 0.2))

        changes = merge_by_instrument([a, b])[0].control_changes

        assert [(c.number, c.value) for c in changes] == [(11, 100), (64, 127)]

    def test_pitch_bends_follow_the_first_source_that_has_them(self):
        a = _inst(0, notes=[(60, 0.0, 0.5, 90)])
        b = _inst(0, notes=[(62, 0.0, 0.5, 90)])
        b.pitch_bends.append(pretty_midi.PitchBend(1000, 0.1))

        bends = merge_by_instrument([a, b])[0].pitch_bends

        assert [(p.pitch, p.time) for p in bends] == [(1000, 0.1)]

    def test_a_kept_name_survives_only_when_every_source_has_it(self):
        voice = _inst(52, name="Vocals", notes=[(60, 0.0, 0.5, 90)])
        other = _inst(52, name="Pad", notes=[(62, 0.0, 0.5, 90)])

        alone = merge_by_instrument([voice], keep_names={"Vocals"})
        mixed = merge_by_instrument([voice, other], keep_names={"Vocals"})
        plain = merge_by_instrument([voice])

        assert alone[0].name == "Vocals"
        assert mixed[0].name == "Choir"
        assert plain[0].name == "Choir"

    def test_shared_names_are_kept_on_request(self):
        named = _inst(0, name="Left hand", notes=[(60, 0.0, 0.5, 90)])
        same = _inst(0, name="Left hand", notes=[(62, 0.0, 0.5, 90)])
        other = _inst(0, name="Right hand", notes=[(64, 0.0, 0.5, 90)])
        blank = _inst(33, notes=[(40, 0.0, 0.5, 90)])

        kept = merge_by_instrument([named, same], keep_shared_names=True)
        mixed = merge_by_instrument([named, other], keep_shared_names=True)
        unnamed = merge_by_instrument([blank], keep_shared_names=True)

        assert kept[0].name == "Left hand"
        assert mixed[0].name == "Grand Piano"
        assert unnamed[0].name == "Bass Guitar"

    def test_classic_shared_names_are_not_kept(self):
        classic = _inst(
            33, name="Electric Bass (finger)", notes=[(40, 0, 0.5, 90)]
        )

        tracks = merge_by_instrument([classic], keep_shared_names=True)

        assert tracks[0].name == "Bass Guitar"

    def test_a_lone_track_keeps_its_notes_untouched(self):
        lone = _inst(0, notes=[(60, 0.0, 1.0, 90), (60, 0.002, 1.0, 90)])

        notes = merge_by_instrument([lone])[0].notes

        assert [(n.start, n.end) for n in notes] == [(0.0, 1.0), (0.002, 1.0)]

    def test_merged_notes_starting_within_the_minimum_length_fold_together(
        self,
    ):
        a = _inst(0, notes=[(60, 0.0, 1.0, 90)])
        b = _inst(0, notes=[(60, 0.005, 1.0, 90)])

        notes = merge_by_instrument([a, b])[0].notes

        assert [(n.start, n.end) for n in notes] == [(0.0, 1.0)]

    def test_nothing_in_nothing_out(self):
        assert merge_by_instrument([]) == []


def _song_with_bass(path):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    piano = _inst(
        0, notes=[(60, k * 0.5, k * 0.5 + 0.4, 90) for k in range(16)]
    )
    bass = _inst(33, notes=[(36, k * 2.0, k * 2.0 + 1.0, 90) for k in range(4)])
    pm.instruments.extend([piano, bass])
    pm.write(str(path))
    return str(path)


def test_the_toolbox_never_writes_two_tracks_of_one_instrument(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(midi_algorithms, "DATA_DIR", str(tmp_path / "data"))
    song = _song_with_bass(tmp_path / "song.mid")

    out = midi_processors.process_bass_gen(song)

    instruments = pretty_midi.PrettyMIDI(out).instruments
    keys = [(i.is_drum, 0 if i.is_drum else i.program) for i in instruments]
    assert len(keys) == len(set(keys))
    assert sum(1 for i in instruments if i.program == 33) == 1


def test_splitting_by_pitch_keeps_its_two_halves(tmp_path, monkeypatch):
    monkeypatch.setattr(midi_algorithms, "DATA_DIR", str(tmp_path / "data"))
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.instruments.append(
        _inst(
            0, notes=[(40 + k, k * 0.5, k * 0.5 + 0.4, 90) for k in range(30)]
        )
    )
    pm.write(str(tmp_path / "song.mid"))

    out = midi_processors.process_split_pitch(str(tmp_path / "song.mid"), 60)

    halves = pretty_midi.PrettyMIDI(out).instruments
    assert sorted(len(i.notes) for i in halves) == [10, 20]


def test_save_midi_writes_one_track_per_instrument(tmp_path):
    pm = pretty_midi.PrettyMIDI()
    pm.instruments.extend(
        [
            _inst(4, notes=[(60, 0.0, 0.5, 90)]),
            _inst(4, notes=[(64, 0.0, 0.5, 90)]),
        ]
    )

    out = save_midi(pm, str(tmp_path / "song.mid"), "Saved")

    instruments = pretty_midi.PrettyMIDI(out).instruments
    assert [i.name for i in instruments] == ["Mellow Keys"]
    assert len(instruments[0].notes) == 2
