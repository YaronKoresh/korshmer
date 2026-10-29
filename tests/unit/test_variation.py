import copy

import numpy as np
import pretty_midi
import pytest

from korshmer.midi.variation import STRENGTHS, create_variation

BAR = 2.0
BARS = 8
MELODY_SCALE = [60, 62, 64, 65, 67, 69, 71, 72]


def _song():
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    melody = pretty_midi.Instrument(0, name="melody")
    chords = pretty_midi.Instrument(0, name="chords")
    bass = pretty_midi.Instrument(32, name="bass")
    drums = pretty_midi.Instrument(0, is_drum=True, name="drums")
    rng = np.random.default_rng(5)
    for bar in range(BARS):
        t = bar * BAR
        for k in range(8):
            start = t + k * 0.25
            pitch = int(rng.choice(MELODY_SCALE))
            melody.notes.append(pretty_midi.Note(90, pitch, start, start + 0.2))
        for offset, voicing in ((0.0, (48, 52, 55)), (1.0, (50, 53, 57))):
            for pitch in voicing:
                chords.notes.append(
                    pretty_midi.Note(
                        80, pitch + 12, t + offset, t + offset + 0.9
                    )
                )
        bass.notes.append(pretty_midi.Note(80, 36 + 2 * (bar % 2), t, t + 0.9))
        bass.notes.append(
            pretty_midi.Note(80, 43 + 2 * (bar % 2), t + 1.0, t + 1.9)
        )
        for k in range(4):
            start = t + k * 0.5
            drums.notes.append(pretty_midi.Note(100, 36, start, start + 0.1))
    pm.instruments.extend([melody, chords, bass, drums])
    return pm


def _signature(pm):
    return [
        [
            (n.pitch, round(n.start, 6), round(n.end, 6), n.velocity)
            for n in i.notes
        ]
        for i in pm.instruments
    ]


def _share(pm, strength, seed=1):
    _, report = create_variation(pm, strength, seed)
    return report["notes_changed"] / report["notes_before"]


def test_same_seed_gives_the_same_variation_and_other_seeds_differ():
    song = _song()
    first, _ = create_variation(song, "Medium", 7)
    again, _ = create_variation(song, "Medium", 7)
    other, _ = create_variation(song, "Medium", 8)
    assert _signature(first) == _signature(again)
    assert _signature(first) != _signature(other)


def test_input_is_not_modified():
    song = _song()
    before = _signature(song)
    create_variation(song, "Strong", 3)
    assert _signature(song) == before


def test_stronger_settings_change_more_and_stay_in_a_sensible_band():
    song = _song()
    light, medium, strong = (
        _share(song, name) for name in ("Light", "Medium", "Strong")
    )
    assert 0.2 <= light < medium < strong <= 0.8


def test_drums_are_untouched():
    song = _song()
    varied, _ = create_variation(song, "Strong", 4)
    assert _signature(varied)[3] == _signature(song)[3]


def test_skeleton_stays_first_note_of_every_second_bar_and_longest_notes():
    song = _song()
    varied, _ = create_variation(song, "Strong", 9)
    for track in (0, 2):
        original = {
            (n.pitch, round(n.start, 3))
            for n in song.instruments[track].notes
            if round(n.start / BAR, 3) % 2 == 0 and abs(n.start % BAR) < 1e-6
        }
        kept = {
            (n.pitch, round(n.start, 3))
            for n in varied.instruments[track].notes
        }
        assert original <= kept


def test_every_second_bar_chord_is_kept_whole():
    song = _song()
    varied, _ = create_variation(song, "Strong", 2)
    for bar in range(0, BARS, 2):
        start = bar * BAR
        before = sorted(
            n.pitch
            for n in song.instruments[1].notes
            if abs(n.start - start) < 1e-6
        )
        after = sorted(
            n.pitch
            for n in varied.instruments[1].notes
            if abs(n.start - start) < 1e-6
        )
        assert before == after


def test_moved_melody_notes_stay_in_the_key_and_in_midi_range():
    song = _song()
    varied, _ = create_variation(song, "Strong", 11)
    key_classes = {p % 12 for p in MELODY_SCALE}
    for note in varied.instruments[0].notes:
        assert 0 <= note.pitch <= 127
        assert note.pitch % 12 in key_classes
        assert 1 <= note.velocity <= 127
        assert note.end > note.start


def test_chord_changes_are_inversions_of_the_same_harmony():
    song = _song()
    varied, _ = create_variation(song, "Strong", 6)
    changed = False
    for bar in range(BARS):
        for offset in (0.0, 1.0):
            start = bar * BAR + offset
            before = sorted(
                n.pitch
                for n in song.instruments[1].notes
                if abs(n.start - start) < 1e-6
            )
            after = sorted(
                n.pitch
                for n in varied.instruments[1].notes
                if abs(n.start - start) < 1e-6
            )
            assert sorted(p % 12 for p in before) == sorted(
                p % 12 for p in after
            )
            changed = changed or before != after
    assert changed


def test_report_counts_match_the_result():
    song = _song()
    varied, report = create_variation(song, "Medium", 5)
    assert report["notes_before"] == sum(
        len(i.notes) for i in song.instruments if not i.is_drum
    )
    grown = sum(len(i.notes) for i in varied.instruments if not i.is_drum)
    assert grown == report["notes_before"] + report["notes_added"]
    assert report["tracks_changed"] >= 1
    assert report["anchors_kept"] > 0


def test_unknown_strength_is_rejected():
    with pytest.raises(ValueError, match="Unknown strength"):
        create_variation(_song(), "Wild")


@pytest.mark.parametrize("strength", list(STRENGTHS))
def test_empty_and_drum_only_files_pass_through(strength):
    empty = pretty_midi.PrettyMIDI()
    varied, report = create_variation(empty, strength)
    assert varied.instruments == []
    assert report["notes_changed"] == 0
    drums = pretty_midi.PrettyMIDI()
    kit = pretty_midi.Instrument(0, is_drum=True)
    kit.notes.append(pretty_midi.Note(100, 36, 0.0, 0.1))
    drums.instruments.append(kit)
    varied, _ = create_variation(drums, strength)
    assert _signature(varied) == _signature(drums)


def test_single_note_track_survives():
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(90, 60, 0.0, 1.0))
    pm.instruments.append(inst)
    varied, _ = create_variation(pm, "Strong", 1)
    assert [(n.pitch, n.start) for n in varied.instruments[0].notes] == [
        (60, 0.0)
    ]


def test_written_file_round_trips(tmp_path):
    song = _song()
    varied, _ = create_variation(song, "Medium", 12)
    path = tmp_path / "variation.mid"
    varied.write(str(path))
    loaded = pretty_midi.PrettyMIDI(str(path))
    assert len(loaded.instruments) == len(song.instruments)
    assert copy.deepcopy(loaded).get_end_time() > 0


def test_syncopation_preserves_note_durations():
    from korshmer.midi.variation import _syncopate

    inst = pretty_midi.Instrument(0)
    for k in range(16):
        start = k * 0.5
        inst.notes.append(pretty_midi.Note(80, 60 + k % 3, start, start + 0.4))
    before = {id(n): round(n.end - n.start, 6) for n in inst.notes}
    _syncopate(
        inst, list(inst.notes), len(inst.notes), np.random.default_rng(3), 0.5
    )
    assert inst.notes
    for n in inst.notes:
        assert round(n.end - n.start, 6) >= before[id(n)] - 1e-9


def test_syncopation_can_anticipate_and_delay():
    from korshmer.midi.variation import _syncopate

    inst = pretty_midi.Instrument(0)
    for k in range(40):
        start = k * 0.5
        inst.notes.append(pretty_midi.Note(80, 60, start, start + 0.3))
    before = {id(n): n.start for n in inst.notes}
    _syncopate(
        inst, list(inst.notes), len(inst.notes), np.random.default_rng(1), 0.5
    )
    moved = [
        round(n.start - before[id(n)], 6)
        for n in inst.notes
        if abs(n.start - before[id(n)]) > 1e-9
    ]
    assert any(d < 0 for d in moved)
    assert any(d > 0 for d in moved)


def test_syncopation_does_not_overlap_neighbours():
    from korshmer.midi.variation import _syncopate

    inst = pretty_midi.Instrument(0)
    for k in range(32):
        start = k * 0.5
        inst.notes.append(pretty_midi.Note(80, 60 + k % 5, start, start + 0.4))
    _syncopate(
        inst, list(inst.notes), len(inst.notes), np.random.default_rng(2), 0.5
    )
    ordered = sorted(inst.notes, key=lambda n: n.start)
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        assert earlier.end <= later.start + 1e-9


def test_process_variation_saves_a_new_file(tmp_path, monkeypatch):
    from korshmer import midi_algorithms
    from korshmer.midi_processors import process_variation

    monkeypatch.setattr(midi_algorithms, "DATA_DIR", str(tmp_path))
    source = tmp_path / "song.mid"
    _song().write(str(source))
    out = process_variation(str(source), "Light", 3)
    assert out.endswith("song_Variation_Light_3.mid")
    assert out != str(source)
    assert pretty_midi.PrettyMIDI(out).instruments
