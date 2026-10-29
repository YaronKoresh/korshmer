import pretty_midi

from korshmer.midi.performance_recorder import PerformanceRecorder


def note(kind, n, v=90, ch=0):
    return {"type": kind, "channel": ch, "note": n, "velocity": v}


def test_three_notes_become_a_midi_file_with_the_same_times(tmp_path):
    rec = PerformanceRecorder(clock=lambda: 100.0)
    rec.start()
    for _i, (n, on, off) in enumerate(
        [(60, 0.0, 0.5), (64, 0.5, 1.25), (67, 1.0, 2.0)]
    ):
        assert rec.feed(note("note_on", n), now=100.0 + on)
        assert rec.feed(note("note_off", n, 0), now=100.0 + off)
    rec.stop()
    path = rec.write(str(tmp_path / "take.mid"))
    notes = sorted(
        pretty_midi.PrettyMIDI(path).instruments[0].notes, key=lambda n: n.start
    )
    assert [n.pitch for n in notes] == [60, 64, 67]
    got = [(round(n.start, 2), round(n.end, 2)) for n in notes]
    assert got == [(0.0, 0.5), (0.5, 1.25), (1.0, 2.0)]


def test_cc_kept_separately_and_other_messages_ignored():
    rec = PerformanceRecorder()
    assert not rec.feed(note("note_on", 60), now=1.0)
    rec.start()
    assert rec.feed(
        {"type": "control_change", "channel": 1, "control": 7, "value": 100},
        now=rec._start + 0.2,
    )
    assert not rec.feed({"type": "pitchwheel", "channel": 0}, now=rec._start)
    assert not rec.feed(note("note_on", 200), now=rec._start)
    assert len(rec.events) == 0
    ((t, ch, num, val),) = rec.control_changes
    assert (round(t, 3), ch, num, val) == (0.2, 1, 7, 100)
    pm = rec.to_pretty_midi()
    assert [(c.number, c.value) for c in pm.instruments[0].control_changes] == [
        (7, 100)
    ]


def test_note_on_with_zero_velocity_ends_the_note_and_held_notes_are_closed():
    rec = PerformanceRecorder()
    rec.start()
    s = rec._start
    rec.feed(note("note_on", 60), now=s)
    rec.feed(note("note_on", 60, 0), now=s + 1.0)
    rec.feed(note("note_on", 62), now=s + 1.5)
    rec.feed(note("note_off", 65, 0), now=s + 3.0)
    notes = rec.to_pretty_midi().instruments[0].notes
    assert [(n.pitch, round(n.end, 1)) for n in notes] == [(60, 1.0), (62, 3.0)]
