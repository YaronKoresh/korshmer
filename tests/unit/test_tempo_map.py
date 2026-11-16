import numpy as np
import pretty_midi
import pytest

from korshmer.midi.tempo_map import TempoMap, retime_instruments

TWO = TempoMap((0.0, 30.0), (100.0, 120.0))


def test_seconds_and_beats_convert_both_ways_across_a_change():
    assert TWO.beats_at(30.0) == pytest.approx(50.0)
    assert TWO.beats_at(31.0) == pytest.approx(52.0)
    assert TWO.seconds_at(52.0) == pytest.approx(31.0)
    times = np.array([0.0, 12.3, 30.0, 47.5, 200.0])
    assert np.allclose(TWO.seconds_at(TWO.beats_at(times)), times)
    assert isinstance(TWO.beats_at(1.0), float)


def test_warping_makes_the_beats_even_and_unwarping_restores_them():
    beats = TWO.seconds_at(np.arange(0, 80))
    even = TWO.warp(beats, 120.0)
    assert np.allclose(np.diff(even), 0.5)
    assert np.allclose(TWO.unwarp(even, 120.0), beats)


def test_anchors_give_the_tempo_between_them():
    found = TempoMap.from_anchors(
        [(2.0, 0), (12.0, 20), (22.0, 40), (30.0, 60)]
    )
    assert found.starts == (0.0, 22.0)
    assert found.bpms == pytest.approx((120.0, 150.0))
    with pytest.raises(ValueError, match="got"):
        TempoMap.from_anchors([(1.0, 0), (1.0, 4)])


def test_main_tempo_is_the_one_that_lasts_longest():
    assert TWO.main_bpm(end=200.0) == pytest.approx(120.0)
    assert TWO.main_bpm(end=40.0) == pytest.approx(100.0)
    assert TempoMap.uniform(97.6).main_bpm() == 97.6


def test_stretching_and_scaling_keep_the_beat_count():
    slow = TWO.stretched(2.0)
    assert slow.starts == (0.0, 60.0)
    assert slow.bpms == (50.0, 60.0)
    assert slow.beats_at(62.0) == pytest.approx(TWO.beats_at(31.0))
    assert TWO.scaled(1.1).bpms == pytest.approx((110.0, 132.0))


@pytest.mark.parametrize(
    "starts, bpms",
    [
        ((), ()),
        ((0.0, 5.0), (120.0,)),
        ((1.0,), (120.0,)),
        ((0.0, 5.0, 5.0), (120.0, 90.0, 80.0)),
        ((0.0,), (0.0,)),
        ((0.0,), (float("nan"),)),
        ((0.0,), ("fast",)),
    ],
)
def test_invalid_maps_are_rejected(starts, bpms):
    with pytest.raises(ValueError, match="got"):
        TempoMap(starts, bpms)


def test_a_written_midi_file_carries_the_tempo_changes(tmp_path):
    pm = TWO.new_pretty_midi()
    inst = pretty_midi.Instrument(0)
    for beat in (0, 49, 50, 51, 90):
        start = TWO.seconds_at(beat)
        inst.notes.append(pretty_midi.Note(90, 60, start, start + 0.2))
    pm.instruments.append(inst)
    path = tmp_path / "map.mid"
    pm.write(str(path))

    loaded = pretty_midi.PrettyMIDI(str(path))
    found = TempoMap.from_pretty_midi(loaded)
    assert found.starts == pytest.approx(TWO.starts, abs=0.002)
    assert found.bpms == pytest.approx(TWO.bpms, rel=1e-4)
    starts = [n.start for n in loaded.instruments[0].notes]
    assert found.beats_at(np.array(starts)) == pytest.approx(
        [0, 49, 50, 51, 90], abs=0.01
    )
    assert loaded.get_beats()[51] == pytest.approx(
        TWO.seconds_at(51), abs=0.002
    )


def test_retiming_moves_notes_and_controllers():
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(90, 60, 30.0, 31.0))
    inst.pitch_bends.append(pretty_midi.PitchBend(100, 31.0))
    inst.control_changes.append(pretty_midi.ControlChange(64, 127, 30.0))
    retime_instruments([inst], lambda t: TWO.warp(t, 120.0))
    assert (inst.notes[0].start, inst.notes[0].end) == pytest.approx(
        (25.0, 26.0)
    )
    assert inst.pitch_bends[0].time == pytest.approx(26.0)
    assert inst.control_changes[0].time == pytest.approx(25.0)
