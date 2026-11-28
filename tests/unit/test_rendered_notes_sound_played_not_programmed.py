import statistics

import numpy as np
import pretty_midi

from korshmer import audio_processors
from korshmer.audio_processing import render_polish as rp

SR = 22050


def _part(program=0, drum=False, count=200, bend=False):
    inst = pretty_midi.Instrument(program, is_drum=drum)
    for index in range(count):
        start = index * 0.25
        inst.notes.append(pretty_midi.Note(90, 60, start, start + 0.2))
    if bend:
        inst.pitch_bends.append(pretty_midi.PitchBend(1000, 0.1))
    return inst


def _tone(seconds=2.0, hz=220.0):
    t = np.arange(int(SR * seconds)) / SR
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_played_notes_drift_a_little_in_time_and_strength():
    inst = _part()
    played = rp.humanized_notes(inst, "a")
    shifts = [p[0] - n.start for p, n in zip(played, inst.notes, strict=True)]
    assert max(abs(s) for s in shifts) <= rp.TIMING_LIMIT + 1e-9
    assert statistics.pstdev(shifts) >= 0.002
    velocities = [p[3] for p in played]
    assert len(set(velocities)) >= 5
    assert all(abs(v - 90) <= rp.VELOCITY_SPREAD for v in velocities)
    assert all(end >= start + rp.SHORTEST_NOTE for start, end, _p, _v in played)
    assert rp.humanized_notes(inst, "a") == played
    assert rp.humanized_notes(inst, "b") != played


def test_drums_and_bent_notes_keep_their_time():
    for inst in (_part(drum=True), _part(bend=True)):
        played = rp.humanized_notes(inst, "a", keep_time=bool(inst.pitch_bends))
        assert [p[0] for p in played] == [n.start for n in inst.notes]
    drums = rp.humanized_notes(_part(drum=True), "a")
    assert all(abs(p[3] - 90) <= rp.DRUM_VELOCITY_SPREAD for p in drums)
    loud = _part()
    for note in loud.notes:
        note.velocity = 127
    assert max(p[3] for p in rp.humanized_notes(loud, "a")) == 127


def test_each_sound_goes_to_the_bus_made_for_it():
    assert rp.bus_of(_part(drum=True)) == rp.DRUMS
    assert rp.bus_of(_part(33)) == rp.BASS
    assert rp.bus_of(_part(48)) == rp.PADS
    assert rp.bus_of(_part(89)) == rp.PADS
    assert rp.bus_of(_part(0)) == rp.BAND
    assert rp.bus_of(_part(25)) == rp.BAND
    assert rp.bus_of(_part(73)) == rp.LEAD
    low = _part(87)
    for note in low.notes:
        note.pitch = 36
    assert rp.bus_of(low) == rp.BASS
    assert rp.bus_of(_part(87)) == rp.LEAD


def test_a_part_panned_to_one_side_stays_on_that_side():
    left = np.stack([_tone(), np.zeros(int(SR * 2.0), np.float32)], axis=1)
    for bus in (rp.DRUMS, rp.BASS, rp.PADS, rp.BAND, rp.LEAD):
        out = rp.polish_bus(bus, left, SR, 120.0, [0.5], centred=False)
        assert out.shape == left.shape
        assert float(np.max(np.abs(out[:, 1]))) == 0.0
        assert float(np.max(np.abs(out[:, 0]))) > 0.0
    mixed = rp.polish_mix(left, SR)
    assert float(np.max(np.abs(mixed[:, 1]))) == 0.0


def test_centred_pads_and_band_parts_spread_out_like_a_room():
    mono = np.stack([_tone(), _tone()], axis=1)
    for bus in (rp.PADS, rp.BAND):
        out = rp.polish_bus(bus, mono, SR, 120.0)
        corr = np.corrcoef(out[:, 0], out[:, 1])[0, 1]
        assert corr < 0.97
        assert np.all(np.isfinite(out))


def test_the_lowest_bass_notes_sit_in_the_middle():
    t = np.arange(SR * 2) / SR
    wide = np.stack(
        [np.sin(2 * np.pi * 50 * t), -np.sin(2 * np.pi * 50 * t)], axis=1
    ).astype(np.float32)
    out = rp.mono_below(wide, SR, rp.MONO_BELOW_HZ)
    side = out[SR:, 0] - out[SR:, 1]
    assert np.sqrt(np.mean(side**2)) < 0.2 * np.sqrt(np.mean(wide[SR:, 0] ** 2))


def test_bass_and_pads_give_way_to_the_kick_and_come_back():
    steady = np.ones((SR, 2), np.float32)
    out = rp.duck(steady, SR, [0.25], rp.DUCK_DB[rp.BASS])
    at_kick = out[int(0.25 * SR) + 1, 0]
    assert at_kick < 10 ** (-2.5 / 20)
    assert out[int(0.1 * SR), 0] == 1.0
    assert out[-1, 0] > 0.97
    assert rp.duck(steady, SR, [], 3.0) is steady


def test_the_tape_stage_keeps_level_and_length():
    stereo = np.stack([_tone(), _tone(hz=330.0)], axis=1)
    out = rp.polish_mix(stereo, SR)
    assert out.shape == stereo.shape
    assert out.dtype == np.float32
    assert float(np.max(np.abs(out))) <= 1.05 * float(np.max(np.abs(stereo)))
    assert float(np.sqrt(np.mean(out**2))) > 0.5 * float(
        np.sqrt(np.mean(stereo**2))
    )
    empty = np.zeros((0, 2), np.float32)
    assert rp.polish_mix(empty, SR).shape == (0, 2)


class _TailSynth:
    made = 0

    def __init__(self, *args, **kwargs):
        type(self).made += 1
        self.loads = 0
        self.tail = 5

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def sf_load(self, path):
        self.loads += 1
        return 1

    def reset(self):
        self.tail = 3

    def get_samples(self, count):
        level = 0.1 if self.tail > 0 else 0.0
        self.tail -= 1
        return np.full(count, level), np.full(count, level)


def test_one_synth_serves_every_pass_and_each_pass_starts_silent(monkeypatch):
    _TailSynth.made = 0
    monkeypatch.setattr(audio_processors, "FluidSynth", _TailSynth)
    shared = audio_processors._SharedSynth("fluid", SR, "font.sf3")
    with shared.pass_synth() as first:
        pass
    with shared.pass_synth() as second:
        left, _right = second.get_samples(10)
    shared.close()
    assert first is second
    assert _TailSynth.made == 1
    assert first.loads == 1
    assert not left.any()
