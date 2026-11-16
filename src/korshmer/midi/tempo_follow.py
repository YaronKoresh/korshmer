from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import pretty_midi

from korshmer.midi.tempo_map import TempoMap, retime_instruments
from korshmer.midi_algorithms import fit_beat_grid

FINE_DIVISIONS = 4
SECTION_SEARCH = 0.012
SECTION_MIN_ONSETS = 16
SECTION_MIN_ALIGNMENT = 0.25
SECTION_MIN_BEATS = 8
BRIDGE_POINTS = 257
BRIDGE_MIN_ONSETS = 12
BRIDGE_MIN_GAIN = 0.05
BRIDGE_WIDE_BENDS = np.linspace(-0.6, 0.6, 31)
BRIDGE_NARROW_BENDS = np.linspace(-0.3, 0.3, 13)


def _section_beats(section, onsets):
    begin, end, bpm = section
    inside = onsets[(onsets >= begin) & (onsets <= end)]
    fine = fit_beat_grid(
        inside,
        bpm,
        FINE_DIVISIONS,
        search=SECTION_SEARCH,
        min_onsets=SECTION_MIN_ONSETS,
        min_alignment=SECTION_MIN_ALIGNMENT,
    )
    if fine is None:
        return None
    tempo, fine_offset, _ = fine
    beat = 60.0 / tempo
    step = beat / FINE_DIVISIONS
    coarse = fit_beat_grid(
        inside, tempo, 1, search=0.0, min_onsets=1, min_alignment=0.0
    )
    origin = fine_offset
    if coarse is not None:
        origin += round((coarse[1] - fine_offset) / step) * step
    first = origin + math.ceil((begin - origin) / beat) * beat
    count = int(math.floor((end - first) / beat))
    if count < SECTION_MIN_BEATS:
        return None
    return first, first + count * beat, count, tempo


def _onset_fit(phase, grid, onsets):
    at = np.interp(onsets, grid, phase)
    return float(np.cos(2 * np.pi * FINE_DIVISIONS * at).mean())


def _bent_phase(phase, grid, onsets):
    inside = onsets[(onsets > grid[0]) & (onsets < grid[-1])]
    if len(inside) < BRIDGE_MIN_ONSETS:
        return phase
    along = (grid - grid[0]) / (grid[-1] - grid[0])
    best, best_fit = phase, _onset_fit(phase, grid, inside) + BRIDGE_MIN_GAIN
    for wide in BRIDGE_WIDE_BENDS:
        for narrow in BRIDGE_NARROW_BENDS:
            bent = (
                phase
                + wide * np.sin(np.pi * along)
                + narrow * np.sin(2 * np.pi * along)
            )
            if np.any(np.diff(bent) <= 0):
                continue
            fit = _onset_fit(bent, grid, inside)
            if fit > best_fit:
                best, best_fit = bent, fit
    return best


def _bridge(anchors, target_time, profile, onsets):
    time, beat = anchors[-1]
    grid = np.linspace(time, target_time, BRIDGE_POINTS)
    local = np.interp(grid, profile.times, profile.bpms)
    rate = (local[1:] + local[:-1]) / 120.0
    phase = np.concatenate([[0.0], np.cumsum(rate * np.diff(grid))])
    count = max(1, int(round(phase[-1])))
    phase = _bent_phase(phase * (count / phase[-1]), grid, onsets)
    marks = np.interp(np.arange(1, count), phase, grid)
    for index, mark in enumerate(marks, start=1):
        anchors.append((float(mark), beat + index))
    anchors.append((float(target_time), beat + count))


def follow_tempo(profile, onset_times) -> TempoMap:
    onsets = np.sort(np.asarray(list(onset_times), dtype=float))
    onsets = onsets[np.isfinite(onsets)]
    anchors: list[tuple[float, float]] = []
    for section in profile.sections:
        found = _section_beats(section, onsets)
        if found is None:
            continue
        first, last, count, tempo = found
        if not anchors:
            anchors.append((first, 0.0))
        else:
            beat = 60.0 / tempo
            while first <= anchors[-1][0] + beat / 2 and count > 1:
                first += beat
                count -= 1
            if first <= anchors[-1][0]:
                continue
            _bridge(anchors, first, profile, onsets)
        anchors.append((last, anchors[-1][1] + count))
    if len(anchors) < 2:
        return TempoMap.uniform(profile.main_bpm)
    return TempoMap.from_anchors(anchors)


@dataclass(frozen=True)
class TempoFollower:
    tempo_map: TempoMap
    bpm: float

    def warp(self, midis):
        for midi in midis:
            retime_instruments(
                midi.instruments, lambda t: self.tempo_map.warp(t, self.bpm)
            )

    def real_time(self, seconds):
        return float(self.tempo_map.unwarp(seconds, self.bpm))

    def write(self, instruments, path, tempo):
        pm = self.tempo_map.scaled(tempo / self.bpm).new_pretty_midi()
        pm.instruments = retime_instruments(
            copy.deepcopy(list(instruments)), self.real_time
        )
        pm.write(path)

    def summary(self):
        bpms = self.tempo_map.bpms
        return (
            f"Following the tempo of the song "
            f"({min(bpms):.1f} to {max(bpms):.1f} BPM)"
        )


def follow_song_tempo(profile, midis, bpm):
    if profile is None or not profile.changes:
        return None
    onsets = [
        note.start
        for midi in midis
        for inst in midi.instruments
        for note in inst.notes
    ]
    tempo_map = follow_tempo(profile, onsets)
    if tempo_map.is_uniform:
        return None
    follower = TempoFollower(tempo_map, float(bpm))
    follower.warp(midis)
    return follower


def write_midi(follower, instruments, path, tempo):
    if follower is not None:
        follower.write(instruments, path, tempo)
        return
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    pm.instruments = list(instruments)
    pm.write(path)


def real_offset(follower, offset):
    if follower is None or offset is None:
        return offset
    return round(follower.real_time(offset), 4)
