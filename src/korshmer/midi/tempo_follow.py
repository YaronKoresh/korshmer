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
LOOSE_SEARCH = 0.003
PIECE_MAX_S = 32.0
BRIDGE_POINTS = 257
BRIDGE_MIN_ONSETS = 12
BRIDGE_MIN_GAIN = 0.05
BRIDGE_WIDE_BENDS = np.linspace(-0.6, 0.6, 31)
BRIDGE_NARROW_BENDS = np.linspace(-0.3, 0.3, 13)


def _pieces(sections, breaks):
    pieces = []
    for begin, end, bpm in sections:
        inside = sorted(b for b in breaks if begin < b < end)
        for first, last in zip([begin, *inside], [*inside, end], strict=True):
            parts = max(1, math.ceil((last - first) / PIECE_MAX_S))
            edges = np.linspace(first, last, parts + 1)
            pieces.extend(
                (float(a), float(b), bpm)
                for a, b in zip(edges, edges[1:], strict=False)
            )
    return pieces


def _piece_grid(piece, onsets):
    begin, end, bpm = piece
    inside = onsets[(onsets >= begin) & (onsets <= end)]
    if len(inside) < SECTION_MIN_ONSETS:
        return None
    fine = fit_beat_grid(
        inside,
        bpm,
        FINE_DIVISIONS,
        search=SECTION_SEARCH,
        min_onsets=SECTION_MIN_ONSETS,
        min_alignment=SECTION_MIN_ALIGNMENT,
    ) or fit_beat_grid(
        inside,
        bpm,
        FINE_DIVISIONS,
        search=LOOSE_SEARCH,
        min_onsets=SECTION_MIN_ONSETS,
        min_alignment=0.0,
    )
    if fine is None:
        return None
    return fine[0], fine[1], inside


def _curve_beats(profile, begin, end):
    grid = np.linspace(begin, end, BRIDGE_POINTS)
    local = np.interp(grid, profile.times, profile.bpms)
    rate = (local[1:] + local[:-1]) / 120.0
    return np.concatenate([[0.0], np.cumsum(rate * np.diff(grid))]), grid


def _first_beat(piece, tempo, fine_offset, inside, anchors, profile):
    begin = piece[0]
    beat = 60.0 / tempo
    step = beat / FINE_DIVISIONS
    if not anchors:
        coarse = fit_beat_grid(
            inside, tempo, 1, search=0.0, min_onsets=1, min_alignment=0.0
        )
        origin = fine_offset
        if coarse is not None:
            origin += round((coarse[1] - fine_offset) / step) * step
        return origin + math.ceil((begin - origin) / beat) * beat
    last = anchors[-1][0]
    start = (
        fine_offset + math.ceil((max(begin, last) - fine_offset) / step) * step
    )
    choices = start + step * np.arange(FINE_DIVISIONS)
    travelled = np.array(
        [_curve_beats(profile, last, c)[0][-1] for c in choices]
    )
    first = float(
        choices[int(np.argmin(np.abs(travelled - np.round(travelled))))]
    )
    while _curve_beats(profile, last, first)[0][-1] < 0.5:
        first += beat
    return first


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
    phase, grid = _curve_beats(profile, time, target_time)
    count = max(1, int(round(phase[-1])))
    phase = _bent_phase(phase * (count / phase[-1]), grid, onsets)
    marks = np.interp(np.arange(1, count), phase, grid)
    for index, mark in enumerate(marks, start=1):
        anchors.append((float(mark), beat + index))
    anchors.append((float(target_time), beat + count))


def follow_tempo(profile, onset_times, breaks=()) -> TempoMap:
    onsets = np.sort(np.asarray(list(onset_times), dtype=float))
    onsets = onsets[np.isfinite(onsets)]
    anchors: list[tuple[float, float]] = []
    for piece in _pieces(profile.sections, breaks):
        found = _piece_grid(piece, onsets)
        if found is None:
            continue
        tempo, fine_offset, inside = found
        beat = 60.0 / tempo
        first = _first_beat(piece, tempo, fine_offset, inside, anchors, profile)
        count = int(math.floor((piece[1] - first) / beat))
        if count < SECTION_MIN_BEATS:
            continue
        if anchors:
            _bridge(anchors, first, profile, onsets)
        else:
            anchors.append((first, 0.0))
        anchors.append((first + count * beat, anchors[-1][1] + count))
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


def follow_song_tempo(profile, midis, bpm, breaks=()):
    if profile is None or not profile.changes:
        return None
    onsets = [
        note.start
        for midi in midis
        for inst in midi.instruments
        for note in inst.notes
    ]
    tempo_map = follow_tempo(profile, onsets, breaks)
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
