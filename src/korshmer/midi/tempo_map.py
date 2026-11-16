from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pretty_midi

SAME_TEMPO = 1e-6
MAIN_TOLERANCE = 0.005
MIDI_RESOLUTION = 960


def _number(value, label):
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{label} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return value


@dataclass(frozen=True)
class TempoMap:
    starts: tuple[float, ...]
    bpms: tuple[float, ...]

    def __post_init__(self):
        starts = tuple(_number(s, "Section start") for s in self.starts)
        bpms = tuple(_number(b, "Section tempo") for b in self.bpms)
        if not starts or len(starts) != len(bpms):
            raise ValueError(
                "A tempo map needs one tempo for every section start, "
                f"got {len(starts)} starts and {len(bpms)} tempos"
            )
        if starts[0] != 0.0:
            raise ValueError(
                f"The first section must start at 0, got {starts[0]!r}"
            )
        if any(b <= a for a, b in zip(starts, starts[1:], strict=False)):
            raise ValueError(f"Section starts must increase, got {starts!r}")
        if any(bpm <= 0 for bpm in bpms):
            raise ValueError(f"Section tempos must be above 0, got {bpms!r}")
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "bpms", bpms)

    @classmethod
    def uniform(cls, bpm) -> TempoMap:
        return cls((0.0,), (bpm,))

    @classmethod
    def from_pairs(cls, pairs) -> TempoMap:
        starts: list[float] = []
        bpms: list[float] = []
        for start, bpm in pairs:
            start, bpm = float(start), float(bpm)
            if bpms and abs(bpm - bpms[-1]) <= SAME_TEMPO * bpms[-1]:
                continue
            starts.append(start if starts else 0.0)
            bpms.append(bpm)
        return cls(tuple(starts), tuple(bpms))

    @classmethod
    def from_anchors(cls, anchors) -> TempoMap:
        points = [(float(t), float(b)) for t, b in anchors]
        if len(points) < 2:
            raise ValueError(
                f"A tempo map needs at least two anchors, got {len(points)}"
            )
        pairs = []
        for (time, beat), (next_time, next_beat) in zip(
            points, points[1:], strict=False
        ):
            if next_time <= time or next_beat <= beat:
                raise ValueError(
                    "Anchors must move forward in time and in beats, "
                    f"got {(time, beat)!r} then {(next_time, next_beat)!r}"
                )
            pairs.append((time, 60.0 * (next_beat - beat) / (next_time - time)))
        return cls.from_pairs(pairs)

    @classmethod
    def from_pretty_midi(cls, pm) -> TempoMap:
        times, tempos = pm.get_tempo_changes()
        if len(tempos) == 0:
            return cls.uniform(120.0)
        return cls.from_pairs(zip(times, tempos, strict=True))

    @property
    def is_uniform(self) -> bool:
        return len(self.bpms) == 1

    def as_pairs(self) -> list[list[float]]:
        return [[s, b] for s, b in zip(self.starts, self.bpms, strict=True)]

    def _beat_starts(self) -> np.ndarray:
        spans = np.diff(self.starts) * np.asarray(self.bpms[:-1]) / 60.0
        return np.concatenate([[0.0], np.cumsum(spans)])

    def beats_at(self, seconds):
        seconds = np.asarray(seconds, dtype=float)
        starts = np.asarray(self.starts)
        index = np.clip(
            np.searchsorted(starts, seconds, side="right") - 1, 0, None
        )
        rate = np.asarray(self.bpms)[index] / 60.0
        beats = self._beat_starts()[index] + (seconds - starts[index]) * rate
        return float(beats) if beats.ndim == 0 else beats

    def seconds_at(self, beats):
        beats = np.asarray(beats, dtype=float)
        beat_starts = self._beat_starts()
        index = np.clip(
            np.searchsorted(beat_starts, beats, side="right") - 1, 0, None
        )
        length = 60.0 / np.asarray(self.bpms)[index]
        seconds = (
            np.asarray(self.starts)[index]
            + (beats - beat_starts[index]) * length
        )
        return float(seconds) if seconds.ndim == 0 else seconds

    def warp(self, seconds, bpm):
        return self.beats_at(seconds) * (60.0 / float(bpm))

    def unwarp(self, seconds, bpm):
        return self.seconds_at(np.asarray(seconds, float) * (float(bpm) / 60.0))

    def main_bpm(self, end=None) -> float:
        last = self.starts[-1] if end is None else max(end, self.starts[-1])
        spans = np.diff([*self.starts, last])
        if spans.sum() <= 0:
            return self.bpms[-1]
        bpms = np.asarray(self.bpms)
        weights = [
            spans[np.abs(bpms - bpm) <= MAIN_TOLERANCE * bpm].sum()
            for bpm in bpms
        ]
        centre = bpms[int(np.argmax(weights))]
        near = np.abs(bpms - centre) <= MAIN_TOLERANCE * centre
        return float(np.average(bpms[near], weights=spans[near]))

    def scaled(self, factor) -> TempoMap:
        factor = _number(factor, "Tempo factor")
        if factor <= 0:
            raise ValueError(f"Tempo factor must be above 0, got {factor!r}")
        return TempoMap(self.starts, tuple(b * factor for b in self.bpms))

    def stretched(self, ratio) -> TempoMap:
        ratio = _number(ratio, "Time ratio")
        if ratio <= 0:
            raise ValueError(f"Time ratio must be above 0, got {ratio!r}")
        return TempoMap(
            tuple(s * ratio for s in self.starts),
            tuple(b / ratio for b in self.bpms),
        )

    def apply_to(self, pm):
        scales = []
        tick = 0
        for index, bpm in enumerate(self.bpms):
            scale = 60.0 / (bpm * pm.resolution)
            if scales and tick <= scales[-1][0]:
                scales[-1] = (scales[-1][0], scale)
            else:
                scales.append((tick, scale))
            if index + 1 < len(self.starts):
                span = self.starts[index + 1] - self.starts[index]
                tick += int(round(span / scale))
        pm._tick_scales = scales
        pm._update_tick_to_time(tick + 1)
        return pm

    def new_pretty_midi(self):
        pm = pretty_midi.PrettyMIDI(
            resolution=MIDI_RESOLUTION, initial_tempo=self.bpms[0]
        )
        return self.apply_to(pm)


def retime_instruments(instruments, convert):
    for inst in instruments:
        for note in inst.notes:
            start = float(convert(note.start))
            end = float(convert(note.end))
            note.start, note.end = start, max(end, start)
        for event in (*inst.pitch_bends, *inst.control_changes):
            event.time = float(convert(event.time))
    return instruments
