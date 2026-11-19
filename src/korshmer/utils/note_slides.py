import bisect
import math

from korshmer.utils.bend_range import (
    BEND_LIMIT,
    MAX_BEND_RANGE,
    bend_for,
    clamp_bend,
)

EDGE = 1e-9


def _pairs(points):
    kept = []
    for entry in points or ():
        try:
            offset, semitones = entry
            offset, semitones = float(offset), float(semitones)
        except (TypeError, ValueError):
            continue
        if math.isfinite(offset) and math.isfinite(semitones):
            kept.append([offset, semitones])
    return kept


def _amount(semitones):
    return f"{semitones:+.1f}".replace(".0", "")


class NoteSlide:
    __slots__ = ("head", "tail")

    def __init__(self, head=(), tail=()):
        self.head = _pairs(head)
        self.tail = _pairs(tail)

    def __bool__(self):
        return bool(self.head or self.tail)

    def copy(self):
        return NoteSlide(self.head, self.tail)

    def to_dict(self):
        return {
            "head": [list(point) for point in self.head],
            "tail": [list(point) for point in self.tail],
        }

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            return None
        slide = cls(data.get("head"), data.get("tail"))
        return slide if slide else None

    def scaled(self, factor):
        return NoteSlide(
            [[offset * factor, value] for offset, value in self.head],
            [[offset * factor, value] for offset, value in self.tail],
        )

    def remapped(self, start, end, new_start, new_end, convert):
        return NoteSlide(
            [
                [float(convert(start + offset)) - new_start, value]
                for offset, value in self.head
            ],
            [
                [float(convert(end + offset)) - new_end, value]
                for offset, value in self.tail
            ],
        )

    def reach(self):
        return max(
            (abs(value) for _offset, value in (*self.head, *self.tail)),
            default=0.0,
        )

    def lead(self):
        return next((value for _offset, value in self.head if value), 0.0)

    def fall(self):
        return max(
            (value for _offset, value in self.tail), key=abs, default=0.0
        )

    def summary(self):
        parts = []
        if self.lead():
            parts.append(f"slides in from {_amount(self.lead())} st")
        if self.fall():
            parts.append(f"slides out to {_amount(self.fall())} st")
        return ", ".join(parts) or "slide"

    def points(self, start, end):
        lead_in = max((offset for offset, _value in self.head), default=0.0)
        lead_out = -min((offset for offset, _value in self.tail), default=0.0)
        inside = max(0.0, lead_in) + max(0.0, lead_out)
        length = end - start
        fit = 1.0 if inside <= length + EDGE else length / inside
        placed = [
            (start + (offset * fit if offset > 0 else offset), value)
            for offset, value in self.head
        ]
        placed += [
            (end + (offset * fit if offset < 0 else offset), value)
            for offset, value in self.tail
        ]
        placed = [(max(0.0, time), value) for time, value in placed]
        placed.sort(key=lambda point: point[0])
        return placed


def semitones_for(value, bend_range):
    return float(value) * bend_range / (BEND_LIMIT + 1)


def needed_range(notes, bend_range):
    reach = max(
        (note.slide.reach() for note in notes if note.slide), default=0.0
    )
    wanted = min(MAX_BEND_RANGE, math.ceil(reach - EDGE))
    return max(int(bend_range), wanted)


class _Spans:
    def __init__(self, spans):
        self.order = sorted(range(len(spans)), key=lambda i: spans[i][0])
        self.starts = [spans[i][0] for i in self.order]
        self.spans = spans
        self.longest = max((end - start for start, end in spans), default=0.0)

    def only_note(self, opened, closed):
        closed = max(closed, opened + 2 * EDGE)
        low = bisect.bisect_left(self.starts, opened - self.longest - EDGE)
        high = bisect.bisect_left(self.starts, closed - EDGE)
        found = [
            index
            for index in self.order[low:high]
            if self.spans[index][1] > opened + EDGE
        ]
        return found[0] if len(found) == 1 else None

    def alone(self, points, own):
        begin = next((time for time, value in points if value), None)
        if begin is None:
            return points
        low = bisect.bisect_left(self.starts, begin - self.longest - EDGE)
        high = bisect.bisect_right(self.starts, begin + EDGE)
        if any(
            index != own and self.spans[index][1] > begin + EDGE
            for index in self.order[low:high]
        ):
            return []
        cut = next(
            (
                self.starts[place]
                for place in range(high, len(self.order))
                if self.order[place] != own
            ),
            None,
        )
        if cut is None or cut >= points[-1][0] - EDGE:
            return points
        return [point for point in points if point[0] < cut] + [(cut, 0.0)]


def played_bends(notes, free, bend_range):
    wide = needed_range(notes, bend_range)
    events = [
        (time, clamp_bend(value * bend_range / wide)) for time, value in free
    ]
    ordered = sorted(notes, key=lambda note: (note.start, note.pitch))
    lookup = _Spans([(note.start, note.end) for note in ordered])
    for index, note in enumerate(ordered):
        if not note.slide:
            continue
        events += [
            (time, bend_for(value, wide))
            for time, value in lookup.alone(
                note.slide.points(note.start, note.end), index
            )
        ]
    events.sort(key=lambda event: event[0])
    return events, wide


def attach_slides(spans, bends, bend_range):
    lookup = _Spans(spans)
    slides = {}
    free = []
    count = len(bends)
    taken = 0
    index = 0
    while index < count:
        if not bends[index][1]:
            index += 1
            continue
        first = index
        close = first + 1
        while close < count and bends[close][1]:
            close += 1
        if close == count:
            break
        owner = lookup.only_note(bends[first][0], bends[close][0])
        if owner is None:
            index = close + 1
            continue
        start, end = spans[owner]
        low = first
        while (
            low > taken
            and not bends[low - 1][1]
            and start <= bends[low - 1][0] <= end
        ):
            low -= 1
        high = close + 1
        while high < count and not bends[high][1] and bends[high][0] <= end:
            high += 1
        free += bends[taken:low]
        slide = slides.setdefault(owner, NoteSlide())
        held = bends[close][0] >= end - EDGE
        for time, value in bends[low:high]:
            if held and time >= start:
                slide.tail.append(
                    [time - end, semitones_for(value, bend_range)]
                )
            else:
                slide.head.append(
                    [time - start, semitones_for(value, bend_range)]
                )
        taken = index = high
    free += bends[taken:]
    return slides, [list(point) for point in free]
