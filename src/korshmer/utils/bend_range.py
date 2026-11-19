import pretty_midi

DEFAULT_BEND_RANGE = 2
MAX_BEND_RANGE = 24
BEND_LIMIT = 8191
RPN_MSB = 101
RPN_LSB = 100
DATA_ENTRY = 6
RANGE_SETUP_GAP = 0.01
_RPN_NUMBERS = (RPN_MSB, RPN_LSB, DATA_ENTRY)
_ORDER = {RPN_MSB: 0, RPN_LSB: 1, DATA_ENTRY: 2}


def clamp_bend_range(semitones):
    try:
        value = int(round(float(semitones)))
    except (TypeError, ValueError):
        return DEFAULT_BEND_RANGE
    return max(1, min(MAX_BEND_RANGE, value))


def clamp_bend(value):
    return max(-BEND_LIMIT - 1, min(BEND_LIMIT, int(round(value))))


def bend_range_of(instrument):
    selected = {RPN_MSB: 127, RPN_LSB: 127}
    semitones = DEFAULT_BEND_RANGE
    changes = sorted(
        (c for c in instrument.control_changes if c.number in _ORDER),
        key=lambda c: (c.time, _ORDER[c.number]),
    )
    for change in changes:
        if change.number != DATA_ENTRY:
            selected[change.number] = change.value
        elif selected[RPN_MSB] == 0 and selected[RPN_LSB] == 0:
            semitones = clamp_bend_range(change.value)
    return semitones


def set_bend_range(instrument, semitones, at=0.0):
    instrument.control_changes = [
        c for c in instrument.control_changes if c.number not in _RPN_NUMBERS
    ]
    instrument.control_changes += [
        pretty_midi.ControlChange(RPN_MSB, 0, at),
        pretty_midi.ControlChange(RPN_LSB, 0, at),
        pretty_midi.ControlChange(
            DATA_ENTRY, clamp_bend_range(semitones), at + RANGE_SETUP_GAP
        ),
    ]
    instrument.control_changes.sort(key=lambda c: c.time)


def bend_for(semitones, bend_range):
    return clamp_bend(float(semitones) / bend_range * (BEND_LIMIT + 1))


def bent_spans(bends):
    spans = []
    opened = None
    for time, value in sorted(bends, key=lambda b: b[0]):
        if value and opened is None:
            opened = time
        elif not value and opened is not None:
            spans.append((opened, time))
            opened = None
    if opened is not None:
        spans.append((opened, float("inf")))
    return spans


def _bends_reach(bends, spans):
    return any(
        start < time <= end for time, _value in bends for start, end in spans
    )


def _bent_notes(spans, notes):
    return any(
        start < span_end and end > span_start
        for span_start, span_end in spans
        for start, end in notes
    )


def bends_clash(notes_a, bends_a, notes_b, bends_b):
    if not bends_a and not bends_b:
        return False
    spans_a = bent_spans(bends_a)
    spans_b = bent_spans(bends_b)
    return (
        _bent_notes(spans_a, notes_b)
        or _bent_notes(spans_b, notes_a)
        or _bends_reach(bends_a, spans_b)
        or _bends_reach(bends_b, spans_a)
    )
