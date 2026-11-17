import os
import struct
from functools import lru_cache

from korshmer.constants import NOTE_NAMES, SF3_PATH
from korshmer.utils.track_layout import DRUMS_NAME, PROGRAM_NAMES

KEY_RANGE = 43
INSTRUMENT = 41
SAMPLE = 53


def _hydra(path):
    chunks = {}
    with open(path, "rb") as fh:
        if fh.read(4) != b"RIFF":
            return chunks
        fh.seek(12)
        while True:
            head = fh.read(8)
            if len(head) < 8:
                return chunks
            cid, size = struct.unpack("<4sI", head)
            if cid != b"LIST":
                fh.seek(size + (size & 1), 1)
                continue
            kind = fh.read(4)
            end = fh.tell() + size - 4
            if kind != b"pdta":
                fh.seek(end + (size & 1))
                continue
            while fh.tell() < end:
                sub, sub_size = struct.unpack("<4sI", fh.read(8))
                chunks[sub] = fh.read(sub_size + (sub_size & 1))[:sub_size]
            return chunks


def _zones(bags, gens, first, last):
    for bag in range(first, last):
        start = struct.unpack_from("<H", bags, bag * 4)[0]
        stop = struct.unpack_from("<H", bags, (bag + 1) * 4)[0]
        keys, target = None, None
        for index in range(start, stop):
            oper, low, high = struct.unpack_from("<HBB", gens, index * 4)
            if oper == KEY_RANGE:
                keys = (low, high)
            elif oper in (INSTRUMENT, SAMPLE):
                target = low | (high << 8)
        yield keys, target


def _meet(a, b):
    low, high = max(a[0], b[0]), min(a[1], b[1])
    return (low, high) if low <= high else None


def read_ranges(path):
    hydra = _hydra(path)
    need = (b"phdr", b"pbag", b"pgen", b"inst", b"ibag", b"igen")
    if any(name not in hydra for name in need):
        return {}
    inst_first = [
        struct.unpack_from("<H", hydra[b"inst"], i * 22 + 20)[0]
        for i in range(len(hydra[b"inst"]) // 22)
    ]
    inst_spans = {}
    for number in range(len(inst_first) - 1):
        general, spans = (0, 127), []
        zones = _zones(
            hydra[b"ibag"],
            hydra[b"igen"],
            inst_first[number],
            inst_first[number + 1],
        )
        for keys, sample in zones:
            if sample is None:
                general = keys or general
            else:
                spans.append(keys or general)
        inst_spans[number] = spans
    ranges = {}
    headers = hydra[b"phdr"]
    for number in range(len(headers) // 38 - 1):
        program, bank, first = struct.unpack_from(
            "<HHH", headers, number * 38 + 20
        )
        last = struct.unpack_from("<H", headers, (number + 1) * 38 + 24)[0]
        general, keys_heard = (0, 127), []
        for keys, instrument in _zones(
            hydra[b"pbag"], hydra[b"pgen"], first, last
        ):
            if instrument is None:
                general = keys or general
                continue
            for span in inst_spans.get(instrument, ()):
                met = _meet(keys or general, span)
                if met:
                    keys_heard.append(met)
        if keys_heard:
            ranges[(bank, program)] = (
                min(k[0] for k in keys_heard),
                max(k[1] for k in keys_heard),
            )
    return ranges


FULL_RANGE = (0, 127)
DRUM_BANK = 128


@lru_cache(maxsize=4)
def _cached_ranges(path, _stamp):
    try:
        return read_ranges(path)
    except (OSError, struct.error):
        return {}


def soundfont_ranges(path=None):
    path = str(path or SF3_PATH)
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        return {}
    return _cached_ranges(path, stamp)


def playable_range(program, is_drum=False, path=None):
    ranges = soundfont_ranges(path)
    if is_drum:
        keys = ranges.get((DRUM_BANK, int(program)))
        return keys or ranges.get((DRUM_BANK, 0), FULL_RANGE)
    return ranges.get((0, int(program)), FULL_RANGE)


def key_name(pitch):
    return f"{NOTE_NAMES[int(pitch) % 12]}{int(pitch) // 12 - 1}"


def range_text(keys):
    return f"{key_name(keys[0])} to {key_name(keys[1])}"


def instrument_label(track):
    if track.is_drum:
        return DRUMS_NAME
    return PROGRAM_NAMES[int(track.program) % len(PROGRAM_NAMES)]
