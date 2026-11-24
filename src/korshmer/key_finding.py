from dataclasses import dataclass

import numpy as np

from korshmer.constants import SCALES

MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

DETECTABLE_SCALES = (
    "Major",
    "Minor",
    "Dorian",
    "Phrygian",
    "Lydian",
    "Mixolydian",
    "Harmonic Minor",
    "Pentatonic Major",
    "Pentatonic Minor",
    "Blues",
    "Arabian (Maqam Hijaz)",
    "Gypsy Minor",
)
MAJOR_MINOR = ("Major", "Minor")
COMMON_KEYS = ("Major", "Minor", "Harmonic Minor")
COMMON_KEY_BONUS = 0.2
OUTSIDE_PENALTY = 3.0
MINOR_KEYS = ("Minor", "Harmonic Minor")
OUTSIDE_WEIGHT = 2.3
INSIDE_WEIGHT = 3.6
THIRD_WEIGHT = 4.2
FIFTH_WEIGHT = 5.0
TONIC_WEIGHT = 6.35
PROFILE_BIAS = 0.04
AUDIO_PROFILE_BIAS = 0.05

BASS_SHARE = 0.3
LAST_CHORD_SHARE = 0.6
FIRST_CHORD_SHARE = 0.4
LOW_PERCENTILE = 5
LOW_BAND = 14
LOW_NOTE_BONUS = 1.0
ROOT_BONUS = 2.0
EDGE_SHARE = 0.25


def scale_profile(name):
    if name == "Major":
        return MAJOR_PROFILE
    if name == "Minor":
        return MINOR_PROFILE
    intervals = SCALES[name]
    profile = np.full(12, OUTSIDE_WEIGHT)
    for interval in intervals:
        profile[interval] = INSIDE_WEIGHT
    thirds = [i for i in intervals if i in (3, 4)]
    if thirds:
        profile[thirds[0]] = THIRD_WEIGHT
    if 7 in intervals:
        profile[7] = FIFTH_WEIGHT
    profile[0] = TONIC_WEIGHT
    return profile


def _build_scale_profiles():
    profiles = {}
    for name in DETECTABLE_SCALES:
        base = scale_profile(name)
        rows = np.array([np.roll(base, root) for root in range(12)])
        rows = rows - rows.mean(axis=1, keepdims=True)
        profiles[name] = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    return profiles


_SCALE_PROFILES = _build_scale_profiles()


def _outside_rows(name):
    outside = np.ones(12)
    outside[SCALES[name]] = 0.0
    return np.array([np.roll(outside, root) for root in range(12)])


def _level_minor_with_relative_major():
    plain = np.zeros(12)
    plain[SCALES["Major"]] = 1.0
    plain = plain - plain.mean()
    plain = plain / np.linalg.norm(plain)
    major = float(_SCALE_PROFILES["Major"][0] @ plain)
    minor = float(_SCALE_PROFILES["Minor"][9] @ plain)
    return major - minor


RELATIVE_TIE_ROUNDING = 1e-9
RELATIVE_TIE = _level_minor_with_relative_major() + RELATIVE_TIE_ROUNDING
_OUTSIDE = {name: _outside_rows(name) for name in DETECTABLE_SCALES}


def _third(name):
    return 4 if 4 in SCALES[name] else 3


@dataclass(frozen=True)
class Sounded:
    pitch: int
    start: float
    end: float
    velocity: int = 100


def played_in_turn(pitches):
    return [
        Sounded(int(pitch), float(index), float(index + 1))
        for index, pitch in enumerate(pitches)
    ]


def note_chroma(notes):
    chroma = np.zeros(12)
    for note in notes:
        chroma[note.pitch % 12] += max(0.0, note.end - note.start) * (
            note.velocity / 127.0
        )
    return chroma


def _share(vector):
    total = float(vector.sum())
    return vector / total if total > 0 else vector


def _length_profile(notes, begin=None, end=None):
    vector = np.zeros(12)
    for note in notes:
        start = note.start if begin is None else max(note.start, begin)
        stop = note.end if end is None else min(note.end, end)
        if stop > start:
            vector[note.pitch % 12] += stop - start
    return vector


def _low_register(notes):
    floor = float(np.percentile([n.pitch for n in notes], LOW_PERCENTILE))
    return [n for n in notes if n.pitch <= floor + LOW_BAND]


def chord_support(weights):
    weights = _share(np.asarray(weights, dtype=float))
    support = np.zeros((2, 12))
    for row, third in enumerate((4, 3)):
        support[row] = (
            ROOT_BONUS * weights
            + np.roll(weights, -third)
            + np.roll(weights, -7)
        ) / ROOT_BONUS
    return support


def _chord_roots(notes, low, begin, end):
    return chord_support(
        _length_profile(notes, begin, end)
        + LOW_NOTE_BONUS * _length_profile(low, begin, end)
    )


def _closing_span(notes):
    finish = max(n.end for n in notes)
    closing = [
        n for n in notes if finish - n.end <= EDGE_SHARE * (n.end - n.start)
    ]
    return min(n.start for n in closing), finish


def _opening_span(notes):
    begin = min(n.start for n in notes)
    opening = [
        n for n in notes if n.start - begin <= EDGE_SHARE * (n.end - n.start)
    ]
    return begin, max(n.end for n in opening)


@dataclass(frozen=True)
class TonalCentre:
    bass: np.ndarray
    last_chord: np.ndarray
    first_chord: np.ndarray

    def support(self, third):
        row = 0 if third == 4 else 1
        return (
            BASS_SHARE * self.bass
            + LAST_CHORD_SHARE * self.last_chord[row]
            + FIRST_CHORD_SHARE * self.first_chord[row]
        )


def tonal_centre(notes):
    notes = [n for n in notes if n.end > n.start]
    if not notes:
        return None
    low = _low_register(notes)
    return TonalCentre(
        bass=_share(_length_profile(low)),
        last_chord=_chord_roots(notes, low, *_closing_span(notes)),
        first_chord=_chord_roots(notes, low, *_opening_span(notes)),
    )


def rank_keys(
    chroma,
    centre=None,
    limit=3,
    bias=PROFILE_BIAS,
    scales=DETECTABLE_SCALES,
    outside_penalty=OUTSIDE_PENALTY,
):
    cv = np.asarray(chroma, dtype=float)
    share = _share(np.clip(cv, 0.0, None))
    cv = cv - np.mean(cv)
    cv_norm = float(np.linalg.norm(cv))
    if cv_norm == 0:
        return [(0, "Major", 0.0)][:limit]
    cv = cv / cv_norm
    fits = np.array([_SCALE_PROFILES[name] @ cv for name in scales])
    for row, name in enumerate(scales):
        if name in MAJOR_MINOR:
            fits[row] += bias
    scores = fits.copy()
    if centre is not None:
        for row, name in enumerate(scales):
            scores[row] += centre.support(_third(name))
            scores[row] -= outside_penalty * (_OUTSIDE[name] @ share)
        for root in range(12):
            if scales[int(np.argmax(fits[:, root]))] in COMMON_KEYS:
                scores[:, root] += COMMON_KEY_BONUS
    scored = [
        (float(scores[row, root]), root, name)
        for row, name in enumerate(scales)
        for root in range(12)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    scored = _relative_minor_first(scored)
    return [(root, name, score) for score, root, name in scored[:limit]]


def _relative_minor_first(scored):
    top_score, top_root, top_name = scored[0]
    if top_name != "Major":
        return scored
    home = (top_root + 9) % 12
    for index, (score, root, name) in enumerate(scored):
        if top_score - score > RELATIVE_TIE:
            break
        if root == home and name in MINOR_KEYS:
            return [scored[index], *scored[:index], *scored[index + 1 :]]
    return scored


def key_fit(chroma, root, name):
    cv = np.asarray(chroma, dtype=float)
    cv = cv - np.mean(cv)
    cv_norm = float(np.linalg.norm(cv))
    if cv_norm == 0:
        return 0.0
    return float(_SCALE_PROFILES[name][root % 12] @ (cv / cv_norm))


def find_keys(notes, limit=3, scales=DETECTABLE_SCALES):
    notes = list(notes)
    return rank_keys(
        note_chroma(notes), tonal_centre(notes), limit, scales=scales
    )


def find_key(notes, scales=DETECTABLE_SCALES):
    root, name, _score = find_keys(notes, 1, scales)[0]
    return root, name


def midi_notes(midi):
    return [
        n for inst in midi.instruments if not inst.is_drum for n in inst.notes
    ]
