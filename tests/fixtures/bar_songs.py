import numpy as np

SAMPLE_RATE = 22050


def make_bar_song(
    bpm=120.0,
    beats_per_bar=4,
    pickup_beats=0,
    full_bars=6,
    tail_beats=0,
    lead_seconds=0.0,
    sr=SAMPLE_RATE,
):
    beat = 60.0 / bpm
    total_beats = pickup_beats + full_bars * beats_per_bar + tail_beats
    audio = np.zeros(int((lead_seconds + total_beats * beat + 0.5) * sr))
    length = int(0.08 * sr)
    t = np.arange(length) / sr
    strong = np.sin(2 * np.pi * 70 * t) * np.exp(-t * 40)
    light = 0.27 * np.sin(2 * np.pi * 1500 * t) * np.exp(-t * 40)
    bar_lines = []
    for index in range(total_beats):
        at = lead_seconds + index * beat
        on_bar = (
            index >= pickup_beats
            and (index - pickup_beats) % (beats_per_bar) == 0
        )
        if on_bar:
            bar_lines.append(at)
        start = int(round(at * sr))
        audio[start : start + length] += strong if on_bar else light
    return audio.astype(np.float32), {
        "bar_lines": np.array(bar_lines),
        "pickup_beats": pickup_beats,
        "last_bar_beats": tail_beats or beats_per_bar,
        "first_downbeat": lead_seconds + pickup_beats * beat,
    }


def _chord(freqs, seconds, sr):
    t = np.arange(int(round(seconds * sr))) / sr
    return sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs) * 0.5


CHORD_NOTES = {
    "C": [261.63, 329.63, 392.0],
    "F": [174.61, 220.0, 261.63],
    "G": [196.0, 246.94, 293.66],
}


def make_chord_song(
    pickup_chord="G",
    bar_chords=("C", "F", "G", "C"),
    tail_chord="F",
    tail_beats=2,
    bpm=120.0,
    sr=SAMPLE_RATE,
):
    beat = 60.0 / bpm
    parts = [_chord(CHORD_NOTES[pickup_chord], beat, sr)]
    for name in bar_chords:
        parts.append(_chord(CHORD_NOTES[name], 4 * beat, sr))
    parts.append(_chord(CHORD_NOTES[tail_chord], tail_beats * beat, sr))
    audio = np.concatenate(parts)
    length = int(0.08 * sr)
    t = np.arange(length) / sr
    hit = np.sin(2 * np.pi * 70 * t) * np.exp(-t * 40)
    for bar in range(len(bar_chords) + 1):
        at = int(round((beat + bar * 4 * beat) * sr))
        audio[at : at + length] += hit
    return audio.astype(np.float32)
