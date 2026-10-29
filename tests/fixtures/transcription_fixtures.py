import csv
import random
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
from scipy import signal

from korshmer.audio_processing.music_generation import MusicGenerator

SAMPLE_RATE = 22050
DURATION = 8.0
TEMPO = 120.0
BARS = 4
FIXTURE_DIR = Path(__file__).parent / "transcription"

PROGRESSIONS = {
    "major": [[0, 4, 7], [5, 9, 0], [7, 11, 2], [0, 4, 7]],
    "minor": [[0, 3, 7], [8, 0, 3], [10, 2, 5], [0, 3, 7]],
}


SONGS = (
    ("piano_c_major", 0, "major", "piano", 2, ("melody", "harmony", "chords")),
    ("bass_a_minor", 9, "minor", "bass", 1, ("bass",)),
    ("guitar_g_major", 7, "major", "pluck", 1, ("melody", "chords")),
    ("pad_d_dorian", 2, "dorian", "pad", 2, ("chords", "harmony")),
    ("lead_e_minor", 4, "minor", "lead", 2, ("melody", "harmony", "bass")),
    ("arp_f_major", 5, "major", "pluck", 1, ("arp", "bass")),
    (
        "full_mix_bb_major",
        10,
        "major",
        "piano",
        2,
        ("melody", "harmony", "chords", "bass", "hats"),
    ),
)

HARMONICS = {
    "piano": [1.0, 0.55, 0.35, 0.2, 0.12, 0.06],
    "bass": [1.0, 0.7, 0.3, 0.12],
    "pluck": [1.0, 0.8, 0.5, 0.3, 0.2, 0.1],
    "pad": [1.0, 0.4, 0.25, 0.1],
    "lead": [1.0, 0.6, 0.45, 0.3, 0.2],
}
DECAY = {"piano": 2.0, "bass": 1.2, "pluck": 4.0, "pad": 0.0, "lead": 0.5}


def _progression(scale):
    return PROGRESSIONS["minor" if scale in ("minor", "dorian") else "major"]


def _scale_pitches(song_scale):
    return MusicGenerator().scale_patterns[song_scale]


def _melody(gen, tonic, scale, seed):
    random.seed(seed)
    midi = gen.generate_melody_from_chords(
        _progression(scale),
        key=60 + tonic - (12 if tonic > 6 else 0),
        scale=scale,
        tempo=TEMPO,
        duration_bars=BARS,
    )
    return midi


def _clean_notes(notes, floor=0.1):
    result = []
    for n in sorted(notes, key=lambda n: (n[0], n[1])):
        if n[2] - n[1] >= floor:
            result.append(n)
    return result


def _notes(midi):
    return [
        (n.pitch, n.start, n.end, n.velocity)
        for inst in midi.instruments
        for n in inst.notes
    ]


def _layer_notes(layer, gen, tonic, scale, melody, seed):
    bar = 60.0 / TEMPO * 4
    prog = _progression(scale)
    base = 48 + tonic - (12 if tonic > 6 else 0)
    if layer == "melody":
        return _notes(melody)
    if layer == "harmony":
        degree = 2 if layer == "harmony" else 4
        harmony = gen.generate_harmony(
            melody, interval=degree, key=tonic, scale=scale
        )
        return [(p - 12, s, e, v) for p, s, e, v in _notes(harmony)]
    if layer == "bass":
        bass = gen.generate_bass_line(
            prog,
            key=base - 12,
            tempo=TEMPO,
            duration_bars=BARS,
            rhythm_pattern="syncopated",
        )
        return _notes(bass)
    if layer == "chords":
        out = []
        for i, chord in enumerate(prog):
            for interval in chord:
                out.append((base + interval, i * bar, (i + 1) * bar - 0.05, 70))
        return out
    if layer == "arp":
        out = []
        step = bar / 8
        t = 0.0
        for chord in prog:
            for k in range(8):
                pitch = base + 12 + chord[k % 3] + (12 if k % 6 >= 3 else 0)
                out.append((pitch, t, t + step * 0.9, 80))
                t += step
        return out
    return []


def song_notes(name):
    index = [s[0] for s in SONGS].index(name)
    _, tonic, scale, _, _, layers = SONGS[index]
    gen = MusicGenerator()
    melody = _melody(gen, tonic, scale, seed=100 + index)
    notes = []
    for layer in layers:
        if layer == "hats":
            continue
        notes += _layer_notes(layer, gen, tonic, scale, melody, 100 + index)
    notes = [
        (int(p), round(float(s), 4), round(float(min(e, DURATION - 0.1)), 4), v)
        for p, s, e, v in notes
        if s < DURATION - 0.3
    ]
    return _clean_notes(notes)


def _tone(pitch, length, timbre, velocity):
    n = int(length * SAMPLE_RATE) + int(0.15 * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    freq = 440.0 * 2 ** ((pitch - 69) / 12)
    wave = np.zeros(n)
    for k, amp in enumerate(HARMONICS[timbre], start=1):
        if freq * k < SAMPLE_RATE / 2 - 500:
            wave += amp * np.sin(2 * np.pi * freq * k * t)
    attack = int(0.008 * SAMPLE_RATE)
    env = np.ones(n)
    env[:attack] = np.linspace(0, 1, attack)
    if DECAY[timbre] > 0:
        env *= np.exp(-t * DECAY[timbre])
    release = int(0.12 * SAMPLE_RATE)
    end = int(length * SAMPLE_RATE)
    env[end : end + release] *= np.linspace(1, 0, min(release, n - end))
    env[end + release :] = 0
    return wave * env * (velocity / 127.0) * 0.25


def _hats(rng):
    out = np.zeros(int(DURATION * SAMPLE_RATE))
    sos = signal.butter(4, 7000, "hp", fs=SAMPLE_RATE, output="sos")
    for k in range(int(DURATION * 4)):
        start = int(k * 0.25 * SAMPLE_RATE)
        length = int(0.04 * SAMPLE_RATE)
        burst = signal.sosfilt(sos, rng.standard_normal(length))
        out[start : start + length] += burst * np.linspace(1, 0, length) * 0.12
    return out


def master(audio):
    sos = signal.butter(2, 35, "hp", fs=SAMPLE_RATE, output="sos")
    audio = signal.sosfiltfilt(sos, audio, axis=0)
    drive = 2.2 / max(np.max(np.abs(audio)), 1e-9)
    audio = np.tanh(audio * drive)
    return audio / np.max(np.abs(audio)) * 10 ** (-1 / 20)


def render_song(name):
    index = [s[0] for s in SONGS].index(name)
    _, _, _, timbre, channels, layers = SONGS[index]
    notes = song_notes(name)
    rng = np.random.default_rng(7 + index)
    total = int(DURATION * SAMPLE_RATE)
    left = np.zeros(total)
    right = np.zeros(total)
    for i, (pitch, start, end, vel) in enumerate(notes):
        tone = _tone(pitch, end - start, timbre, vel)
        s = int(start * SAMPLE_RATE)
        tone = tone[: total - s]
        pan = 0.5 + 0.35 * np.sin(i * 1.7)
        left[s : s + len(tone)] += tone * (1 - pan)
        right[s : s + len(tone)] += tone * pan
    if "hats" in layers:
        hats = _hats(rng)
        left += hats
        right += hats * 0.8
    if channels == 1:
        mono = master((left + right) / 2)
        return mono.astype(np.float32)
    return master(np.stack([left, right], axis=1)).astype(np.float32)


def write_fixtures(directory=FIXTURE_DIR):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for song in SONGS:
        name = song[0]
        sf.write(
            directory / f"{name}.wav",
            render_song(name),
            SAMPLE_RATE,
            subtype="PCM_16",
        )
        with open(
            directory / f"{name}.csv", "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["pitch", "start", "end", "velocity"])
            writer.writerows(song_notes(name))


def load_truth(name, directory=FIXTURE_DIR):
    with open(
        Path(directory) / f"{name}.csv", newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    return [(int(r["pitch"]), float(r["start"]), float(r["end"])) for r in rows]


def song_key(name):
    for entry in SONGS:
        if entry[0] == name:
            return entry[1], entry[2]
    raise KeyError(name)


def note_scores(reference, estimated, onset_tolerance=0.05):
    ref = sorted(reference, key=lambda n: n[1])
    used = set()
    hits = 0
    for note in sorted(estimated, key=lambda n: n[1]):
        pitch, start = note[0], note[1]
        for i, r in enumerate(ref):
            if i in used or r[0] != pitch:
                continue
            if abs(r[1] - start) <= onset_tolerance:
                used.add(i)
                hits += 1
                break
    precision = hits / len(estimated) if estimated else 0.0
    recall = hits / len(ref) if ref else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def to_pretty_midi(notes, program=0):
    midi = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
    inst = pretty_midi.Instrument(program=program)
    for n in notes:
        inst.notes.append(
            pretty_midi.Note(
                velocity=int(n[3]) if len(n) > 3 else 90,
                pitch=int(n[0]),
                start=float(n[1]),
                end=float(n[2]),
            )
        )
    midi.instruments.append(inst)
    return midi


if __name__ == "__main__":
    write_fixtures()
