import numpy as np

from korshmer.audio_processing.drum_transcription import (
    HAT_CLOSED,
    SNARE,
    find_hits,
)

SAMPLE_RATE = 44100
BEAT = 60.0 / 110.0


def _stem():
    rng = np.random.default_rng(3)
    audio = np.zeros(int(SAMPLE_RATE * BEAT * 8) + SAMPLE_RATE)

    def add(start, burst, level):
        i = int(start * SAMPLE_RATE)
        audio[i : i + len(burst)] += level * burst[: len(audio) - i]

    def snare():
        t = np.arange(int(0.2 * SAMPLE_RATE)) / SAMPLE_RATE
        return 0.6 * rng.standard_normal(len(t)) * np.exp(-t * 25) + 0.4 * (
            np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30)
        )

    def hat():
        t = np.arange(int(0.06 * SAMPLE_RATE)) / SAMPLE_RATE
        noise = rng.standard_normal(len(t))
        return 0.25 * np.diff(noise, prepend=0) * np.exp(-t * 80)

    snares, hats = [], []
    for b in range(8):
        start = b * BEAT
        if b % 2 == 1:
            add(start, snare(), 1.0)
            snares.append(start)
        if b % 4 == 3:
            add(start + BEAT * 0.25, snare(), 0.25)
            snares.append(start + BEAT * 0.25)
        for j in range(4):
            level = 1.0 if j % 2 == 0 else 0.35
            add(start + j * BEAT / 4, hat(), level)
            hats.append(start + j * BEAT / 4)
    audio += 0.01 * rng.standard_normal(len(audio))
    return audio / np.abs(audio).max(), snares, hats


def _found(hits, pitch, times):
    seen = np.array([t for t, p, _, _ in hits if p == pitch])
    return sum(bool(np.any(np.abs(seen - t) < 0.04)) for t in times)


def test_quiet_ghost_snares_and_weak_hats_are_found():
    audio, snares, hats = _stem()
    hits = find_hits(audio, SAMPLE_RATE)
    assert _found(hits, SNARE, snares) == len(snares)
    assert _found(hits, HAT_CLOSED, hats) == len(hats)
