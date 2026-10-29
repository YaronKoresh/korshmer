import numpy as np

SAMPLE_RATE = 44100


def _kick(sr):
    n = int(0.25 * sr)
    t = np.arange(n) / sr
    freq = 120 * np.exp(-t * 25) + 45
    return np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-t * 14)


def _snare(sr):
    n = int(0.2 * sr)
    t = np.arange(n) / sr
    noise = np.random.default_rng(1).standard_normal(n)
    return (noise * 0.7 + np.sin(2 * np.pi * 190 * t) * 0.5) * np.exp(-t * 22)


def _hat(sr):
    n = int(0.06 * sr)
    t = np.arange(n) / sr
    noise = np.diff(np.random.default_rng(2).standard_normal(n), prepend=0)
    return noise * np.exp(-t * 60) * 0.4


def _open_hat(sr):
    n = int(0.35 * sr)
    t = np.arange(n) / sr
    noise = np.diff(np.random.default_rng(3).standard_normal(n), prepend=0)
    return noise * np.exp(-t * 9) * 0.35


def _tom(sr, freq):
    n = int(0.3 * sr)
    t = np.arange(n) / sr
    sweep = freq * (1 + 0.25 * np.exp(-t * 30))
    return np.sin(2 * np.pi * np.cumsum(sweep) / sr) * np.exp(-t * 11)


def make_drum_loop(
    bpm=120.0, bars=4, sr=SAMPLE_RATE, fills=False, open_hats=False
):
    beat = 60.0 / bpm
    total = int(sr * (bars * 4 * beat + 1.0))
    audio = np.zeros(total, dtype=np.float32)
    truth = []

    def put(sound, when, label):
        start = int(round(when * sr))
        end = min(total, start + len(sound))
        audio[start:end] += sound[: end - start].astype(np.float32)
        truth.append((label, when))

    for index in range(bars * 4):
        start = index * beat
        bar, in_bar = divmod(index, 4)
        if fills and bar % 4 == 3 and in_bar == 3:
            for k, freq in enumerate((320.0, 240.0, 170.0)):
                put(_tom(sr, freq), start + k * beat / 3, "tom")
            continue
        put(
            _kick(sr) if in_bar % 2 == 0 else _snare(sr),
            start,
            "kick" if in_bar % 2 == 0 else "snare",
        )
        put(_hat(sr), start, "hat")
        if open_hats and in_bar == 3:
            put(_open_hat(sr), start + beat / 2, "open_hat")
        else:
            put(_hat(sr), start + beat / 2, "hat")
    peak = float(np.max(np.abs(audio)))
    audio *= np.float32(0.9 / max(peak, 1e-9))
    return audio, truth
