import numpy as np
from scipy.signal import butter, sosfilt

SR = 22050
RIFF = [40, 43, 45, 40, 47, 45, 43, 38]


def _hz(pitch):
    return 440.0 * 2 ** ((pitch - 69) / 12)


def plucked(seconds=12.0, step=0.25, seed=0, pitches=RIFF):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    out = np.zeros(n)
    for i, start in enumerate(np.arange(0, seconds, step)):
        period = max(2, int(SR / _hz(pitches[i % len(pitches)] + 12)))
        length = min(int(1.2 * SR), n - int(start * SR))
        buf = rng.uniform(-1, 1, period)
        note = np.empty(length)
        for k in range(length):
            note[k] = buf[k % period]
            nxt = buf[(k + 1) % period]
            buf[k % period] = 0.996 * 0.5 * (buf[k % period] + nxt)
        first = int(start * SR)
        out[first : first + length] += note
    return 0.5 * out / np.abs(out).max()


def distorted(seconds=12.0):
    clean = plucked(seconds, step=0.5)

    hot = np.tanh(30 * clean / np.abs(clean).max())
    nyq = SR / 2
    hot = sosfilt(butter(4, 5000 / nyq, "lowpass", output="sos"), hot)
    return 0.5 * hot / np.abs(hot).max()


def _saw(freq, count):
    t = np.arange(count) / SR
    return 2 * ((t * freq) % 1.0) - 1


def soft_strings(seconds=12.0):
    n = int(seconds * SR)
    out = np.zeros(n)
    chords = [(57, 60, 64), (53, 57, 60)]
    bar = int(3.0 * SR)
    for i, first in enumerate(range(0, n, bar)):
        count = min(bar, n - first)
        env = np.minimum(1, np.arange(count) / (0.8 * SR))
        env *= np.minimum(1, (count - np.arange(count)) / (0.5 * SR))
        for p in chords[i % 2]:
            out[first : first + count] += _saw(_hz(p), count) * env
    nyq = SR / 2
    out = sosfilt(butter(2, 1200 / nyq, "lowpass", output="sos"), out)
    return 0.4 * out / np.abs(out).max()


def bright_lead(seconds=12.0):
    n = int(seconds * SR)
    out = np.zeros(n)
    note = int(1.5 * SR)
    for i, first in enumerate(range(0, n, note)):
        count = min(note, n - first)
        env = np.minimum(1, np.arange(count) / (0.15 * SR))
        out[first : first + count] = _saw(_hz(RIFF[i % 8] + 36), count) * env
    return 0.4 * out
