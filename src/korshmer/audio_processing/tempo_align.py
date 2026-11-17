import hashlib
import json
import os

import numpy as np
import soundfile as sf

WINDOW_S = 0.046
REACH_S = 0.012
ALIGNED_SUBTYPE = "PCM_24"


def _read(path):
    try:
        data, rate = sf.read(path, dtype="float32", always_2d=True)
        return data.T, rate
    except RuntimeError:
        import librosa

        data, rate = librosa.load(path, sr=None, mono=False)
        return np.atleast_2d(data), int(rate)


def _source_positions(count, hop, rate, start, tempo_map, bpm):
    first = tempo_map.warp(start, bpm)
    even = first + np.arange(count) * (hop / rate)
    return np.round((tempo_map.unwarp(even, bpm) - start) * rate).astype(int)


def _best_shift(mono, energy, wanted, natural, hop, reach):
    template = mono[natural : natural + hop]
    region = mono[wanted - reach : wanted + reach + hop]
    match = np.correlate(region, template, "valid")
    power = energy[wanted - reach + hop : wanted + reach + hop + 1]
    power = power - energy[wanted - reach : wanted + reach + 1]
    return int(np.argmax(match / np.sqrt(power + 1e-9))) - reach


def stretch_to_tempo(audio, rate, start, tempo_map, bpm):
    audio = np.atleast_2d(np.asarray(audio, dtype=np.float32))
    seconds = audio.shape[1] / rate
    length = tempo_map.warp(start + seconds, bpm) - tempo_map.warp(start, bpm)
    samples = max(1, int(round(length * rate)))
    hop = max(1, int(WINDOW_S * rate) // 2)
    size = 2 * hop
    reach = max(1, int(REACH_S * rate))
    pad = size + reach
    padded = np.pad(audio, ((0, 0), (pad, pad + size)))
    mono = padded.mean(axis=0)
    energy = np.concatenate([[0.0], np.cumsum(mono.astype(np.float64) ** 2)])
    window = np.hanning(size + 1)[:-1].astype(np.float32)
    count = 2 + samples // hop
    wanted = _source_positions(count, hop, rate, start, tempo_map, bpm)
    wanted = np.clip(wanted - hop, -hop, audio.shape[1]) + pad
    out = np.zeros((audio.shape[0], (count + 1) * hop), dtype=np.float32)
    taken = None
    for index, position in enumerate(wanted):
        if taken is not None:
            natural = taken + hop
            if abs(natural - position) <= reach:
                position = natural
            else:
                position += _best_shift(
                    mono, energy, int(position), natural, hop, reach
                )
        taken = int(position)
        first = index * hop
        out[:, first : first + size] += padded[:, taken : taken + size] * window
    return out[:, hop : hop + samples]


def _aligned_name(path, start, tempo_map, bpm):
    stat = os.stat(path)
    key = json.dumps(
        [
            os.path.abspath(path),
            stat.st_mtime_ns,
            stat.st_size,
            round(float(start), 6),
            tempo_map.as_pairs(),
            float(bpm),
        ]
    )
    stem = os.path.splitext(os.path.basename(path))[0]
    return f"{stem}_even_{hashlib.sha1(key.encode()).hexdigest()[:8]}.wav"


def align_clip_to_tempo(path, start, tempo_map, bpm, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, _aligned_name(path, start, tempo_map, bpm))
    if not os.path.isfile(target):
        audio, rate = _read(path)
        even = stretch_to_tempo(audio, rate, start, tempo_map, bpm)
        partial = target + ".part"
        sf.write(partial, even.T, rate, subtype=ALIGNED_SUBTYPE, format="WAV")
        os.replace(partial, target)
    info = sf.info(target)
    return target, info.frames / info.samplerate
