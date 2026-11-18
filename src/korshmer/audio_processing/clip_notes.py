import logging
import os
import tempfile

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def notes_audio(notes, bpm, sample_rate):
    from korshmer.audio_processors import render_midi_to_wav
    from korshmer.constants import FLUID_DIR, SF3_PATH

    if notes is None or not any(part.notes for part in notes.instruments):
        return None
    if not os.path.exists(SF3_PATH):
        return None
    with tempfile.TemporaryDirectory() as folder:
        written = os.path.join(folder, "notes.mid")
        notes.write(written)
        rendered = render_midi_to_wav(
            written,
            os.path.join(folder, "notes.wav"),
            str(FLUID_DIR),
            str(SF3_PATH),
            bpm=float(bpm),
        )
        if not rendered or not os.path.isfile(rendered):
            logger.warning("The notes could not be rendered for comparison")
            return None
        data, rate = sf.read(rendered, dtype="float32", always_2d=True)
    heard = data.mean(axis=1)
    if int(rate) != int(sample_rate):
        import librosa

        heard = librosa.resample(
            heard, orig_sr=int(rate), target_sr=int(sample_rate)
        )
    return np.asarray(heard, dtype=np.float32)


def beside_clip(heard, sample_rate, origin, frames):
    beside = np.zeros(int(frames), dtype=np.float32)
    if heard is None:
        return beside
    first = int(round(float(origin) * sample_rate))
    source = heard[max(0, first) : max(0, first + beside.shape[0])]
    start = max(0, -first)
    beside[start : start + source.shape[0]] = source[: beside.shape[0] - start]
    return beside
