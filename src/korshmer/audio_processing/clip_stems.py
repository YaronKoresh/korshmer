import numpy as np
import soundfile as sf

from korshmer.audio_processing.vocal_cleanup import active_level_db

VOCALS = "vocals"
MUSIC = "music"
KINDS = (VOCALS, MUSIC)
ABSENT_UNDER_DB = 30.0
SILENT_DB = -70.0
STEM_SUBTYPE = "FLOAT"
CLIP_QUALITY = "balanced"


class ClipSeparationUnavailable(RuntimeError):
    pass


def vocals_of(mix, progress=None, should_stop=None):
    from korshmer.audio_processing.separation_engine import (
        SeparationOptions,
        vocals_of as split,
    )

    return split(
        mix,
        SeparationOptions(quality=CLIP_QUALITY),
        progress=progress,
        should_stop=should_stop,
    )


def _level_db(part):
    level = active_level_db(part.T)
    return -np.inf if level is None else float(level)


def _fitted(vocals, frames):
    vocals = np.asarray(vocals, dtype=np.float32)[:, :frames]
    if vocals.shape[1] < frames:
        vocals = np.pad(vocals, ((0, 0), (0, frames - vocals.shape[1])))
    return vocals


def separate_clip_file(path, folder=None, progress=None, should_stop=None):
    from korshmer.audio_processing.clip_polish import fixed_clip_path
    from korshmer.audio_processing.separation_engine import (
        SAMPLE_RATE,
        read_stereo,
    )

    mix = read_stereo(path).astype(np.float32)
    vocals = vocals_of(mix, progress=progress, should_stop=should_stop)
    if vocals is None:
        raise ClipSeparationUnavailable(
            "No vocal separation model could be loaded, so the clip was "
            "not split into vocals and music"
        )
    vocals = _fitted(vocals, mix.shape[1])
    music = mix - vocals
    levels = {VOCALS: _level_db(vocals), MUSIC: _level_db(music)}
    loudest = max(levels.values())
    if not np.isfinite(loudest) or loudest < SILENT_DB:
        return []
    parts = []
    for kind, data in ((VOCALS, vocals), (MUSIC, music)):
        if levels[kind] < loudest - ABSENT_UNDER_DB:
            continue
        target = fixed_clip_path(path, kind, folder)
        sf.write(target, data.T, SAMPLE_RATE, subtype=STEM_SUBTYPE)
        parts.append((kind, target))
    return parts
