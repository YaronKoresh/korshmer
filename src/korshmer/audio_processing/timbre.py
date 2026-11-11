from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
EXCERPT_S = 60.0
MIN_AUDIO_S = 1.0

DISTORTED = 0.55
PLUCKED = 0.35
BRIGHT = 0.5


DISTORTED_GUITAR = (29, 30)
CLEAN_ELECTRIC_GUITAR = (27, 28, 26)
ACOUSTIC_GUITAR = (25, 24)
PLUCKED_KEYS = (4, 0, 5, 27)
BRIGHT_SUSTAINED = (81, 62, 61, 80, 18)
SOFT_SUSTAINED = (89, 48, 49, 90, 16)


@dataclass(frozen=True)
class TimbreProfile:
    distortion: float
    percussiveness: float
    brightness: float

    @property
    def distorted(self):
        return self.distortion >= DISTORTED

    @property
    def plucked(self):
        return self.percussiveness >= PLUCKED

    @property
    def bright(self):
        return self.brightness >= BRIGHT


def _excerpt(y):
    size = int(EXCERPT_S * SAMPLE_RATE)
    if len(y) <= size:
        return y

    step = size // 4
    starts = range(0, len(y) - size + 1, step)
    best = max(starts, key=lambda s: float(np.mean(y[s : s + size] ** 2)))
    return y[best : best + size]


def _ramp(value, low, high):
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def analyze_timbre(audio, sample_rate):
    import librosa

    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=int(np.argmin(y.shape)))
    if sample_rate != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    if len(y) < MIN_AUDIO_S * SAMPLE_RATE or float(np.abs(y).max()) < 1e-4:
        return None
    y = _excerpt(y)
    spectrum = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    harmonic, percussive = librosa.decompose.hpss(spectrum)
    power = spectrum**2
    frame_power = power.sum(0)
    active = frame_power > 1e-3 * frame_power.max()
    if not active.any():
        return None
    percussive_share = float(
        (percussive[:, active] ** 2).sum() / ((power[:, active]).sum() + 1e-12)
    )
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)

    band = harmonic[(freqs >= 200) & (freqs < 5000)][:, active] + 1e-9
    flatness = float(
        np.median(np.exp(np.mean(np.log(band), 0)) / np.mean(band, 0))
    )
    centroid = float(
        np.median(
            librosa.feature.spectral_centroid(
                S=spectrum[:, active], sr=SAMPLE_RATE
            )
        )
    )
    decay = _note_decay(spectrum)
    return TimbreProfile(
        distortion=_ramp(flatness, 0.12, 0.30),
        percussiveness=max(
            _ramp(percussive_share, 0.05, 0.20),
            _ramp(1.0 - decay, 0.15, 0.50),
        ),
        brightness=_ramp(centroid, 800.0, 3000.0),
    )


def _note_decay(spectrum):
    import librosa

    rms = librosa.feature.rms(S=spectrum)[0]
    envelope = librosa.onset.onset_strength(
        S=librosa.amplitude_to_db(spectrum), sr=SAMPLE_RATE
    )
    onsets = librosa.onset.onset_detect(
        onset_envelope=envelope, sr=SAMPLE_RATE, units="frames"
    )
    floor = 1e-3 * float(rms.max())
    ratios = []
    for frame in onsets:
        peak = float(rms[frame : frame + 3].max(initial=0.0))
        later = rms[frame + 6 : frame + 10]
        if len(later) and peak > floor:
            ratios.append(float(later.mean()) / peak)
    return float(np.median(ratios)) if ratios else 1.0


def analyze_timbre_file(path):
    import librosa

    try:
        audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    except Exception:
        return None
    return analyze_timbre(audio, SAMPLE_RATE)


def preferred_programs(stem_name, profile):
    if profile is None:
        return ()
    stem = stem_name.lower()
    if "drum" in stem or "vocal" in stem or "bass" in stem:
        return ()
    if "guitar" in stem:
        if profile.distorted:
            return DISTORTED_GUITAR

        if profile.plucked and profile.bright:
            return ACOUSTIC_GUITAR
        return CLEAN_ELECTRIC_GUITAR
    if "piano" in stem or "keys" in stem:
        return ()
    if profile.distorted:
        return DISTORTED_GUITAR
    if profile.plucked:
        return PLUCKED_KEYS
    if profile.bright:
        return BRIGHT_SUSTAINED
    return SOFT_SUSTAINED
