import filecmp

import numpy as np
import soundfile as sf

from tests.fixtures.transcription_fixtures import (
    FIXTURE_DIR,
    SONGS,
    load_truth,
    note_scores,
    write_fixtures,
)

NAMES = [s[0] for s in SONGS]


def test_seven_songs_exist_and_are_small():
    assert len(NAMES) == 7
    total = 0
    for name in NAMES:
        path = FIXTURE_DIR / f"{name}.wav"
        info = sf.info(path)
        assert info.duration <= 10.0
        assert info.samplerate in (22050, 44100)
        assert info.channels in (1, 2)
        total += path.stat().st_size
    assert total < 6 * 1024 * 1024


def test_generation_is_deterministic(tmp_path):
    write_fixtures(tmp_path)
    for name in NAMES:
        for ext in ("wav", "csv"):
            assert filecmp.cmp(
                tmp_path / f"{name}.{ext}",
                FIXTURE_DIR / f"{name}.{ext}",
                shallow=False,
            )


def test_songs_are_mastered():
    for name in NAMES:
        audio, _ = sf.read(FIXTURE_DIR / f"{name}.wav")
        peak = np.max(np.abs(audio))
        assert 0.85 < peak <= 0.9
        assert abs(np.mean(audio)) < 0.01


def test_truth_notes_are_audible_at_their_pitch():
    for name in NAMES:
        audio, sr = sf.read(FIXTURE_DIR / f"{name}.wav")
        mono = audio if audio.ndim == 1 else audio.mean(axis=1)
        truth = load_truth(name)
        assert len(truth) >= 12
        hits = 0
        for pitch, start, end in truth:
            seg = mono[int(start * sr) : int(min(end, start + 0.2) * sr)]
            if len(seg) < 512:
                continue
            spectrum = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
            freqs = np.fft.rfftfreq(len(seg), 1 / sr)
            target = 440.0 * 2 ** ((pitch - 69) / 12)
            band = (freqs > target * 0.97) & (freqs < target * 1.03)
            if spectrum[band].max() > 0.2 * spectrum.max():
                hits += 1
        assert hits / len(truth) > 0.8


def test_note_scores_perfect_and_wrong_octave():
    truth = load_truth(NAMES[0])
    assert note_scores(truth, truth) == (1.0, 1.0, 1.0)
    shifted = [(p + 12, s, e) for p, s, e in truth]
    assert note_scores(truth, shifted)[2] < 0.2
