import numpy as np

from korshmer.audio_processing.stem_classification import VocalProcessingAI


class TestRemoveBreaths:
    def test_voiced_regions_preserved(self):
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = np.zeros_like(t)
        voiced_start = int(0.2 * sr)
        voiced_end = int(0.8 * sr)
        audio[voiced_start:voiced_end] = 0.5 * np.sin(
            2 * np.pi * 440 * t[voiced_start:voiced_end]
        )

        vp = VocalProcessingAI()
        cleaned = vp.remove_breaths(audio, sr, threshold_db=-30.0)

        voiced_energy = np.sum(cleaned[voiced_start:voiced_end] ** 2)
        assert voiced_energy > 0, (
            "Voiced region was silenced instead of preserved"
        )

    def test_quiet_regions_silenced(self):
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = np.full_like(t, 0.0001)
        loud_start = int(0.3 * sr)
        loud_end = int(0.7 * sr)
        audio[loud_start:loud_end] = 0.5 * np.sin(
            2 * np.pi * 440 * t[loud_start:loud_end]
        )

        vp = VocalProcessingAI()
        cleaned = vp.remove_breaths(audio, sr, threshold_db=-30.0)

        quiet_before = np.sum(np.abs(cleaned[: int(0.1 * sr)]))
        loud_section = np.sum(np.abs(cleaned[loud_start:loud_end]))
        assert loud_section > quiet_before, (
            "Loud section should have more energy than quiet section after breath removal"
        )
