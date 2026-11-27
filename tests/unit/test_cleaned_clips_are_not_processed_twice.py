import os
import sys

import numpy as np
import pytest
import soundfile as sf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

if not QApplication.instance():
    app = QApplication(sys.argv)

from korshmer.audio_processing import clip_level, clip_polish, clip_stems
from korshmer.midi.tempo_map import TempoMap
from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline

RATE = 44100


def _sung(seconds, level=0.1, quiet_phrase=1.0):
    time = np.arange(int(seconds * RATE)) / RATE
    voice = sum(
        np.sin(2 * np.pi * 220.0 * harmonic * time) / harmonic
        for harmonic in (1, 2, 3, 4)
    )
    phrases = (np.sin(2 * np.pi * time / 4.0) > -0.3).astype(np.float64)
    softer = np.where((time % 8.0) < 4.0, 1.0, quiet_phrase)
    return (level * voice * phrases * softer).astype(np.float32)


def _with_room(audio, level=0.004):
    room = np.random.default_rng(3).normal(0.0, level, audio.shape)
    return (audio + room).astype(np.float32)


def _pause_db(audio):
    levels = clip_level.frame_levels_db(audio)
    return float(np.percentile(levels, 5))


def test_a_cleaned_vocal_keeps_its_room_sound_when_its_quality_is_improved():
    take = _with_room(_sung(12.0))

    improved, _readings = clip_level.improve(take, RATE, cleaned=True)

    level_change = clip_level.active_level_db(improved) - (
        clip_level.active_level_db(take)
    )
    room_change = _pause_db(improved) - _pause_db(take)
    assert room_change == pytest.approx(level_change, abs=1.5)


def test_an_uncleaned_vocal_still_has_its_room_noise_turned_down():
    take = _with_room(_sung(12.0))

    improved, _readings = clip_level.improve(take, RATE)

    level_change = clip_level.active_level_db(improved) - (
        clip_level.active_level_db(take)
    )
    assert _pause_db(improved) - _pause_db(take) < level_change - 3.0


def test_a_cleaned_vocal_is_not_given_a_second_noise_cut_or_tone_balance():
    _improved, readings = clip_level.improve(
        _with_room(_sung(12.0)), RATE, cleaned=True
    )

    names = {reading.name for reading in readings}
    assert clip_level.NOISE not in names
    assert clip_level.TONE not in names


def test_a_cleaned_vocal_is_only_turned_down_by_the_level_ride():
    take = _sung(16.0, quiet_phrase=0.2)
    wanted = clip_level.goals(clip_level.measure(take, RATE), cleaned=True)

    ridden = clip_level._ridden(take, RATE, wanted, may_lift=False)

    assert np.all(np.abs(ridden) <= np.abs(take) + 1e-6)


def test_the_phrase_and_word_rides_are_one_gain_within_one_limit():
    take = _sung(16.0, quiet_phrase=0.05)
    wanted = clip_level.goals(clip_level.measure(take, RATE))

    ridden = clip_level._ridden(take, RATE, wanted)

    loud = np.abs(take) > 1e-3
    change_db = 20.0 * np.log10(np.abs(ridden[loud]) / np.abs(take[loud]))
    passes = clip_level.MAX_PASSES
    assert np.max(np.abs(change_db)) <= clip_level.MAX_RIDE_DB * passes + 0.01


def test_improving_measures_the_whole_take_only_before_and_after(monkeypatch):
    calls = []
    measure = clip_level.measure

    def counted(*args, **kwargs):
        calls.append(args)
        return measure(*args, **kwargs)

    monkeypatch.setattr(clip_level, "measure", counted)

    clip_level.improve(_with_room(_sung(12.0, quiet_phrase=0.1)), RATE)

    assert len(calls) == 2


@pytest.mark.parametrize(
    ("name", "cleaned"),
    [
        ("take_cleanvocals_0123abcd.wav", True),
        ("take_cleanmusic_0123abcd.wav", True),
        ("take_clean_0123abcd.wav", True),
        ("take_improved_0123abcd.wav", False),
        ("clean take.wav", False),
    ],
)
def test_a_clip_file_made_by_the_noise_removal_is_recognised(name, cleaned):
    assert clip_polish.was_cleaned(os.path.join("folder", name)) is cleaned


def test_improving_a_file_from_the_noise_removal_skips_the_noise_and_tone_steps(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(clip_polish, "DATA_DIR", tmp_path / "data")
    path = tmp_path / "take_cleanvocals_0123abcd.wav"
    sf.write(path, _with_room(_sung(12.0)), RATE)

    _fixed, readings = clip_polish.improve_quality_file(
        str(path), str(tmp_path)
    )

    assert clip_level.NOISE not in {reading.name for reading in readings}


def _tone(seconds, freq, level):
    time = np.arange(int(seconds * RATE)) / RATE
    return level * np.sin(2 * np.pi * freq * time)


def _stems(voice, music):
    return np.stack([voice, voice]), np.stack([music, music])


def test_weak_music_that_only_sounds_while_singing_is_not_kept(
    tmp_path, monkeypatch
):
    voice = _sung(12.0, level=0.2)
    phrase = np.sin(2 * np.pi * np.arange(12 * RATE) / RATE / 4.0)
    residual = _tone(12.0, 3100.0, 0.02) * (phrase > 0.2)
    monkeypatch.setattr(
        clip_stems, "vocals_of", lambda mix, **kw: np.stack([voice, voice])
    )
    path = tmp_path / "take.wav"
    sf.write(path, np.stack([voice + residual] * 2, axis=1), RATE)

    parts = clip_stems.separate_clip_file(str(path), folder=tmp_path)

    assert [kind for kind, _path in parts] == [clip_stems.VOCALS]


def test_weak_music_that_also_plays_between_phrases_is_kept(
    tmp_path, monkeypatch
):
    voice = _sung(12.0, level=0.2)
    bed = _tone(12.0, 3100.0, 0.02) * (
        1.0 + 0.5 * np.sin(2 * np.pi * 1.3 * np.arange(12 * RATE) / RATE)
    )
    monkeypatch.setattr(
        clip_stems, "vocals_of", lambda mix, **kw: np.stack([voice, voice])
    )
    path = tmp_path / "take.wav"
    sf.write(path, np.stack([voice + bed] * 2, axis=1), RATE)

    parts = clip_stems.separate_clip_file(str(path), folder=tmp_path)

    assert [kind for kind, _path in parts] == [
        clip_stems.VOCALS,
        clip_stems.MUSIC,
    ]


def test_split_parts_move_onto_the_beat_by_the_same_beats_under_a_tempo_map(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(clip_polish, "DATA_DIR", tmp_path / "data")
    editor = ArrangerTimeline()
    try:
        editor.model.tempo_map = TempoMap((0.0, 16.0), (90.0, 150.0))
        files = []
        for name in ("take", "first", "second"):
            path = tmp_path / f"{name}.wav"
            sf.write(path, np.stack([_tone(2.0, 220.0, 0.1)] * 2, axis=1), RATE)
            files.append(str(path))
        clip = editor.add_audio_clip(files[0], 1.0, name="Take")
        result = {
            "shift": 0.25,
            "split": [
                (
                    clip_stems.VOCALS,
                    files[0],
                    {
                        "path": files[0],
                        "parts": [
                            {"path": files[1], "start": 1.0},
                            {"path": files[2], "start": 20.0},
                        ],
                    },
                )
            ],
        }

        editor._apply_split(clip, result)

        tempo_map = editor.model.tempo_map
        starts = sorted(part.start for part in editor.model.clips)
        expected = [
            tempo_map.seconds_at(tempo_map.beats_at(start) + 0.25)
            for start in (1.0, 20.0)
        ]
        assert starts == pytest.approx(expected, abs=1e-6)
        assert starts[1] - 20.0 != pytest.approx(starts[0] - 1.0, abs=1e-3)
    finally:
        editor.wait_for_waveforms(20000)
        editor.close()
