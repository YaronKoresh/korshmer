import sys

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtWidgets import QApplication

from korshmer.audio_processing import clip_elastic, clip_polish
from korshmer.midi.tempo_map import TempoMap
from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline

RATE = 44100
BEAT_SECONDS = 0.5
OFF_GRID_STARTS = (0.5, 1.07, 1.46, 2.12, 2.5, 3.09, 3.44, 4.0, 4.6)


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def fixed_files(tmp_path, monkeypatch):
    monkeypatch.setattr(clip_polish, "DATA_DIR", tmp_path / "data")
    return tmp_path / "data" / clip_polish.POLISHED_FOLDER


def _syllables(starts=OFF_GRID_STARTS, seconds=6.0):
    audio = np.zeros(int(RATE * seconds), dtype=np.float32)
    for index, start in enumerate(starts):
        time = np.arange(int(0.2 * RATE)) / RATE
        shape = np.minimum(1.0, time / 0.008) * np.minimum(
            1.0, (0.2 - time) / 0.08
        )
        voice = 0.3 * np.sin(2 * np.pi * (180 + 40 * (index % 3)) * time)
        first = int(start * RATE)
        audio[first : first + time.shape[0]] += voice * shape
    hum = 0.02 * np.sin(2 * np.pi * 150 * np.arange(audio.shape[0]) / RATE)
    return (audio + hum).astype(np.float32)


def _recording(path, audio):
    sf.write(str(path), audio, RATE)
    return str(path)


def _off_grid_ms(starts, step):
    return [
        abs(start / step - round(start / step)) * step * 1000.0
        for start in starts
    ]


def _aligned(qapp, recording, start=0.0, grid="1/8", tempo_map=None):
    timeline = ArrangerTimeline()
    timeline.resize(1100, 600)
    timeline.show()
    qapp.processEvents()
    timeline.btn_snap.setChecked(False)
    timeline.btn_even_bpm.setChecked(False)
    timeline.model.tempo = 120.0
    timeline.set_grid(grid)
    clip = timeline.add_audio_clip(recording, start, name="Words")
    timeline.wait_for_waveforms()
    if tempo_map is not None:
        timeline.model.tempo_map = tempo_map
    said = []
    timeline.clip_edit_status.connect(said.append)
    timeline.align_clip_to_beat(clip.id)
    timeline.wait_for_clip_edits()
    QApplication.processEvents()
    QApplication.processEvents()
    return timeline, timeline.model.clip_by_id(clip.id), said[-1]


def _starts_in_the_song(clip):
    audio, rate = sf.read(clip.path, dtype="float32")
    found = clip_elastic.sound_starts(audio, rate)
    return [clip.start - clip.offset + second for second, _weight in found]


def test_sounds_joined_without_silence_each_land_on_the_grid(
    qapp, tmp_path, fixed_files
):
    recording = _recording(tmp_path / "words.wav", _syllables())
    before = _off_grid_ms(OFF_GRID_STARTS, BEAT_SECONDS / 2)

    _timeline, clip, said = _aligned(qapp, recording)

    after = _off_grid_ms(_starts_in_the_song(clip), BEAT_SECONDS / 2)
    assert max(before) > 100.0
    assert np.mean(after) < 15.0
    assert max(after) < 30.0
    assert "stretched onto the 1/8 grid" in said


def test_stretching_inside_a_clip_keeps_its_length_and_the_recording(
    qapp, tmp_path, fixed_files
):
    recording = _recording(tmp_path / "words.wav", _syllables())
    audio = sf.read(recording, dtype="float32")[0]

    _timeline, clip, _said = _aligned(qapp, recording)

    assert clip.path != recording
    assert clip.duration == pytest.approx(6.0, abs=0.01)
    assert sf.info(clip.path).frames == audio.shape[0]
    assert np.array_equal(sf.read(recording, dtype="float32")[0], audio)


def test_one_step_back_returns_the_sounds_to_where_they_were(
    qapp, tmp_path, fixed_files
):
    recording = _recording(tmp_path / "words.wav", _syllables())
    timeline, clip, _said = _aligned(qapp, recording, start=2.0)

    timeline.undo()

    restored = timeline.model.clip_by_id(clip.id)
    assert (restored.path, restored.start) == (recording, 2.0)


def test_a_finer_grid_moves_the_sounds_less(qapp, tmp_path, fixed_files):
    recording = _recording(tmp_path / "words.wav", _syllables())

    _first, coarse, said_coarse = _aligned(qapp, recording, grid="1/8")
    _second, fine, said_fine = _aligned(qapp, recording, grid="1/16")

    def largest(said):
        return float(said.split("largest move ")[1].split(" ms")[0])

    assert largest(said_fine) < largest(said_coarse)
    after = _off_grid_ms(_starts_in_the_song(fine), BEAT_SECONDS / 4)
    assert max(after) < 30.0
    assert "1/16 grid" in said_fine


def test_a_clip_whose_sounds_already_sit_on_the_grid_is_not_stretched(
    qapp, tmp_path, fixed_files
):
    on_grid = tuple(0.5 + 0.25 * step for step in range(16))
    recording = _recording(tmp_path / "words.wav", _syllables(on_grid))

    _timeline, clip, said = _aligned(qapp, recording)

    assert clip.path == recording
    assert "stretched" not in said


def test_sounds_are_not_stretched_while_the_song_itself_changes_tempo(
    qapp, tmp_path, fixed_files
):
    recording = _recording(tmp_path / "words.wav", _syllables())

    _timeline, clip, said = _aligned(
        qapp, recording, tempo_map=TempoMap((0.0, 8.0), (120.0, 126.0))
    )

    assert clip.path == recording
    assert "stretched" not in said


def test_the_automatic_fix_also_stretches_the_sounds_of_each_part(
    qapp, tmp_path, fixed_files
):
    recording = _recording(tmp_path / "words.wav", _syllables())
    timeline = ArrangerTimeline()
    timeline.show()
    qapp.processEvents()
    timeline.btn_snap.setChecked(False)
    timeline.btn_even_bpm.setChecked(False)
    timeline.model.tempo = 120.0
    timeline.set_grid("1/8")
    timeline.notes_audio = lambda _song, _tempo, _rate: None
    clip = timeline.add_audio_clip(recording, 0.0)
    said = []
    timeline.clip_edit_status.connect(said.append)

    timeline.fix_clip_automatically(clip.id)
    timeline.wait_for_clip_edits()
    QApplication.processEvents()
    QApplication.processEvents()

    assert "stretched onto the 1/8 grid" in said[-1]


def test_a_sound_is_moved_to_the_place_its_anchor_asks_for():
    audio = np.zeros(RATE * 4, dtype=np.float32)
    click = np.hanning(2000) * np.sin(np.arange(2000) * 0.2) * 0.4
    audio[int(1.9 * RATE) : int(1.9 * RATE) + 2000] = click

    moved = clip_elastic.stretched_onto_beat(
        audio[:, np.newaxis], RATE, [(0.0, 0.0), (1.9, 2.0), (4.0, 4.0)]
    )[:, 0]

    assert moved.shape == audio.shape
    assert np.argmax(np.abs(moved)) / RATE == pytest.approx(
        np.argmax(np.abs(audio)) / RATE + 0.1, abs=0.015
    )


def test_a_sound_that_would_need_too_much_stretching_stays_where_it_is():
    moves = [(1.0, 1.0, 5.0), (1.2, 1.9, 1.0), (3.0, 3.0, 4.0)]

    anchors = clip_elastic.beat_map(moves, 0.0, 5.0)

    assert (1.2, 1.9) not in anchors
    assert (1.0, 1.0) in anchors and (3.0, 3.0) in anchors


def test_sounds_never_change_their_order():
    rng = np.random.default_rng(8)
    sources = np.sort(rng.uniform(0.3, 9.7, 30))
    moves = [
        (
            float(source),
            float(source + rng.uniform(-0.12, 0.12)),
            float(rng.uniform(1, 5)),
        )
        for source in sources
    ]

    anchors = clip_elastic.beat_map(moves, 0.0, 10.0)

    targets = [target for _source, target in anchors]
    assert targets == sorted(targets)
    for (from_a, to_a), (from_b, to_b) in zip(
        anchors, anchors[1:], strict=False
    ):
        ratio = (to_b - to_a) / (from_b - from_a)
        assert (
            clip_elastic.SLOWEST_STRETCH
            <= ratio
            <= clip_elastic.FASTEST_STRETCH
        )


def test_the_strongest_sounds_are_placed_first_when_two_compete():
    moves = [(1.00, 1.25, 1.0), (1.10, 1.25, 6.0)]

    anchors = clip_elastic.beat_map(moves, 0.0, 4.0)

    assert (1.10, 1.25) in anchors
    assert (1.00, 1.25) not in anchors
