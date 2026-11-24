import numpy as np
import pytest
import soundfile as sf

from korshmer.audio_processing import clip_polish
from korshmer.audio_processing.clip_polish import (
    PART_MIN_FADE_S,
    PART_ROOM_S,
    ClipAudio,
)

RATE = 22050


def test_a_part_keeps_room_on_each_side_up_to_its_neighbour():
    cuts = clip_polish._part_cuts([(1.0, 2.0), (2.6, 4.0), (8.0, 9.0)], 0, 10)
    assert clip_polish._part_cuts([(1.0, 2.0), (2.3, 3.0)], 0, 10) == [
        (1.0 - PART_ROOM_S, 2.3),
        (2.0, 3.0 + PART_ROOM_S),
    ]
    assert cuts == [
        (1.0 - PART_ROOM_S, 2.0 + PART_ROOM_S),
        (2.6 - PART_ROOM_S, 4.0 + PART_ROOM_S),
        (8.0 - PART_ROOM_S, 9.0 + PART_ROOM_S),
    ]


def test_room_never_reaches_past_the_kept_clip():
    assert clip_polish._part_cuts([(0.1, 1.0), (3.0, 4.9)], 0.0, 5.0) == [
        (0.0, 1.0 + PART_ROOM_S),
        (3.0 - PART_ROOM_S, 5.0),
    ]


@pytest.mark.parametrize("channels", [1, 2])
def test_a_part_starts_and_ends_at_silence_without_a_jump(channels):
    shape = (RATE,) if channels == 1 else (RATE, 2)
    faded = clip_polish._faded(np.ones(shape, np.float32), RATE, 0.0, 0.0)
    assert faded.shape == shape
    assert np.abs(faded[0]).max() < 0.01
    assert np.abs(faded[-1]).max() < 0.01
    middle = int(PART_MIN_FADE_S * RATE) + 1
    assert np.allclose(faded[middle:-middle], 1.0)


def test_neighbouring_parts_crossfade_the_room_sound_at_steady_power():
    head = clip_polish._faded(np.ones(RATE, np.float32), RATE, 0.4, 0.0)
    tail = clip_polish._faded(np.ones(RATE, np.float32), RATE, 0.0, 0.4)
    fade = int(0.4 * RATE)
    power = head[:fade] ** 2 + tail[-fade:] ** 2
    assert np.allclose(power, 1.0, atol=1e-4)


def test_parts_written_by_fix_clip_parts_overlap_through_a_short_pause(
    tmp_path, monkeypatch
):
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.003, RATE * 6).astype(np.float32)
    tone = 0.5 * np.sin(np.arange(RATE) * 0.2).astype(np.float32)
    audio[RATE : 2 * RATE] += tone
    audio[int(2.4 * RATE) : int(2.4 * RATE) + RATE] += tone
    path = tmp_path / "take.wav"
    sf.write(str(path), audio, RATE, subtype="FLOAT")
    spans = [(0.95, 2.05), (2.35, 3.45)]
    monkeypatch.setattr(clip_polish, "clip_parts", lambda *a, **k: spans)
    pieces = []

    def fake_fix(piece, *args):
        pieces.append(piece)
        return {"path": piece.path, "made": []}

    monkeypatch.setattr(clip_polish, "fix_clip", fake_fix)

    result = clip_polish.fix_clip_parts(
        ClipAudio(str(path)), (), [], folder=str(tmp_path)
    )

    first, second = (
        sf.read(piece.path, dtype="float32")[0] for piece in pieces
    )
    assert pieces[0].start == pytest.approx(0.95 - PART_ROOM_S, abs=1e-3)
    assert pieces[1].start == pytest.approx(2.05, abs=1e-3)
    assert pieces[0].start + first.shape[0] / RATE == pytest.approx(
        2.35, abs=1e-3
    )
    for part in (first, second):
        assert abs(part[0]) < 1e-3 and abs(part[-1]) < 1e-3
    assert [part["seconds"] for part in result["parts"]] == pytest.approx(
        [1.1, 1.1]
    )
