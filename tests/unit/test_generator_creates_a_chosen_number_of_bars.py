import sys

import pytest
from PySide6.QtWidgets import QApplication

from korshmer.midi_generation.composer import ComposeOptions, compose
from korshmer.ui.widgets.material.length_box import BarsPicker

BEATS_PER_BAR = 4


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


def _song(bars, bpm=120.0, seed=3):
    return compose(
        ComposeOptions(
            root="A", scale="Minor", bpm=bpm, bars=bars, style="Pop", seed=seed
        )
    )


def _last_note_end(song):
    return max(note.end for part in song.instruments for note in part.notes)


@pytest.mark.parametrize("bars", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("bpm", [80.0, 120.0, 174.0])
def test_a_new_song_has_exactly_the_number_of_bars_asked_for(bars, bpm):
    song = _song(bars, bpm)
    bar_seconds = BEATS_PER_BAR * 60.0 / bpm

    assert song.total_bars == bars
    assert _last_note_end(song) <= bars * bar_seconds + 1e-6
    assert _last_note_end(song) > (bars - 2) * bar_seconds


def test_the_same_number_of_bars_lasts_longer_at_a_slower_tempo():
    slow = _last_note_end(_song(16, bpm=80.0))
    fast = _last_note_end(_song(16, bpm=160.0))

    assert slow == pytest.approx(2.0 * fast, rel=0.1)


def test_a_bar_count_between_groups_of_four_is_rounded_up_to_the_next_group():
    assert _song(13).total_bars == 16
    assert _song(17).total_bars == 20


@pytest.mark.parametrize("bars", [0, 3, 5000, 12.5, "many", True])
def test_an_impossible_number_of_bars_is_refused_with_a_reason(bars):
    with pytest.raises(ValueError, match="bars"):
        _song(bars)


def test_a_length_in_minutes_still_works_when_no_bars_are_given():
    song = compose(
        ComposeOptions(
            root="A", scale="Minor", bpm=120.0, minutes=1.0, style="Pop", seed=3
        )
    )

    assert song.total_bars == 32


def test_the_bar_picker_starts_at_its_default_and_steps_by_whole_groups(qapp):
    picker = BarsPicker(4, 2048, 64, 4)

    assert picker.box.text() == "64 bars"
    picker.box.stepUp()
    assert picker.box.value() == 68
    picker.box.stepDown()
    picker.box.stepDown()
    assert picker.box.value() == 60


def test_quick_bar_buttons_set_the_count_and_show_which_one_is_chosen(qapp):
    picker = BarsPicker(4, 2048, 64, 4)

    assert [button.text() for button in picker.quick.values()] == [
        "8",
        "16",
        "32",
        "48",
        "64",
        "96",
        "128",
    ]
    picker.quick[32].click()
    assert picker.box.value() == 32
    assert [
        bars for bars, button in picker.quick.items() if button.isChecked()
    ] == [32]
    picker.box.setValue(36)
    assert not any(button.isChecked() for button in picker.quick.values())


def test_the_bar_picker_shows_how_long_the_bars_last_at_the_song_tempo(qapp):
    picker = BarsPicker(4, 2048, 32, 4, tempo=120.0)

    assert picker.seconds() == pytest.approx(64.0)
    assert picker.time.text() == "About 1:04 at 120 BPM"
    picker.set_tempo(96.0)
    assert picker.time.text() == "About 1:20 at 96 BPM"


def test_a_bar_count_outside_the_allowed_range_is_brought_back_inside(qapp):
    picker = BarsPicker(4, 2048, 64, 4)

    picker.box.setValue(1)
    assert picker.box.value() == 4
    picker.box.setValue(99999)
    assert picker.box.value() == 2048


def test_turning_the_bar_picker_off_also_turns_off_its_box(qapp):
    picker = BarsPicker(4, 2048, 64, 4)

    picker.setEnabled(False)

    assert not picker.box.isEnabled()
