import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

if not QApplication.instance():
    QApplication(sys.argv)

from korshmer.core.control_bar import ControlBar
from korshmer.utils import audio_host

HOUR_MS = 3600 * 1000


@pytest.fixture
def bar(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_host, "AUDIO_HOST_FILE", tmp_path / "audio.json")
    widget = ControlBar()
    widget.sys_timer.stop()
    yield widget
    widget.close()


def click_time(bar, times=1):
    for _ in range(times):
        QTest.mouseClick(bar.lbl_time, Qt.LeftButton)


def shown(bar, current_ms, total_ms):
    bar.update_time_label(current_ms, total_ms)
    return bar.lbl_time.text()


def count_tooltip_changes(bar):
    calls = []
    original = bar.lbl_time.setToolTip

    def record(text):
        calls.append(text)
        original(text)

    bar.lbl_time.setToolTip = record
    return calls


def test_safeguard_the_time_label_starts_as_minutes_and_seconds(bar):
    assert shown(bar, 83456, 185000) == "01:23 / 03:05"


def test_a_file_longer_than_an_hour_shows_hours(bar):
    assert shown(bar, 83456, HOUR_MS + 61000) == "0:01:23 / 1:01:01"


def test_one_click_on_the_time_label_adds_milliseconds(bar):
    click_time(bar)
    assert shown(bar, 83456, 185000) == "01:23.456 / 03:05.000"


def test_milliseconds_also_show_hours_for_a_long_file(bar):
    click_time(bar)
    assert shown(bar, HOUR_MS + 83456, 2 * HOUR_MS) == (
        "1:01:23.456 / 2:00:00.000"
    )


def test_two_clicks_on_the_time_label_show_timecode_with_hours(bar):
    click_time(bar, 2)
    assert shown(bar, 83456, 185000) == "0:01:23 / 0:03:05"


def test_three_clicks_on_the_time_label_show_bars_and_beats(bar):
    click_time(bar, 3)
    assert shown(bar, 2000, 4000) == "2.1.000 / 3.1.000"


def test_safeguard_four_clicks_on_the_time_label_return_to_minutes_and_seconds(
    bar,
):
    click_time(bar, 4)
    assert shown(bar, 83456, 185000) == "01:23 / 03:05"


def test_bars_and_beats_start_at_bar_one_beat_one(bar):
    click_time(bar, 3)
    assert shown(bar, 0, 0) == "1.1.000 / 1.1.000"


def test_bars_and_beats_show_the_part_of_a_beat_as_ticks(bar):
    click_time(bar, 3)
    assert shown(bar, 2750, 4000) == "2.2.240 / 3.1.000"


def test_the_bars_display_follows_a_new_tempo(bar):
    click_time(bar, 3)
    bar.update_time_label(2000, 4000)
    bar.set_tempo(60.0)
    assert bar.lbl_time.text() == "1.3.000 / 2.1.000"


def test_the_bars_display_follows_a_new_time_signature(bar):
    click_time(bar, 3)
    bar.update_time_label(1500, 3000)
    bar.set_beats_per_bar(3)
    assert bar.lbl_time.text() == "2.1.000 / 3.1.000"


def test_the_bars_display_uses_the_editor_beat_positions_when_given(bar):
    click_time(bar, 3)
    bar.set_beat_mapper(lambda seconds: (seconds - 0.5) * 3.0)
    assert shown(bar, 1500, 2500) == "1.4.000 / 2.3.000"


def test_time_before_the_first_editor_beat_shows_as_bar_one(bar):
    click_time(bar, 3)
    bar.set_beat_mapper(lambda seconds: (seconds - 0.5) * 3.0)
    assert shown(bar, 0, 2500) == "1.1.000 / 2.3.000"


def test_without_editor_beat_positions_the_steady_tempo_is_used(bar):
    click_time(bar, 3)
    bar.set_beat_mapper(lambda seconds: None)
    assert shown(bar, 2000, 4000) == "2.1.000 / 3.1.000"


def test_clicking_the_time_label_announces_the_new_format(bar):
    heard = []
    bar.time_display_changed.connect(heard.append)
    click_time(bar)
    assert heard == [bar.time_display_mode()] and heard != ["clock"]


def test_a_saved_time_format_can_be_put_back(bar):
    bar.set_time_display_mode("bars")
    assert shown(bar, 2000, 4000) == "2.1.000 / 3.1.000"


def test_an_unknown_time_format_keeps_the_current_one(bar):
    bar.set_time_display_mode("frames")
    assert shown(bar, 83456, 185000) == "01:23 / 03:05"


def test_the_time_label_says_how_to_change_the_format(bar):
    assert "Click" in bar.lbl_time.toolTip()


def test_safeguard_the_time_label_hint_is_not_rewritten_while_the_song_plays(
    bar,
):
    hints = count_tooltip_changes(bar)
    for position in range(0, 5000, 100):
        bar.update_time_label(position, 185000)
    assert hints == []


def test_the_time_label_hint_names_the_new_format_after_a_click(bar):
    before = bar.lbl_time.toolTip()
    click_time(bar)
    assert bar.lbl_time.toolTip() != before


def test_the_time_label_reserves_room_for_its_widest_text(bar):
    click_time(bar)
    bar.update_time_label(0, 185000)
    widest = bar.lbl_time.fontMetrics().horizontalAdvance(
        "00:00.000 / 00:00.000"
    )
    assert bar.lbl_time.minimumWidth() >= widest


def test_the_time_label_keeps_its_width_while_the_song_plays(bar):
    click_time(bar)
    bar.update_time_label(0, 185000)
    reserved = bar.lbl_time.minimumWidth()
    bar.update_time_label(111111, 185000)
    assert reserved > 0 and bar.lbl_time.minimumWidth() == reserved
