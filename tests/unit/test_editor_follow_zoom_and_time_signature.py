import numpy as np
import pretty_midi
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from korshmer.midi.tempo_map import TempoMap
from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline
from korshmer.ui.widgets.midi.piano_roll import InteractivePianoRoll

TIME_SIGNATURES = ["2/4", "3/4", "4/4", "5/4", "6/8", "7/8", "12/8"]


def song(seconds=60.0, pitch=64):
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    piano = pretty_midi.Instrument(program=0, name="Piano")
    start = 0.0
    while start < seconds - 0.5:
        piano.notes.append(
            pretty_midi.Note(
                velocity=90, pitch=pitch, start=start, end=start + 0.5
            )
        )
        start += 1.0
    midi.instruments.append(piano)
    return midi


def settle():
    for _ in range(5):
        QApplication.processEvents()


def shown_roll(seconds=60.0):
    roll = InteractivePianoRoll()
    roll.resize(900, 500)
    if seconds:
        roll.load_notes(song(seconds))
    roll.show()
    settle()
    return roll


def shown_timeline(seconds=0.0):
    timeline = ArrangerTimeline()
    timeline.resize(1400, 600)
    if seconds:
        timeline.load_midi_object(song(seconds))
    timeline.show()
    settle()
    return timeline


def playing(roll, *positions_ms):
    roll.is_monitoring = True
    for position in positions_ms:
        roll.update_playhead(position)
    settle()


def visible_span(roll):
    area = roll.mapToScene(roll.viewport().rect()).boundingRect()
    return area.left(), area.right()


def lightness(roll):
    image = roll.viewport().grab().toImage()
    values = np.zeros((image.height(), image.width()))
    for y in range(image.height()):
        for x in range(image.width()):
            values[y, x] = image.pixelColor(x, y).lightness()
    return values


def beats_drawn_as_bar_lines(roll, beats=12, seconds_at=None):
    row = roll.mapFromScene(0.0, 54 * roll.grid_size + 10.0).y()
    light = lightness(roll)[row]
    darkness = {}
    for beat in range(1, beats + 1):
        seconds = seconds_at(beat) if seconds_at else beat * 60.0 / roll.tempo
        x = roll.mapFromScene(seconds * roll.pixels_per_second, 0.0).x()
        darkness[beat] = light.max() - light[x - 2 : x + 2].min()
    middle = (max(darkness.values()) + min(darkness.values())) / 2
    return [beat for beat, dark in darkness.items() if dark > middle]


def pick(combo, text):
    combo.setFocus()
    while combo.currentText() != text:
        index = combo.currentIndex()
        key = Qt.Key_Down if combo.findText(text) > index else Qt.Key_Up
        QTest.keyClick(combo, key)
        assert combo.currentIndex() != index


def test_a_new_editor_still_draws_a_bar_line_every_four_beats(qapp):
    roll = shown_roll(seconds=0)

    assert beats_drawn_as_bar_lines(roll) == [4, 8, 12]
    roll.close()


def test_three_beats_per_bar_draws_a_bar_line_every_three_beats(qapp):
    roll = shown_roll(seconds=0)

    roll.set_beats_per_bar(3)

    assert roll.beats_per_bar == 3
    assert beats_drawn_as_bar_lines(roll) == [3, 6, 9, 12]
    roll.close()


def test_bars_keep_their_beat_count_when_the_tempo_changes_mid_song(qapp):
    roll = shown_roll(seconds=0)
    tempo_map = TempoMap((0.0, 2.0), (120.0, 90.0))
    roll.model.tempo_map = tempo_map
    roll.load_model()

    roll.set_beats_per_bar(5)

    assert beats_drawn_as_bar_lines(
        roll, beats=11, seconds_at=tempo_map.seconds_at
    ) == [5, 10]
    roll.close()


@pytest.mark.parametrize("beats", [0, -4, 33, 2.5, "4", None, True])
def test_beats_per_bar_outside_one_to_thirty_two_are_refused(qapp, beats):
    roll = InteractivePianoRoll()

    with pytest.raises(ValueError):
        roll.set_beats_per_bar(beats)

    assert roll.beats_per_bar == 4


@pytest.mark.parametrize("beats", [1, 32])
def test_one_and_thirty_two_beats_per_bar_are_allowed(qapp, beats):
    roll = InteractivePianoRoll()

    roll.set_beats_per_bar(beats)

    assert roll.beats_per_bar == beats


def test_the_toolbar_offers_the_common_time_signatures(qapp):
    timeline = ArrangerTimeline()
    combo = timeline.combo_time_signature

    assert [combo.itemText(i) for i in range(combo.count())] == TIME_SIGNATURES
    assert combo.currentText() == "4/4"


def test_picking_five_four_in_the_toolbar_gives_five_beats_per_bar(qapp):
    timeline = shown_timeline()

    pick(timeline.combo_time_signature, "5/4")

    assert timeline.piano_roll.beats_per_bar == 5
    assert beats_drawn_as_bar_lines(timeline.piano_roll) == [5, 10]
    timeline.close()


def test_in_six_eight_the_tempo_beat_is_the_eighth_note_so_a_bar_has_six(
    qapp,
):
    timeline = shown_timeline()

    pick(timeline.combo_time_signature, "6/8")

    assert timeline.piano_roll.beats_per_bar == 6
    timeline.close()


def test_in_seven_eight_the_tempo_beat_is_the_eighth_note_so_a_bar_has_seven(
    qapp,
):
    timeline = shown_timeline()

    pick(timeline.combo_time_signature, "7/8")

    assert timeline.piano_roll.beats_per_bar == 7
    timeline.close()


def test_in_twelve_eight_the_tempo_beat_is_the_eighth_note_so_a_bar_has_twelve(
    qapp,
):
    timeline = shown_timeline()

    pick(timeline.combo_time_signature, "12/8")

    assert timeline.piano_roll.beats_per_bar == 12
    timeline.close()


def test_picking_a_time_signature_announces_its_beats_per_bar(qapp):
    timeline = shown_timeline()
    announced = []
    timeline.time_signature_changed.connect(announced.append)

    pick(timeline.combo_time_signature, "3/4")
    pick(timeline.combo_time_signature, "6/8")

    assert announced[0] == 3
    assert announced[-1] == 6
    assert 4 in announced[1:-1]
    timeline.close()


def test_the_time_signature_is_saved_and_restored_with_the_project(qapp):
    timeline = ArrangerTimeline()
    timeline.set_time_signature("7/8")
    saved = timeline.to_project_dict()

    restored = ArrangerTimeline()
    announced = []
    restored.time_signature_changed.connect(announced.append)
    restored.load_project_dict(saved)

    assert saved["time_signature"] == "7/8"
    assert restored.combo_time_signature.currentText() == "7/8"
    assert restored.piano_roll.beats_per_bar == 7
    assert announced == [7]


def test_a_project_saved_before_time_signatures_existed_opens_in_four_four(
    qapp,
):
    timeline = ArrangerTimeline()
    saved = timeline.to_project_dict()
    del saved["time_signature"]
    timeline.set_time_signature("3/4")
    announced = []
    timeline.time_signature_changed.connect(announced.append)

    timeline.load_project_dict(saved)

    assert timeline.combo_time_signature.currentText() == "4/4"
    assert timeline.piano_roll.beats_per_bar == 4
    assert announced == [4]


def test_a_project_with_an_unknown_time_signature_opens_in_four_four(qapp):
    timeline = ArrangerTimeline()
    saved = timeline.to_project_dict()
    saved["time_signature"] = "9/16"
    timeline.set_time_signature("3/4")

    timeline.load_project_dict(saved)

    assert timeline.combo_time_signature.currentText() == "4/4"
    assert timeline.piano_roll.beats_per_bar == 4


def test_clearing_the_editor_goes_back_to_four_four_and_says_so(qapp):
    timeline = shown_timeline(seconds=8.0)
    pick(timeline.combo_time_signature, "7/8")
    announced = []
    timeline.time_signature_changed.connect(announced.append)

    QTest.mouseClick(timeline.btn_clear, Qt.LeftButton)

    assert timeline.model.notes == []
    assert timeline.combo_time_signature.currentText() == "4/4"
    assert timeline.piano_roll.beats_per_bar == 4
    assert announced == [4]
    timeline.close()


def test_the_view_still_turns_the_page_when_the_playhead_runs_off_the_right(
    qapp,
):
    roll = shown_roll()
    _left, right = visible_span(roll)

    playing(roll, right * 10 - 1000, right * 10 - 100)

    left, right = visible_span(roll)
    assert left < roll.playhead.line().x1() < right
    assert left > 0
    roll.close()


def test_the_view_jumps_back_when_playback_restarts_at_the_beginning(qapp):
    roll = shown_roll()
    playing(roll, 20000, 30000, 40000)

    playing(roll, 500)

    left, right = visible_span(roll)
    assert left <= 50 < right
    roll.close()


def test_the_view_still_does_not_move_while_the_playhead_stays_on_screen(
    qapp,
):
    roll = shown_roll()
    before = visible_span(roll)

    playing(roll, 1000, 2000, 3000)

    assert visible_span(roll) == before
    roll.close()


def test_scrolling_ahead_while_playing_is_still_not_undone_by_following(
    qapp,
):
    roll = shown_roll()
    playing(roll, 1000)
    roll.horizontalScrollBar().setValue(3000)
    ahead = visible_span(roll)

    playing(roll, 1100, 1200, 1300)

    assert visible_span(roll) == ahead
    roll.close()


def test_scrolling_back_while_playing_is_not_undone_by_following(qapp):
    roll = shown_roll()
    playing(roll, 29000, 30000)
    roll.horizontalScrollBar().setValue(0)
    behind = visible_span(roll)

    playing(roll, 30100, 30200, 30300)

    assert visible_span(roll) == behind
    roll.close()


def test_a_playhead_that_jumps_back_behind_a_scrolled_ahead_view_is_followed(
    qapp,
):
    roll = shown_roll()
    playing(roll, 20000)
    roll.horizontalScrollBar().setValue(4000)

    playing(roll, 20100, 2000)

    left, right = visible_span(roll)
    assert left <= 200 < right
    roll.close()


def test_the_view_stays_where_it_is_when_following_is_switched_off(qapp):
    timeline = shown_timeline(seconds=60.0)
    roll = timeline.piano_roll
    before = visible_span(roll)

    QTest.mouseClick(timeline.btn_follow, Qt.LeftButton)
    playing(roll, 20000, 30000, 500)

    assert not timeline.btn_follow.isChecked()
    assert visible_span(roll) == before
    assert roll.playhead.line().x1() == 50
    timeline.close()


def test_switching_following_on_while_playing_brings_the_playhead_into_view(
    qapp,
):
    timeline = shown_timeline(seconds=60.0)
    roll = timeline.piano_roll
    QTest.mouseClick(timeline.btn_follow, Qt.LeftButton)
    playing(roll, 20000, 30000)

    QTest.mouseClick(timeline.btn_follow, Qt.LeftButton)

    left, right = visible_span(roll)
    assert timeline.btn_follow.isChecked()
    assert left < 3000 < right
    timeline.close()


def test_switching_following_on_while_stopped_leaves_the_view_alone(qapp):
    timeline = shown_timeline(seconds=60.0)
    roll = timeline.piano_roll
    QTest.mouseClick(timeline.btn_follow, Qt.LeftButton)
    playing(roll, 20000, 30000)
    roll.is_monitoring = False
    before = visible_span(roll)

    QTest.mouseClick(timeline.btn_follow, Qt.LeftButton)

    assert timeline.btn_follow.isChecked()
    assert visible_span(roll) == before
    timeline.close()


def test_the_follow_button_is_on_by_default(qapp):
    timeline = ArrangerTimeline()

    assert timeline.btn_follow.isCheckable()
    assert timeline.btn_follow.isChecked()
    assert timeline.piano_roll.follow_playhead


def test_zoom_to_fit_shows_a_five_minute_song_from_start_to_end(qapp):
    roll = shown_roll(seconds=300.0)

    roll.zoom_to_fit()

    left, right = visible_span(roll)
    assert left <= 0
    assert right >= roll.song_width()
    assert roll.song_width() > 0.9 * (right - left)
    roll.close()


def test_zoom_to_fit_zooms_in_on_a_short_song(qapp):
    roll = shown_roll(seconds=3.0)

    roll.zoom_to_fit()

    left, right = visible_span(roll)
    assert roll.transform().m11() > 1.0
    assert left <= 0 and right >= roll.song_width()
    roll.close()


def test_zoom_to_fit_on_an_empty_editor_shows_eight_bars(qapp):
    roll = shown_roll(seconds=0)

    roll.zoom_to_fit()

    left, right = visible_span(roll)
    eight_bars = 8 * 4 * roll.get_beat_width()
    assert left <= 0
    assert eight_bars <= right - left <= 1.1 * eight_bars
    roll.close()


def test_zoom_to_fit_keeps_the_height_of_the_note_rows(qapp):
    roll = shown_roll(seconds=300.0)
    rows = roll.transform().m22()

    roll.zoom_to_fit()

    assert roll.transform().m22() == rows
    roll.close()


def test_the_fit_button_sits_right_after_the_zoom_buttons(qapp):
    timeline = shown_timeline()

    plus = timeline.btn_zoom_in.geometry()
    fit = timeline.btn_zoom_fit.geometry()
    text = timeline.btn_zoom_fit.fontMetrics().horizontalAdvance(
        timeline.btn_zoom_fit.text()
    )

    assert fit.y() == plus.y()
    assert 0 < fit.x() - plus.right() <= 12
    assert fit.width() >= text + 12
    timeline.close()


def test_clicking_fit_shows_the_whole_song(qapp):
    timeline = shown_timeline(seconds=300.0)

    QTest.mouseClick(timeline.btn_zoom_fit, Qt.LeftButton)

    left, right = visible_span(timeline.piano_roll)
    assert left <= 0 and right >= timeline.piano_roll.song_width()
    timeline.close()


def test_time_labels_stay_apart_when_a_fifteen_minute_song_is_fitted(qapp):
    roll = shown_roll(seconds=900.0)

    roll.zoom_to_fit()
    settle()

    band = lightness(roll)[2:11]
    inked = (band < 128).any(axis=0)
    assert 0.02 < inked.mean() < 0.6
    roll.close()


@pytest.mark.parametrize(
    "factor", [0.0, -1.0, float("nan"), float("inf"), float("-inf")]
)
def test_a_nonsense_zoom_factor_leaves_the_view_usable(qapp, factor):
    roll = InteractivePianoRoll()

    roll.scale(factor, factor)

    assert roll.transform().isInvertible()
    assert roll.transform().m11() == 1.0
    assert roll.transform().m22() == 1.0


def test_zooming_out_and_in_a_thousand_times_still_never_breaks_the_view(
    qapp,
):
    roll = shown_roll()

    for _ in range(500):
        roll.zoom(0.5, 0.5)
    for _ in range(500):
        roll.zoom(3.0, 3.0)
    for _ in range(500):
        roll.scale(1e-9, 1e-9)

    assert roll.transform().isInvertible()
    assert 0.01 <= roll.transform().m11() <= 100.0
    assert 0.1 <= roll.transform().m22() <= 100.0
    roll.close()
