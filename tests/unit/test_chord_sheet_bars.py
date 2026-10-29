from unittest import mock

import pytest

import korshmer.analysis as analysis


def make_sheet(tmp_path, mode, meter, count):
    chords = [(float(i), "CMaj") for i in range(count)]
    with (
        mock.patch.object(analysis, "analyse_tempo", return_value=60),
        mock.patch.object(analysis, "estimate_meter", return_value=meter),
        mock.patch.object(analysis, "analyze_chords_full", return_value=chords),
    ):
        out = analysis.create_chord_sheet("/x/song.wav", str(tmp_path), mode)
    return open(out, encoding="utf-8").read().splitlines()


def bar_headers(lines):
    return [line for line in lines if line.startswith("Bar ")]


@pytest.mark.parametrize(
    "mode, meter, count, expected_bars",
    [
        ("1 Bar", 3, 8, 8),
        ("1 Bar", 4, 5, 5),
        ("Every Beat", 3, 7, 3),
        ("Every Beat", 4, 8, 2),
        ("Half Bar", 4, 6, 3),
        ("2 Bars", 4, 4, 4),
    ],
)
def test_bar_count_follows_precision_mode(
    tmp_path, mode, meter, count, expected_bars
):
    lines = make_sheet(tmp_path, mode, meter, count)
    assert len(bar_headers(lines)) == expected_bars


def test_no_empty_bar_at_the_end(tmp_path):
    lines = make_sheet(tmp_path, "Every Beat", 4, 8)
    assert lines[-1].startswith("[")


def test_one_bar_mode_numbers_each_chord(tmp_path):
    lines = make_sheet(tmp_path, "1 Bar", 3, 3)
    assert bar_headers(lines) == ["Bar 1", "Bar 2", "Bar 3"]


def test_two_bars_mode_skips_bar_numbers(tmp_path):
    lines = make_sheet(tmp_path, "2 Bars", 4, 3)
    assert bar_headers(lines) == ["Bar 1", "Bar 3", "Bar 5"]
