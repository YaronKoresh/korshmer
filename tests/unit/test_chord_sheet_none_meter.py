from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

from korshmer.analysis import create_chord_sheet


def test_create_chord_sheet_none_meter_does_not_crash():
    with tempfile.TemporaryDirectory() as td:
        audio_path = os.path.join(td, "test.wav")
        with patch("korshmer.analysis.analyse_tempo", return_value=120):
            with patch("korshmer.analysis.estimate_meter", return_value=None):
                with patch(
                    "korshmer.analysis.analyze_chords_full",
                    return_value=[(0.0, "C"), (2.0, "G")],
                ):
                    result = create_chord_sheet(audio_path, td)
        assert result is not None
        assert os.path.isfile(result)
        content = open(result).read()
        assert "Meter: 4/4" in content


def test_create_chord_sheet_none_meter_half_bar():
    with tempfile.TemporaryDirectory() as td:
        audio_path = os.path.join(td, "test.wav")
        with patch("korshmer.analysis.analyse_tempo", return_value=100):
            with patch("korshmer.analysis.estimate_meter", return_value=None):
                with patch(
                    "korshmer.analysis.analyze_chords_full",
                    return_value=[(0.0, "Am")],
                ) as mock_chords:
                    create_chord_sheet(
                        audio_path, td, precision_mode="Half Bar"
                    )
        _, kwargs = mock_chords.call_args
        assert kwargs["interval"] == (60.0 / 100) * 2.0


def test_create_chord_sheet_valid_meter_used():
    with tempfile.TemporaryDirectory() as td:
        audio_path = os.path.join(td, "test.wav")
        with patch("korshmer.analysis.analyse_tempo", return_value=120):
            with patch("korshmer.analysis.estimate_meter", return_value=3):
                with patch(
                    "korshmer.analysis.analyze_chords_full",
                    return_value=[(0.0, "D")],
                ):
                    result = create_chord_sheet(audio_path, td)
        content = open(result).read()
        assert "Meter: 3/4" in content
