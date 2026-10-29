import pytest

from korshmer.midi.transcription_cleanup import CleanupOptions
from tests.fixtures import transcription_eval as ev
from tests.fixtures.transcription_fixtures import load_truth

mir_eval = pytest.importorskip("mir_eval")


def test_mir_scores_perfect_and_shifted():
    truth = [(60, 0.0, 0.5, 90), (64, 0.5, 1.0, 90), (67, 1.0, 1.5, 90)]
    assert ev.mir_scores(truth, truth)[2] == pytest.approx(1.0)
    late = [(p, s + 0.2, e + 0.2, v) for p, s, e, v in truth]
    assert ev.mir_scores(truth, late)[2] == 0.0
    assert ev.mir_scores(truth, [])[2] == 0.0
    off = [(p, s, e + 0.4, v) for p, s, e, v in truth]
    assert ev.mir_scores(truth, off)[2] == pytest.approx(1.0)
    assert ev.mir_scores(truth, off, with_offset=True)[2] < 1.0


def test_format_table_lists_every_song():
    results = {
        "a": {"precision": 1, "recall": 1, "f1": 1, "f1_offset": 1, "notes": 3},
        "mean": {"precision": 1, "recall": 1, "f1": 1, "f1_offset": 1},
    }
    text = ev.format_table(results)
    assert "a" in text and "mean" in text


@pytest.mark.skipif(
    not ev.model_available(), reason="basic_pitch model is not installed"
)
def test_real_model_scores_on_fixture_song():
    name = "arp_f_major"
    notes = ev.run_model(name, cache=False)
    assert notes
    truth = load_truth(name)
    assert ev.mir_scores(truth, notes)[2] > 0.6


@pytest.mark.skipif(
    not ev.model_available(), reason="basic_pitch model is not installed"
)
def test_evaluate_reports_every_song_and_mean():
    results = ev.evaluate(names=["bass_a_minor"])
    assert set(results) == {"bass_a_minor", "mean"}
    assert 0.0 <= results["mean"]["f1"] <= 1.0


@pytest.mark.skipif(
    not ev.model_available(), reason="basic_pitch model is not installed"
)
def test_two_audio_variants_beat_one_on_average():
    names = ["piano_c_major", "guitar_g_major"]
    one = ev.evaluate(names=names, ensemble=False)["mean"]["f1"]
    two = ev.evaluate(names=names, ensemble=True)["mean"]["f1"]
    assert two > one


@pytest.mark.skipif(
    not ev.model_available(), reason="basic_pitch model is not installed"
)
def test_default_cleanup_keeps_f1_and_improves_offset_f1():
    names = ["piano_c_major", "guitar_g_major"]
    raw = ev.evaluate(names=names)["mean"]
    cleaned = ev.evaluate(cleanup=CleanupOptions(), names=names)["mean"]
    assert cleaned["f1"] >= raw["f1"] - 0.005
    assert cleaned["f1_offset"] > raw["f1_offset"]
