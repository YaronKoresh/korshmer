import pytest

from korshmer.audio_processing import stem_session as ss
from korshmer.audio_processing.music_regions import MusicRegionOptions
from korshmer.audio_processing.separation_engine import SeparationOptions
from korshmer.workers.generic import WorkerThread
from korshmer.workers.stem_separation import StemSeparationThread

PARAMS = {
    "octave": "Original",
    "root": "Auto",
    "scale": "Auto (Detect)",
    "grid": "1/16",
    "quantize": 1.0,
    "swing": 0.0,
    "vel_mode": "Original",
    "vel_val": 0.0,
    "stems": ["bass"],
    "export": "Merged",
    "gm_drums": True,
    "cc_gen": False,
    "wav32": False,
}


def _run_worker(tmp_path, params):
    song = tmp_path / "song.wav"
    song.write_bytes(b"")
    seen = []

    def fake_process(*args, **kwargs):
        seen.append(kwargs)
        return []

    worker = WorkerThread([str(song)], params, process_track_func=fake_process)
    worker._process_single_file(0, str(song), 1)
    return seen[0]


def test_worker_forwards_the_music_and_separation_options(tmp_path):
    music = MusicRegionOptions()
    separation = SeparationOptions(quality="best")

    kwargs = _run_worker(
        tmp_path,
        {**PARAMS, "music_options": music, "separation_options": separation},
    )

    assert kwargs["music_options"] is music
    assert kwargs["separation_options"] is separation


def test_worker_leaves_the_options_out_when_the_ui_gave_none(tmp_path):
    kwargs = _run_worker(tmp_path, {**PARAMS, "music_options": None})

    assert "music_options" not in kwargs
    assert "separation_options" not in kwargs


@pytest.mark.parametrize("options", [None, SeparationOptions(quality="fast")])
def test_stem_thread_hands_its_options_to_the_separation(
    options, monkeypatch, tmp_path
):
    seen = {}

    def fake_separate(_path, **kwargs):
        seen.update(kwargs)
        return ss.StemSession(
            source_path="song.wav",
            output_dir=str(tmp_path),
            tempo=100.0,
            root="C",
            scale="Major",
        )

    monkeypatch.setattr(ss, "separate_stems", fake_separate)
    thread = StemSeparationThread("song.wav", options=options)
    thread.run()

    assert thread.options is options
    if options is None:
        assert "options" not in seen
    else:
        assert seen["options"] is options
