import os
import shutil
import sys
import time

import numpy as np
import pretty_midi
import pytest
import soundfile as sf
from PySide6.QtWidgets import QApplication, QMessageBox

if not QApplication.instance():
    QApplication(sys.argv)

from korshmer import audio_processors
from korshmer.audio_processing.smart_mastering import (
    SmartMastering,
    SmartMasteringConfig,
)
from korshmer.event_bus import Event, subscribe, unsubscribe
from korshmer.exceptions import (
    MasteringCancelledError,
    ProcessingCancelledError,
)
from korshmer.mixins.actions.audio import AudioActionsMixin
from korshmer.video import music_clip

RATE = 44100
BAND_COUNT = 40


def _song(seconds=2.0, rate=RATE):
    time_axis = np.arange(int(seconds * rate)) / rate
    rng = np.random.default_rng(5)
    bass = 0.3 * np.sin(2 * np.pi * 60 * time_axis)
    chord = 0.1 * sum(
        np.sin(2 * np.pi * pitch * time_axis + index)
        for index, pitch in enumerate((220, 330, 440, 1760, 3520))
    )
    hats = 0.04 * rng.normal(0, 1, time_axis.shape)
    left = bass + chord + hats
    right = bass + 0.8 * chord + np.roll(hats, 100)
    return np.stack([left, right])


def _files_in(folder):
    return sorted(os.listdir(folder)) if os.path.isdir(folder) else []


def _master(song, rate=RATE, config=None, **callbacks):
    return SmartMastering(config or SmartMasteringConfig()).process(
        song, rate, **callbacks
    )


@pytest.fixture(scope="module")
def reported():
    song = _song()
    reports = []
    result, _rate = _master(
        song,
        progress_callback=lambda percent, name: reports.append((percent, name)),
    )
    return song, result, reports


def test_progress_only_moves_forward_and_ends_at_one_hundred(reported):
    _song_in, _result, reports = reported
    percents = [percent for percent, _name in reports]

    assert len(percents) > BAND_COUNT
    assert percents[0] < 5.0
    assert percents == sorted(percents)
    assert all(0.0 <= percent <= 100.0 for percent in percents)
    assert percents[-1] == 100.0


def test_every_report_names_the_current_step_in_plain_words(reported):
    _song_in, _result, reports = reported
    names = [name for _percent, name in reports]
    steps = list(dict.fromkeys(name.split(" (")[0] for name in names))

    assert steps == [
        "Preparing the song",
        "Balancing the tone",
        "Evening out loud and quiet parts",
        "Warming the bass",
        "Adding treble harmonics",
        "Filling in the missing treble",
        "Bringing out quiet details",
        "Widening the stereo image",
        "Setting the final loudness",
        "Mastering finished",
    ]


def test_the_treble_harmonics_step_reports_each_of_its_forty_bands(reported):
    _song_in, _result, reports = reported
    bands = [name for _percent, name in reports if "band " in name]

    assert bands == [
        f"Adding treble harmonics (band {number} of {BAND_COUNT})"
        for number in range(1, BAND_COUNT + 1)
    ]


def test_the_treble_harmonics_step_moves_the_bar_while_it_works(reported):
    _song_in, _result, reports = reported
    during = [percent for percent, name in reports if "band " in name]

    assert len(set(during)) >= BAND_COUNT // 2
    assert during[-1] - during[0] > 30.0


def test_the_final_loudness_step_reports_each_pass(reported):
    _song_in, _result, reports = reported
    passes = [name for _percent, name in reports if "pass " in name]

    assert passes[0] == "Setting the final loudness (pass 1)"
    assert passes == [
        f"Setting the final loudness (pass {number})"
        for number in range(1, len(passes) + 1)
    ]


def test_watching_progress_does_not_change_the_mastered_sound(reported):
    song, watched, _reports = reported

    unwatched, _rate = _master(song)
    never_stopped, _rate = _master(song, stop_callback=lambda: False)

    assert np.array_equal(watched, unwatched)
    assert np.array_equal(never_stopped, unwatched)


def test_progress_still_ends_at_one_hundred_when_treble_and_bass_steps_are_off():
    reports = []
    _master(
        _song(1.0),
        config=SmartMasteringConfig(exciter_mix=0.0, bass_enhance_mix=0.0),
        progress_callback=lambda percent, name: reports.append((percent, name)),
    )
    percents = [percent for percent, _name in reports]
    names = {name.split(" (")[0] for _percent, name in reports}

    assert percents[-1] == 100.0
    assert percents == sorted(percents)
    assert max(np.diff(percents)) < 60.0
    assert "Adding treble harmonics" not in names
    assert "Warming the bass" not in names


def test_asking_to_stop_ends_mastering_with_its_own_error():
    with pytest.raises(MasteringCancelledError) as stopped:
        _master(_song(1.0), stop_callback=lambda: True)

    assert isinstance(stopped.value, ProcessingCancelledError)
    assert "cancelled" in str(stopped.value).lower()


def test_stopping_during_treble_harmonics_ends_before_the_later_steps():
    reports = []

    def watch(percent, name):
        reports.append(name)

    def stop_after_third_band():
        return any("band 3 of" in name for name in reports)

    with pytest.raises(MasteringCancelledError):
        _master(
            _song(),
            progress_callback=watch,
            stop_callback=stop_after_third_band,
        )

    assert reports[-1] == f"Adding treble harmonics (band 3 of {BAND_COUNT})"
    assert not any("final loudness" in name for name in reports)


@pytest.fixture
def song_file(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_processors, "_PROC_DIR", str(tmp_path / "out"))
    path = tmp_path / "demo.wav"
    sf.write(path, _song(1.5).T, RATE)
    return str(path)


def test_mastering_a_file_reports_progress_and_writes_the_mastered_file(
    song_file, tmp_path
):
    reports = []

    out_path = audio_processors.process_smart_mastering_file(
        song_file,
        progress_callback=lambda percent, name: reports.append((percent, name)),
    )

    assert reports[-1] == (100.0, "Mastering finished")
    assert any("Adding treble harmonics" in name for _percent, name in reports)
    assert _files_in(tmp_path / "out") == ["demo_Mastered.wav"]
    assert sf.info(out_path).frames == int(1.5 * RATE)


def test_mastering_a_file_sounds_the_same_with_and_without_progress(
    song_file,
):
    plain, _rate = sf.read(
        audio_processors.process_smart_mastering_file(song_file)
    )
    watched, _rate = sf.read(
        audio_processors.process_smart_mastering_file(
            song_file,
            progress_callback=lambda percent, name: None,
            stop_callback=lambda: False,
        )
    )

    assert np.array_equal(plain, watched)


def test_cancelling_file_mastering_leaves_no_file_behind(song_file, tmp_path):
    reports = []

    with pytest.raises(MasteringCancelledError):
        audio_processors.process_smart_mastering_file(
            song_file,
            progress_callback=lambda percent, name: reports.append(name),
            stop_callback=lambda: any("band 2 of" in name for name in reports),
        )

    assert _files_in(tmp_path / "out") == []


def test_a_cancel_that_arrives_while_saving_keeps_the_earlier_mastered_file(
    song_file, tmp_path
):
    earlier = audio_processors.process_smart_mastering_file(song_file)
    with open(earlier, "rb") as handle:
        earlier_bytes = handle.read()
    louder = tmp_path / "demo.wav"
    sf.write(louder, 0.5 * _song(1.5).T, RATE)
    reports = []

    with pytest.raises(MasteringCancelledError):
        audio_processors.process_smart_mastering_file(
            str(louder),
            progress_callback=lambda percent, name: reports.append(percent),
            stop_callback=lambda: bool(reports) and reports[-1] == 100.0,
        )

    assert _files_in(tmp_path / "out") == ["demo_Mastered.wav"]
    with open(earlier, "rb") as handle:
        assert handle.read() == earlier_bytes


class Window(AudioActionsMixin):
    is_processing = False

    def __init__(self, path):
        self.path = path

    def _get_selected_file(self, is_input=False, requires_midi=False):
        return self.path

    def _dialog_parent(self):
        return None


@pytest.fixture
def screen(monkeypatch):
    seen = {
        "bar": [],
        "progress": [],
        "status": [],
        "button": [],
        "button_enabled": [],
        "info": [],
        "critical": [],
        "question": [],
    }
    handlers = {
        Event.PROGRESS_BAR_STATE: lambda visible=False, value=0: seen[
            "bar"
        ].append(visible),
        Event.PROGRESS_UPDATE: lambda value=0, text="": seen["progress"].append(
            (value, text)
        ),
        Event.STATUS_UPDATE: lambda text="": seen["status"].append(text),
        Event.CONVERT_BUTTON_TEXT: lambda text="": seen["button"].append(text),
        Event.CONVERT_BUTTON_STATE: lambda enabled=True: seen[
            "button_enabled"
        ].append(enabled),
    }
    for event, handler in handlers.items():
        subscribe(event, handler)
    monkeypatch.setattr(
        QMessageBox, "information", lambda *a, **k: seen["info"].append(a[2])
    )
    monkeypatch.setattr(
        QMessageBox, "critical", lambda *a, **k: seen["critical"].append(a[2])
    )

    def ask(*args, **kwargs):
        seen["question"].append(args[2])
        return QMessageBox.Yes

    monkeypatch.setattr(QMessageBox, "question", ask)
    yield seen
    for event, handler in handlers.items():
        unsubscribe(event, handler)


def _pump(condition, timeout=60):
    end = time.time() + timeout
    while time.time() < end and not condition():
        QApplication.processEvents()
        time.sleep(0.005)
    QApplication.processEvents()
    return condition()


def test_the_mastering_tool_shows_a_moving_bar_with_the_step_name(
    song_file, screen
):
    window = Window(song_file)

    window.run_smart_mastering()

    assert screen["bar"] == [True]
    assert screen["button"] == ["Stop Mastering"]
    assert _pump(lambda: screen["info"])
    values = [value for value, _text in screen["progress"]]
    texts = [text for _value, text in screen["progress"]]
    assert values == sorted(values)
    assert values[-1] == 100
    assert len(set(values)) > 10
    assert any(
        text.startswith("Mastering ") and "Adding treble harmonics" in text
        for text in texts
    )
    assert screen["bar"] == [True, False]
    assert screen["button"] == ["Stop Mastering", "Start Processing"]
    assert screen["info"] == ["Mastering Complete: demo_Mastered.wav"]
    assert not window.cancellable_action_is_running()


def test_cancelling_the_mastering_tool_stops_it_without_an_error_box(
    song_file, screen, tmp_path
):
    sf.write(song_file, _song(8.0).T, RATE)
    window = Window(song_file)
    window.run_smart_mastering()
    assert _pump(lambda: screen["progress"])

    assert window.cancel_running_action()

    assert _pump(lambda: "Mastering cancelled." in screen["status"])
    assert "Stopping mastering..." in screen["status"]
    assert screen["bar"] == [True, False]
    assert screen["button"][-1] == "Start Processing"
    assert screen["info"] == []
    assert screen["critical"] == []
    assert _files_in(tmp_path / "out") == []
    assert not window.cancellable_action_is_running()


def test_the_stop_button_works_during_mastering_even_with_no_input_files(
    song_file, screen
):
    window = Window(song_file)
    window.file_list_paths = []

    window.run_smart_mastering()

    assert screen["button_enabled"] == [True]
    assert _pump(lambda: screen["info"])
    assert screen["button_enabled"] == [True, False]


def test_choosing_mastering_again_while_it_runs_offers_to_stop_it(
    song_file, screen
):
    sf.write(song_file, _song(8.0).T, RATE)
    window = Window(song_file)
    window.run_smart_mastering()
    assert _pump(lambda: screen["progress"])

    window.run_smart_mastering()

    assert screen["question"] == ["Mastering is still running.\nStop it?"]
    assert _pump(lambda: "Mastering cancelled." in screen["status"])
    assert screen["critical"] == []


def test_mastering_leaves_the_bar_alone_while_a_conversion_is_using_it(
    song_file, screen
):
    window = Window(song_file)
    window.is_processing = True

    window.run_smart_mastering()

    assert _pump(lambda: screen["info"])
    assert screen["bar"] == []
    assert screen["button"] == []
    assert screen["progress"] == []
    assert any("Adding treble harmonics" in text for text in screen["status"])


def test_the_main_stop_button_cancels_a_running_mastering():
    main = pytest.importorskip("korshmer.__main__")
    calls = []

    class Stand:
        is_processing = False

        def cancellable_action_is_running(self):
            return True

        def cancel_running_action(self):
            calls.append("cancel")

        def start_conversion_logic(self):
            calls.append("convert")

        def stop_conversion(self):
            calls.append("stop conversion")

    main.MainWindow.toggle_processing(Stand())

    assert calls == ["cancel"]


def _midi(path):
    song = pretty_midi.PrettyMIDI(initial_tempo=120)
    keys = pretty_midi.Instrument(program=0, name="Keys")
    keys.notes.append(pretty_midi.Note(90, 60, 0.0, 0.9))
    song.instruments.append(keys)
    song.write(str(path))


def _render(
    midi_path, save_path, fluid_dir, sf3_path, bpm=120.0, stop_callback=None
):
    sf.write(save_path, _song(1.5).T, RATE)
    return save_path


needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg is required",
)


@needs_ffmpeg
def test_exporting_a_music_clip_reports_the_mastering_steps(tmp_path):
    _midi(tmp_path / "song.mid")
    reports = []

    music_clip.export_music_clip(
        str(tmp_path / "song.mid"),
        str(tmp_path / "clip.mp4"),
        None,
        None,
        fps=5,
        resolution="720p",
        render_func=_render,
        progress_callback=lambda percent, text: reports.append((percent, text)),
    )

    mastering = [
        (percent, text)
        for percent, text in reports
        if text.startswith("Smart Mastering: ")
    ]
    percents = [percent for percent, _text in reports]
    assert any(
        "Adding treble harmonics (band" in text for _p, text in mastering
    )
    assert len({percent for percent, _text in mastering}) > 5
    assert all(12 <= percent <= 24 for percent, _text in mastering)
    assert percents == sorted(percents)


@needs_ffmpeg
def test_cancelling_a_music_clip_during_mastering_stops_the_export(tmp_path):
    _midi(tmp_path / "song.mid")
    reports = []

    with pytest.raises(music_clip.ClipExportCancelled):
        music_clip.export_music_clip(
            str(tmp_path / "song.mid"),
            str(tmp_path / "clip.mp4"),
            None,
            None,
            fps=5,
            resolution="720p",
            render_func=_render,
            progress_callback=lambda percent, text: reports.append(text),
            stop_callback=lambda: any("band 2 of" in text for text in reports),
        )

    assert not any("Analyzing the song" in text for text in reports)
    assert not os.path.exists(tmp_path / "clip.wav")
    assert not os.path.exists(tmp_path / "clip.mp4")
