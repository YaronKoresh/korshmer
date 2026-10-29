import json
import shutil
import subprocess

import numpy as np
import pretty_midi
import pyloudnorm
import pytest
import soundfile as sf

from korshmer.video import music_clip

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg is required",
)

SR = 44100


def _make_midi(path, bpm=120, extra_tracks=0):
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    piano = pretty_midi.Instrument(program=0, name="Keys")
    bass = pretty_midi.Instrument(program=33, name="Bass")
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="Kit")
    for i, chord in enumerate(([60, 64, 67], [65, 69, 72])):
        for p in chord:
            piano.notes.append(pretty_midi.Note(90, p, i * 1.0, i + 0.9))
    bass.notes.append(pretty_midi.Note(80, 36, 0.0, 1.9))
    drums.notes.append(pretty_midi.Note(100, 36, 0.0, 0.1))
    drums.notes.append(pretty_midi.Note(100, 38, 1.0, 1.1))
    pm.instruments.extend([piano, bass, drums])
    for k in range(extra_tracks):
        extra = pretty_midi.Instrument(program=(k * 5) % 100, name=f"X{k}")
        extra.notes.append(pretty_midi.Note(70, 50 + k % 20, 0.2, 0.8))
        pm.instruments.append(extra)
    pm.write(str(path))
    return pretty_midi.PrettyMIDI(str(path))


def _fake_render(
    midi_path, save_path, fluid_dir, sf3_path, bpm=120.0, stop_callback=None
):
    t = np.arange(int(2.2 * SR)) / SR
    tone = 0.3 * np.sin(2 * np.pi * 261.63 * t)
    sf.write(save_path, np.stack([tone, tone], axis=1), SR)
    return save_path


def _specs(tmp_path, pm, midi, **kwargs):
    wav = tmp_path / "song.wav"
    _fake_render(str(midi), str(wav), None, None)
    return music_clip.collect_tech_specs(pm, str(midi), str(wav), **kwargs), wav


def test_tech_specs_describe_the_song(tmp_path):
    midi = tmp_path / "song.mid"
    pm = _make_midi(midi)
    specs, _ = _specs(tmp_path, pm, midi)
    assert specs.track_count == 3
    assert specs.note_count == 9
    assert specs.tempo_label == "120 BPM"
    assert specs.time_signature == "4/4"
    assert specs.max_polyphony == 4
    assert specs.pitch_range == "C2 - C5"
    assert specs.sample_rate == SR
    assert specs.duration == pytest.approx(2.2, abs=0.01)
    assert [t.instrument for t in specs.tracks] == [
        "Grand Piano",
        "Bass Guitar",
        "Drums",
    ]


def test_tempo_comes_from_the_midi_file_not_from_the_caller(tmp_path):
    midi = tmp_path / "slow.mid"
    pm = _make_midi(midi, bpm=90)
    specs, _ = _specs(tmp_path, pm, midi)
    assert specs.tempo == pytest.approx(90, abs=0.01)
    assert specs.tempo_label == "90 BPM"


def test_tempo_changes_are_shown_as_a_range(tmp_path):
    midi = tmp_path / "varied.mid"
    pm = _make_midi(midi, bpm=100)
    import mido

    file = mido.MidiFile(str(midi))
    file.tracks[0].append(
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(140), time=480)
    )
    file.save(str(midi))
    pm = pretty_midi.PrettyMIDI(str(midi))
    specs, _ = _specs(tmp_path, pm, midi)
    assert specs.tempo_label == "100-140 BPM (1 changes)"


def test_loudness_matches_a_real_lufs_meter_before_and_after(tmp_path):
    midi = tmp_path / "song.mid"
    pm = _make_midi(midi)
    wav = tmp_path / "song.wav"
    _fake_render(str(midi), str(wav), None, None)
    quiet = tmp_path / "raw.wav"
    tone, _ = sf.read(str(wav))
    sf.write(str(quiet), tone * 0.25, SR)
    specs = music_clip.collect_tech_specs(
        pm, str(midi), str(wav), raw_audio_path=str(quiet)
    )
    meter = pyloudnorm.Meter(SR)
    assert specs.loudness_lufs == pytest.approx(
        meter.integrated_loudness(tone), abs=0.05
    )
    assert specs.raw_loudness_lufs == pytest.approx(
        meter.integrated_loudness(tone * 0.25), abs=0.05
    )
    assert specs.raw_loudness_lufs < specs.loudness_lufs
    labels = [label for label, _ in specs.lines()]
    assert labels[-3:] == ["LUFS before", "LUFS mastered", "True peak"]
    assert specs.true_peak_dbtp >= -10.6


def test_unmeasurable_audio_is_reported_as_not_available(tmp_path):
    assert music_clip.measure_loudness(np.zeros((SR, 2)), SR) == (None, None)
    lufs, _ = music_clip.measure_loudness(np.full((100, 2), 0.1), SR)
    assert lufs is None


def test_notes_fall_onto_the_keyboard(tmp_path):
    midi = tmp_path / "song.mid"
    pm = _make_midi(midi)
    specs, wav = _specs(tmp_path, pm, midi)
    scene = music_clip.ClipScene(pm, specs, str(wav), scale=1.0)
    key_top = scene._keys[1]
    x0, x1 = scene._key_rect(65)
    column = int((x0 + x1) / 2)

    base = specs.tracks[0].color
    shades = {base, tuple(int(c * 0.7) for c in base)}

    def colored_rows(frame):
        data = np.asarray(frame)
        return [
            y
            for y in range(scene._roll[1], key_top - 2)
            if tuple(data[y, column]) in shades
        ]

    early = colored_rows(scene.frame(0.2))
    later = colored_rows(scene.frame(0.9))
    assert early and later
    assert max(later) > max(early)
    assert max(early) < key_top
    assert scene.chord_at(0.5) == "C"
    assert scene.chord_at(1.5) == "F/C"


def test_active_keys_light_up_and_progress_advances(tmp_path):
    midi = tmp_path / "song.mid"
    pm = _make_midi(midi)
    specs, wav = _specs(tmp_path, pm, midi)
    scene = music_clip.ClipScene(pm, specs, str(wav))
    first = np.asarray(scene.frame(0.5))
    last = np.asarray(scene.frame(2.1))
    key_y = scene._keys[1] + 3
    x0, x1 = scene._key_rect(64)
    lit = tuple(first[key_y, int((x0 + x1) / 2)])
    assert lit != music_clip.WHITE_KEY
    box = scene._box(music_clip.WAVE_BOX)
    bright_first = np.all(
        first[box[1] : box[3], box[0] : box[2]] == music_clip.WAVE_BRIGHT,
        axis=2,
    )
    bright_last = np.all(
        last[box[1] : box[3], box[0] : box[2]] == music_clip.WAVE_BRIGHT,
        axis=2,
    )
    assert bright_last.sum() > bright_first.sum() > 0


def test_too_many_tracks_are_announced_in_the_clip(tmp_path):
    midi = tmp_path / "big.mid"
    pm = _make_midi(midi, extra_tracks=25)
    specs, wav = _specs(tmp_path, pm, midi)
    capacity = music_clip.listed_track_capacity()
    assert specs.track_count == 28
    assert specs.listed_tracks == capacity < specs.track_count
    assert dict(specs.lines())["Tracks"] == f"28 ({capacity} listed)"
    scene = music_clip.ClipScene(pm, specs, str(wav))
    frame = np.asarray(scene.frame(0.5))
    box = scene._box(music_clip.INFO_BOX)
    region = frame[box[1] : box[3], box[0] : box[2]]
    assert np.any(np.all(region == music_clip.WARNING, axis=2))


def test_few_tracks_show_no_overflow_message(tmp_path):
    midi = tmp_path / "song.mid"
    pm = _make_midi(midi)
    specs, wav = _specs(tmp_path, pm, midi)
    scene = music_clip.ClipScene(pm, specs, str(wav))
    frame = np.asarray(scene.frame(0.5))
    box = scene._box(music_clip.INFO_BOX)
    region = frame[box[1] : box[3], box[0] : box[2]]
    assert not np.any(np.all(region == music_clip.WARNING, axis=2))
    assert dict(specs.lines())["Tracks"] == "3"


def test_track_colors_are_distinct_for_many_tracks():
    colors = {music_clip.track_color(i, 40) for i in range(40)}
    assert len(colors) == 40


def _probe(path):
    return json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ]
        )
    )


def test_export_defaults_to_4k_video_with_audio_and_mastered_wav(tmp_path):
    midi = tmp_path / "song.mid"
    _make_midi(midi, bpm=90)
    out = tmp_path / "clip.mp4"
    steps = []
    result = music_clip.export_music_clip(
        str(midi),
        str(out),
        None,
        None,
        bpm=140,
        fps=5,
        progress_callback=lambda p, t: steps.append(p),
        render_func=_fake_render,
    )
    assert result == str(out)
    assert (tmp_path / "clip.wav").exists()
    assert steps[-1] == 100
    probe = _probe(out)
    assert {s["codec_type"] for s in probe["streams"]} == {"video", "audio"}
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (3840, 2160)


def test_export_can_pick_a_smaller_resolution(tmp_path):
    midi = tmp_path / "song.mid"
    _make_midi(midi)
    out = tmp_path / "small.mp4"
    music_clip.export_music_clip(
        str(midi),
        str(out),
        None,
        None,
        fps=5,
        resolution="720p",
        render_func=_fake_render,
    )
    video = next(
        s for s in _probe(out)["streams"] if s["codec_type"] == "video"
    )
    assert (video["width"], video["height"]) == (1280, 720)


def test_export_without_mastering_keeps_rendered_audio(tmp_path):
    midi = tmp_path / "song.mid"
    _make_midi(midi)
    out = tmp_path / "plain.mp4"
    music_clip.export_music_clip(
        str(midi),
        str(out),
        None,
        None,
        fps=5,
        master=False,
        resolution="720p",
        render_func=_fake_render,
    )
    y, sr = sf.read(tmp_path / "plain.wav")
    assert sr == SR
    assert np.max(np.abs(y)) == pytest.approx(0.3, abs=0.01)


def test_export_can_be_cancelled(tmp_path):
    midi = tmp_path / "song.mid"
    _make_midi(midi)
    with pytest.raises(music_clip.ClipExportCancelled):
        music_clip.export_music_clip(
            str(midi),
            str(tmp_path / "x.mp4"),
            None,
            None,
            fps=5,
            stop_callback=lambda: True,
            render_func=_fake_render,
        )


def test_failed_render_is_reported(tmp_path):
    midi = tmp_path / "song.mid"
    _make_midi(midi)
    with pytest.raises(
        music_clip.ClipExportError, match="could not be rendered"
    ):
        music_clip.export_music_clip(
            str(midi),
            str(tmp_path / "x.mp4"),
            None,
            None,
            render_func=lambda *a, **k: None,
        )


def test_invalid_options_are_rejected(tmp_path):
    out = str(tmp_path / "x.mp4")
    with pytest.raises(music_clip.ClipExportError, match="fps"):
        music_clip.export_music_clip("a.mid", out, None, None, fps=0)
    with pytest.raises(music_clip.ClipExportError, match="resolution"):
        music_clip.export_music_clip("a.mid", out, None, None, resolution="8K")


def test_missing_ffmpeg_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(music_clip.shutil, "which", lambda name: None)
    with pytest.raises(music_clip.ClipExportError, match="ffmpeg"):
        music_clip.export_music_clip(
            "a.mid", str(tmp_path / "x.mp4"), None, None
        )


def test_notes_now_list_is_capped_and_counts_the_rest(tmp_path):
    midi = tmp_path / "big.mid"
    pm = _make_midi(midi, extra_tracks=18)
    specs, wav = _specs(tmp_path, pm, midi)
    scene = music_clip.ClipScene(pm, specs, str(wav))
    seen = []

    from PIL import ImageDraw

    real_text = ImageDraw.ImageDraw.text

    def spy(self, xy, text, *args, **kwargs):
        seen.append(text)
        return real_text(self, xy, text, *args, **kwargs)

    ImageDraw.ImageDraw.text = spy
    try:
        scene.frame(0.5)
    finally:
        ImageDraw.ImageDraw.text = real_text
    notes_line = next(
        t
        for t in seen
        if isinstance(t, str) and "+" in t and "C" in t and "  " in t
    )
    assert len(notes_line.split("  ")) == music_clip.MAX_NOTE_NAMES + 1


def _tone_clip(path, hz, seconds, start, gain=1.0, muted=False):
    t = np.arange(int(seconds * SR)) / SR
    sf.write(str(path), 0.4 * np.sin(2 * np.pi * hz * t), SR)
    return {
        "path": str(path),
        "start": start,
        "gain": gain,
        "muted": muted,
    }


def _band_energy(y, sr, lo, hi, t0, t1):
    seg = y[int(t0 * sr) : int(t1 * sr)]
    seg = seg[:, 0] if seg.ndim > 1 else seg
    spec = np.abs(np.fft.rfft(seg)) / max(len(seg), 1)
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    return float(spec[(freqs >= lo) & (freqs <= hi)].max())


def test_export_mixes_audio_clips_with_the_midi_render(tmp_path):
    midi = tmp_path / "song.mid"
    _make_midi(midi)
    clip = _tone_clip(tmp_path / "vox.wav", 880, 3.0, 1.0)
    out = tmp_path / "clip.mp4"
    music_clip.export_music_clip(
        str(midi),
        str(out),
        None,
        None,
        fps=5,
        resolution="720p",
        master=False,
        render_func=_fake_render,
        clips=[clip],
    )
    y, sr = sf.read(str(tmp_path / "clip.wav"))
    assert len(y) / sr == pytest.approx(4.0, abs=0.05)
    assert _band_energy(y, sr, 255, 270, 0.0, 0.9) > 0.05
    assert _band_energy(y, sr, 870, 890, 0.0, 0.9) < 0.005
    assert _band_energy(y, sr, 870, 890, 1.2, 3.8) > 0.05
    audio = next(
        s for s in _probe(out)["streams"] if s["codec_type"] == "audio"
    )
    assert float(audio["duration"]) == pytest.approx(4.0, abs=0.2)


def test_muted_audio_clips_stay_out_of_the_clip(tmp_path):
    midi = tmp_path / "song.mid"
    _make_midi(midi)
    clip = _tone_clip(tmp_path / "vox.wav", 880, 3.0, 1.0, muted=True)
    music_clip.export_music_clip(
        str(midi),
        str(tmp_path / "clip.mp4"),
        None,
        None,
        fps=5,
        resolution="720p",
        master=False,
        render_func=_fake_render,
        clips=[clip],
    )
    y, sr = sf.read(str(tmp_path / "clip.wav"))
    assert len(y) / sr == pytest.approx(2.2, abs=0.05)


def test_clip_gain_is_applied_in_the_clip(tmp_path):
    midi = tmp_path / "song.mid"
    _make_midi(midi)
    clip = _tone_clip(tmp_path / "vox.wav", 880, 2.0, 0.0, gain=0.25)
    music_clip.export_music_clip(
        str(midi),
        str(tmp_path / "clip.mp4"),
        None,
        None,
        fps=5,
        resolution="720p",
        master=False,
        render_func=_fake_render,
        clips=[clip],
    )
    y, sr = sf.read(str(tmp_path / "clip.wav"))
    vox = _band_energy(y, sr, 870, 890, 0.2, 1.8)
    assert vox == pytest.approx(0.4 * 0.25 / 2, rel=0.1)


def test_audio_clips_alone_can_make_a_clip(tmp_path):
    midi = tmp_path / "empty.mid"
    pretty_midi.PrettyMIDI(initial_tempo=120).write(str(midi))
    clip = _tone_clip(tmp_path / "vox.wav", 880, 2.0, 0.0)
    calls = []

    def render(*a, **k):
        calls.append(a)
        return None

    music_clip.export_music_clip(
        str(midi),
        str(tmp_path / "clip.mp4"),
        None,
        None,
        fps=5,
        resolution="720p",
        master=False,
        render_func=render,
        clips=[clip],
    )
    y, sr = sf.read(str(tmp_path / "clip.wav"))
    assert calls == []
    assert _band_energy(y, sr, 870, 890, 0.2, 1.8) > 0.05
