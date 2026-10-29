from korshmer.ui.viewmodels.piano_roll_vm import NoteData, PianoRollViewModel
from korshmer.ui.viewmodels.spectrogram_vm import SpectrogramViewModel
from korshmer.ui.viewmodels.visualizer_vm import VisualizerViewModel


class TestNoteData:
    def test_y_calculation(self):
        note = NoteData(pitch=60, x=0, width=100, velocity=80, grid_x=50)
        assert note.y == (127 - 60) * 20

    def test_snap_position(self):
        note = NoteData(
            pitch=60, x=0, width=100, velocity=80, grid_x=50, grid_y=20
        )
        x, y = note.snap_position(47, 38)
        assert x == 50
        assert y == 40

    def test_snap_clamps_min(self):
        note = NoteData(
            pitch=60, x=0, width=100, velocity=80, grid_x=50, grid_y=20
        )
        x, y = note.snap_position(-10, -10)
        assert x == 0
        assert y == 0

    def test_snap_clamps_max(self):
        note = NoteData(
            pitch=60, x=0, width=100, velocity=80, grid_x=50, grid_y=20
        )
        x, y = note.snap_position(100, 127 * 20 + 50)
        assert y == 127 * 20


class TestPianoRollViewModel:
    def test_defaults(self):
        vm = PianoRollViewModel()
        assert vm.grid_size == 20
        assert vm.pixels_per_second == 100.0
        assert vm.tempo == 120.0
        assert vm.root_note == "C"
        assert vm._root_note_idx == 0

    def test_total_divisions(self):
        vm = PianoRollViewModel()
        assert vm.total_divisions == 128

    def test_beat_width(self):
        vm = PianoRollViewModel()
        expected = (60.0 / 120.0) * 100.0
        assert abs(vm.beat_width - expected) < 0.001

    def test_set_scale(self):
        vm = PianoRollViewModel()
        vm.set_scale("A", "Minor")
        assert vm.root_note == "A"
        assert vm._root_note_idx == 9

    def test_set_scale_auto(self):
        vm = PianoRollViewModel()
        vm.set_scale("Auto", "Auto (Detect)")
        assert vm.root_note == "C"

    def test_set_scale_invalid(self):
        vm = PianoRollViewModel()
        vm.set_scale("ZZZ", "Major")
        assert vm.root_note == "C"
        assert vm._root_note_idx == 0

    def test_set_divisions(self):
        vm = PianoRollViewModel()
        old = vm.set_divisions(24)
        assert old == 12
        assert vm.divisions_per_octave == 24
        assert vm.total_divisions == 128 * 2

    def test_set_divisions_invalid(self):
        vm = PianoRollViewModel()
        result = vm.set_divisions(7)
        assert result == 12
        assert vm.divisions_per_octave == 12

    def test_compute_tempo_change(self):
        vm = PianoRollViewModel()
        result = vm.compute_tempo_change(140)
        assert result is not None
        old_bw, new_bw = result
        assert vm.tempo == 140.0
        assert old_bw > new_bw

    def test_compute_tempo_no_change(self):
        vm = PianoRollViewModel()
        result = vm.compute_tempo_change(120)
        assert result is None

    def test_compute_tempo_invalid(self):
        vm = PianoRollViewModel()
        assert vm.compute_tempo_change(0) is None
        assert vm.compute_tempo_change(-10) is None

    def test_pitch_info_root(self):
        vm = PianoRollViewModel()
        vm.set_scale("C", "Major")
        info = vm.pitch_info(60)
        assert info["is_root"]
        assert info["in_scale"]
        assert "C" in info["name"]

    def test_pitch_info_non_root(self):
        vm = PianoRollViewModel()
        vm.set_scale("C", "Chromatic")
        info = vm.pitch_info(62)
        assert not info["is_root"]
        assert info["in_scale"]

    def test_time_to_x(self):
        vm = PianoRollViewModel()
        assert vm.time_to_x(1.0) == 100.0
        assert vm.time_to_x(2.5) == 250.0

    def test_x_to_time(self):
        vm = PianoRollViewModel()
        assert vm.x_to_time(100.0) == 1.0
        assert vm.x_to_time(250.0) == 2.5

    def test_snap_to_beat(self):
        vm = PianoRollViewModel()
        bw = vm.beat_width
        assert vm.snap_to_beat(bw * 1.3) == bw

    def test_pitch_from_y(self):
        vm = PianoRollViewModel()
        assert vm.pitch_from_y(0) == 127
        assert vm.pitch_from_y(127 * 20) == 0

    def test_y_from_pitch(self):
        vm = PianoRollViewModel()
        assert vm.y_from_pitch(127) == 0
        assert vm.y_from_pitch(0) == 127 * 20

    def test_playhead_x(self):
        vm = PianoRollViewModel()
        assert vm.playhead_x(1000) == 100.0
        assert vm.playhead_x(0) == 0.0


class TestSpectrogramViewModel:
    def test_defaults(self):
        vm = SpectrogramViewModel()
        assert vm.duration == 0.0
        assert not vm.loading
        assert not vm.has_image

    def test_set_loading_audio(self):
        vm = SpectrogramViewModel()
        assert vm.set_loading("test.wav")
        assert vm.loading

    def test_set_loading_midi(self):
        vm = SpectrogramViewModel()
        assert not vm.set_loading("test.mid")
        assert not vm.loading

    def test_on_ready(self):
        vm = SpectrogramViewModel()
        vm.set_loading("test.wav")
        vm.on_ready(True)
        assert not vm.loading
        assert vm.has_image

    def test_hz_to_y(self):
        vm = SpectrogramViewModel()
        y = vm.hz_to_y(440, 100)
        assert 0 < y < 100

    def test_hz_to_y_zero(self):
        vm = SpectrogramViewModel()
        assert vm.hz_to_y(0, 100) == 100

    def test_y_to_hz(self):
        vm = SpectrogramViewModel()
        freq = vm.y_to_hz(50, 100)
        assert freq > 0

    def test_hz_y_roundtrip(self):
        vm = SpectrogramViewModel()
        original_freq = 440
        y = vm.hz_to_y(original_freq, 1000)
        recovered = vm.y_to_hz(y, 1000)
        assert abs(recovered - original_freq) < 1.0

    def test_hover_info_no_image(self):
        vm = SpectrogramViewModel()
        assert vm.hover_info(50, 50, 100, 100) is None

    def test_hover_info_with_image(self):
        vm = SpectrogramViewModel()
        vm.has_image = True
        vm.duration = 10.0
        info = vm.hover_info(50, 50, 100, 100)
        assert info is not None
        assert "freq" in info
        assert "time_str" in info

    def test_hover_info_out_of_bounds(self):
        vm = SpectrogramViewModel()
        vm.has_image = True
        assert vm.hover_info(-1, 50, 100, 100) is None


class TestVisualizerViewModel:
    def test_defaults(self):
        vm = VisualizerViewModel()
        assert vm.bars_count == 32
        assert len(vm.values) == 32
        assert vm.current_energy == 0.0
        assert not vm.is_playing

    def test_set_analysis(self):
        vm = VisualizerViewModel()
        import numpy as np

        data = np.ones(100)
        vm.set_analysis(data, 5.0)
        assert vm.rms_data is not None
        assert vm.duration_sec == 5.0

    def test_clear_analysis(self):
        vm = VisualizerViewModel()
        import numpy as np

        vm.set_analysis(np.ones(100), 5.0)
        vm.clear_analysis()
        assert vm.rms_data is None
        assert vm.duration_sec == 0.0

    def test_set_position(self):
        vm = VisualizerViewModel()
        vm.set_position(5000)
        assert vm.current_time_sec == 5.0

    def test_tick_idle(self):
        vm = VisualizerViewModel()
        vm.tick()
        assert vm.current_energy < 0.01
        assert all(v < 0.01 for v in vm.values)

    def test_tick_playing(self):
        vm = VisualizerViewModel()
        import numpy as np

        vm.set_analysis(np.full(100, 0.8), 10.0)
        vm.is_playing = True
        vm.current_time_sec = 5.0
        for _ in range(20):
            vm.tick()
        assert vm.current_energy > 0

    def test_bar_color_idle(self):
        vm = VisualizerViewModel()
        r, g, b = vm.bar_color_rgb()
        assert r == 26
        assert g == 115
        assert b == 232

    def test_flash_alpha_zero(self):
        vm = VisualizerViewModel()
        assert vm.flash_alpha() == 0.0

    def test_custom_bars_count(self):
        vm = VisualizerViewModel(bars_count=16)
        assert vm.bars_count == 16
        assert len(vm.values) == 16
