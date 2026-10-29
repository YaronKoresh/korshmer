import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from korshmer.event_bus import Event, publish
from korshmer.utils.filesystem import is_midi_path
from korshmer.visualization.spectrum import create_spectrum_visualization

from .base import BaseActionsMixin

try:
    from korshmer.constants import DATA_DIR
except ImportError:
    DATA_DIR = Path.home() / ".korshmer" / "data"

try:
    from korshmer.ui.widgets.dialogs.image_preview import ImagePreviewDialog
    from korshmer.ui.widgets.dialogs.range_eq import RangeEQDialog
    from korshmer.workers.chord_analysis import ChordAnalysisThread
except ImportError:
    ImagePreviewDialog = None
    RangeEQDialog = None
    ChordAnalysisThread = None


class AudioActionsMixin(BaseActionsMixin):
    def run_spectrum_action(self):
        path = self._get_selected_file(is_input=None)
        if not path:
            return

        if is_midi_path(path):
            QMessageBox.warning(
                self,
                "Error",
                "Frequency analysis works on Audio files, not MIDI.",
            )
            return

        publish(Event.STATUS_UPDATE, text="Generating Frequency Spectrum...")
        QApplication.processEvents()

        img_path = create_spectrum_visualization(path)

        if img_path:
            publish(Event.STATUS_UPDATE, text="Done.")
            dlg = ImagePreviewDialog(
                img_path,
                title=f"Frequency Analysis: {os.path.basename(path)}",
                parent=self,
            )
            dlg.exec()
            try:
                os.remove(img_path)
            except OSError:
                pass
        else:
            publish(Event.STATUS_UPDATE, text="Error.")
            QMessageBox.critical(
                self, "Error", "Failed to generate spectrum analysis."
            )

    def run_smart_mastering(self):
        from korshmer.audio_processors import process_smart_mastering_file

        path = self._get_selected_file(is_input=None)
        if not path:
            return

        self._run_action(
            process_smart_mastering_file,
            path,
            success_msg="Mastering Complete",
            is_audio=True,
        )

    def run_range_eq_action(self):
        from korshmer.audio_processors import process_range_eq_file

        path = self._get_selected_file(is_input=None)
        if not path:
            return
        dialog = RangeEQDialog(self)
        if dialog.exec():
            p = dialog.get_params()
            self._run_action(
                process_range_eq_file,
                path,
                p["freq_start"],
                p["freq_end"],
                p["gain"],
                p["peak_pos"],
                p["curve_l"],
                p["curve_r"],
                success_msg="Range EQ Complete",
                is_audio=True,
            )

    def open_controller_mapping_dialog(self):
        from korshmer.ui.widgets.dialogs.controller_mapping import (
            ControllerMappingDialog,
        )

        hotplug = getattr(self, "midi_hotplug", None)
        devices = hotplug.connected_devices if hotplug else None
        dialog = ControllerMappingDialog(devices, parent=self)
        dialog.exec()
        if hotplug:
            hotplug.reload_mappings()

    def generate_chords_action(self):
        audio_path = self._get_selected_file(is_input=None)
        if not audio_path:
            return

        mode = self.combo_chord_precision.currentText()

        publish(Event.STATUS_UPDATE, text="Analyzing chords... please wait")
        publish(Event.CONVERT_BUTTON_STATE, enabled=False)
        self.chord_thread = ChordAnalysisThread(audio_path, DATA_DIR, mode)
        self.chord_thread.finished_signal.connect(self.on_chord_sheet_ready)
        self.chord_thread.error_signal.connect(self.on_chord_error)
        self.chord_thread.start()

    def on_chord_sheet_ready(self, output_path):
        publish(Event.CONVERT_BUTTON_STATE, enabled=True)
        publish(Event.STATUS_UPDATE, text="Chord analysis complete")
        msg = QMessageBox(self)
        msg.setWindowTitle("Complete")
        msg.setText(f"Chord sheet file created successfully:\n{output_path}")
        msg.exec()
        if sys.platform == "win32":
            os.startfile(output_path)
        else:
            subprocess.call(["open", output_path])

    def on_chord_error(self, err_msg):
        publish(Event.CONVERT_BUTTON_STATE, enabled=True)
        publish(Event.STATUS_UPDATE, text="Analysis Error")
        QMessageBox.critical(self, "Error", f"Analysis failed:\n{err_msg}")

    def run_audio_pitch_action(self):
        from korshmer.audio_processors import process_audio_pitch

        path = self._get_selected_file(is_input=None)
        if not path:
            return
        steps = int(self.spin_pitch.currentText())
        self._run_action(
            process_audio_pitch,
            path,
            steps,
            success_msg="Pitch Shift Complete",
            is_audio=True,
        )

    def run_audio_gate(self):
        from korshmer.audio_processors import process_audio_gate

        path = self._get_selected_file(is_input=None)
        if not path:
            return
        db = float(self.spin_gate.currentText().replace("dB", ""))
        self._run_action(
            process_audio_gate,
            path,
            db,
            success_msg="Noise Gate Complete",
            is_audio=True,
        )

    def run_audio_norm(self):
        from korshmer.audio_processors import process_audio_norm

        path = self._get_selected_file(is_input=None)
        if not path:
            return
        self._run_action(
            process_audio_norm,
            path,
            success_msg="Normalization Complete",
            is_audio=True,
        )

    def run_audio_filter(self):
        from korshmer.audio_processors import process_audio_filter

        path = self._get_selected_file(is_input=None)
        if not path:
            return
        ftype = (
            "high" if "High" in self.combo_filt_type.currentText() else "low"
        )
        freq = int(self.combo_filt_hz.currentText().replace("Hz", ""))
        self._run_action(
            process_audio_filter,
            path,
            freq,
            ftype,
            success_msg="Filter Complete",
            is_audio=True,
        )

    def run_audio_reverse(self):
        from korshmer.audio_processors import process_audio_reverse

        path = self._get_selected_file(is_input=None)
        if not path:
            return
        self._run_action(
            process_audio_reverse,
            path,
            success_msg="Reverse Complete",
            is_audio=True,
        )

    def run_audio_mono(self):
        from korshmer.audio_processors import process_audio_mono

        path = self._get_selected_file(is_input=None)
        if not path:
            return
        self._run_action(
            process_audio_mono,
            path,
            success_msg="Mono Mix Complete",
            is_audio=True,
        )

    def run_transient(self):
        from korshmer.audio_processors import process_transient

        path = self._get_selected_file(is_input=None)
        if not path:
            return
        amt = self.slider_transient.value() / 10.0
        self._run_action(
            process_transient,
            path,
            amt,
            success_msg="Transient Shaping Complete",
            is_audio=True,
        )

    def run_slicer(self):
        from korshmer.audio_processors import process_slicer

        path = self._get_selected_file(is_input=None)
        if not path:
            return
        db = int(self.spin_slice_db.currentText().replace("dB", ""))

        def open_folder(files):
            if files and sys.platform == "win32":
                os.startfile(os.path.dirname(files[0]))

        self._run_action(
            process_slicer,
            path,
            db,
            success_msg="Audio Slicing Complete",
            on_done=open_folder,
            is_audio=True,
        )

    def run_parametric_eq(self):
        from korshmer.audio_processors import process_parametric_eq

        path = self._get_selected_file(is_input=None)
        if not path:
            return
        freq = float(self.spin_eq_freq.currentText().replace("Hz", ""))
        gain = float(self.slider_eq_gain.value())
        self._run_action(
            process_parametric_eq,
            path,
            freq,
            gain,
            success_msg="Parametric EQ Complete",
            is_audio=True,
        )

    def run_compressor(self):
        from korshmer.audio_processors import process_compressor

        path = self._get_selected_file(is_input=None)
        if not path:
            return
        self._run_action(
            process_compressor,
            path,
            success_msg="Compression Complete",
            is_audio=True,
        )

    def run_reverb(self):
        from korshmer.audio_processors import process_reverb

        path = self._get_selected_file(is_input=None)
        if not path:
            return
        self._run_action(
            process_reverb, path, success_msg="Reverb Complete", is_audio=True
        )

    def run_delay_fx(self):
        from korshmer.audio_processors import process_delay

        path = self._get_selected_file(is_input=None)
        if not path:
            return
        self._run_action(
            process_delay, path, success_msg="Delay Complete", is_audio=True
        )

    def run_chorus(self):
        from korshmer.audio_processors import process_chorus

        path = self._get_selected_file(is_input=None)
        if not path:
            return
        self._run_action(
            process_chorus, path, success_msg="Chorus Complete", is_audio=True
        )

    def run_limiter(self):
        from korshmer.audio_processors import process_limiter

        path = self._get_selected_file(is_input=None)
        if not path:
            return
        self._run_action(
            process_limiter, path, success_msg="Limiter Complete", is_audio=True
        )

    def run_audio_to_midi_conversion(self):
        path = self._get_selected_file(is_input=None)
        if not path:
            return

        if not path.lower().endswith(
            (".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif")
        ):
            QMessageBox.warning(
                self,
                "Error",
                "Please select an audio file (WAV, MP3, FLAC, OGG, or AIFF).",
            )
            return

        if getattr(self, "is_processing", False):
            QMessageBox.information(
                self,
                "Audio to MIDI",
                "A conversion is already running. Stop it first.",
            )
            return

        from korshmer.exceptions import AudioToMidiError

        try:
            from korshmer.audio_processing.ai_audio_to_midi import (
                EnhancedAudioToMIDI,
            )

            EnhancedAudioToMIDI()
        except (ImportError, AudioToMidiError) as e:
            QMessageBox.warning(
                self,
                "Audio to MIDI unavailable",
                f"Audio-to-MIDI feature cannot be used: {e}",
            )
            return

        if path not in getattr(self, "file_list_paths", []):
            self.add_file(path)
        self.start_conversion_logic(paths=[path])
