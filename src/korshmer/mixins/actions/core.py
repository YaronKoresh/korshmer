import html
import os
import subprocess
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from korshmer.constants import DATA_DIR, FLUID_DIR, INSTRUMENT_DB, SF3_PATH
from korshmer.event_bus import Event, publish
from korshmer.midi.utils import generate_piano_roll_image
from korshmer.mixins.path_guard import reject_unsafe_path
from korshmer.project_management.validators.project_validator import (
    resolve_inside,
)
from korshmer.services.file_management import FileManagementService
from korshmer.ui.widgets.dialogs.stats import StatsDialog
from korshmer.utils.filesystem import is_midi_path
from korshmer.utils.instrument_names import sanitize_instrument_name
from korshmer.utils.midi_loading import load_pretty_midi
from korshmer.utils.track_layout import instrument_name


def _build_instrument_report(path):
    pm = load_pretty_midi(path)
    inst_counts = {}
    fam_counts = {}
    total_insts = len(pm.instruments)

    for inst in pm.instruments:
        name = sanitize_instrument_name(inst.name) or instrument_name(
            inst.program, inst.is_drum
        )
        inst_counts[name] = inst_counts.get(name, 0) + 1

        family = "Unknown"
        if inst.is_drum:
            family = "Drums"
        else:
            found = False
            for cat, data in INSTRUMENT_DB.items():
                for _, _, _, prog in data:
                    if inst.program == prog:
                        family = cat.replace("_", " ").title()
                        found = True
                        break
                if found:
                    break
        fam_counts[family] = fam_counts.get(family, 0) + 1

    report = [f"<b>Total Instruments:</b> {total_insts}<hr>"]
    report.append("<b>By Family:</b>")
    for f, c in fam_counts.items():
        report.append(f"• {html.escape(str(f))}: {c}")

    report.append("<br><b>By Name:</b>")
    for n, c in inst_counts.items():
        report.append(f"• {html.escape(str(n))}: {c}")
    return "<br>".join(report)


class CoreActionsMixin:
    def export_audio_action(self):

        from korshmer.audio_processors import render_midi_to_wav

        midi_path = self.get_selected_midi()
        if not midi_path:
            return
        if not os.path.exists(SF3_PATH):
            QMessageBox.warning(
                self,
                "SoundFont File Missing",
                f"To convert MIDI to audio, a SoundFont file is required.\n"
                f"Place 'soundfont.sf3' in the folder:\n{FLUID_DIR}",
            )
            return
        default_name = os.path.splitext(os.path.basename(midi_path))[0] + ".wav"
        default_path = os.path.join(DATA_DIR, default_name)
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Audio File", default_path, "WAV Audio (*.wav)"
        )
        if save_path:
            if reject_unsafe_path(self, save_path):
                return
            if resolve_inside(DATA_DIR, save_path) is None:
                QMessageBox.warning(
                    self,
                    "Safety Error",
                    "Files can only be saved within the 'saved_data' directory.",
                )
                return

            publish(
                Event.STATUS_UPDATE,
                text="Rendering to Audio (WAV), please wait...",
            )
            publish(Event.CONVERT_BUTTON_STATE, enabled=False)
            QApplication.processEvents()
            try:
                current_bpm = float(self.spin_bpm.value())
                render_midi_to_wav(
                    midi_path, save_path, FLUID_DIR, SF3_PATH, bpm=current_bpm
                )
                publish(
                    Event.STATUS_UPDATE, text="Export completed successfully"
                )
                QMessageBox.information(
                    self, "Success", f"File saved successfully:\n{save_path}"
                )
                parent_dir = os.path.dirname(save_path)
                if sys.platform == "win32":
                    os.startfile(parent_dir)
                else:
                    subprocess.call(["open", parent_dir])
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Conversion failed:\n{str(e)}"
                )
            finally:
                publish(Event.CONVERT_BUTTON_STATE, enabled=True)

    def run_json_export(self):

        from korshmer.midi_processors import process_json_export

        p = self.get_selected_midi()
        current_bpm = float(self.spin_bpm.value())
        if p:
            self._run_action(
                process_json_export,
                p,
                current_bpm,
                success_msg="JSON Export Complete",
            )

    def run_piano_roll_img(self):
        p = self.get_selected_midi()
        if p:
            default_name = (
                os.path.splitext(os.path.basename(p))[0] + "_PianoRoll.png"
            )
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Piano Roll Image", default_name, "PNG Image (*.png)"
            )
            if save_path:
                if reject_unsafe_path(self, save_path):
                    return
                result = generate_piano_roll_image(p, save_path)
                if result:
                    QMessageBox.information(
                        self, "Success", f"Image saved to:\n{result}"
                    )
                    if sys.platform == "win32":
                        os.startfile(result)
                    elif sys.platform == "darwin":
                        subprocess.call(["open", result])

    def run_stats(self):

        from korshmer.midi_processors import process_stats

        p = self.get_selected_midi()
        if p:

            def open_report(res):
                if res and sys.platform == "win32":
                    os.startfile(res)

            self._run_action(
                process_stats,
                p,
                success_msg="Scale Stats Report Complete",
                on_done=open_report,
            )

    def run_instrument_counter(self):
        path = self.get_selected_midi()
        if not path:
            return

        def show_report(report_html):
            StatsDialog(report_html, self).exec()

        def show_error(message):
            QMessageBox.critical(self, "Error", f"Failed to analyze: {message}")

        self._run_in_background(
            _build_instrument_report,
            path,
            on_result=show_report,
            on_error=show_error,
        )

    def run_batch_folder_action(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder for Processing"
        )
        if folder:
            valid_extensions = (
                ".mp3",
                ".wav",
                ".flac",
                ".ogg",
                ".aiff",
                ".aif",
            )
            count = 0
            for root, _dirs, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith(valid_extensions):
                        self.add_file(os.path.join(root, f))
                        count += 1
            if count > 0:
                publish(
                    Event.STATUS_UPDATE,
                    text=f"Loaded {count} files from folder",
                )
            else:
                QMessageBox.warning(
                    self,
                    "Empty",
                    "No audio files found in the selected folder.",
                )

    def toggle_ab_compare(self):
        item = self.output_list.currentItem()
        if not item:
            self.control_bar.btn_ab.setChecked(False)
            return

        midi_path = item.data(Qt.UserRole)
        if not is_midi_path(midi_path):
            self.control_bar.btn_ab.setChecked(False)
            return

        pos = self.control_bar.player.position()
        playing = self.control_bar.player.playbackState() == 1

        base_name = os.path.basename(midi_path)

        if self.control_bar.btn_ab.isChecked():
            source_path = FileManagementService.find_source_for_comparison(
                midi_path, self.file_list_paths
            )

            if source_path and os.path.exists(source_path):
                publish(
                    Event.AUDIO_LOAD_REQUESTED,
                    path=source_path,
                    analyze=True,
                    load_player=True,
                )
                self.control_bar.player.setPosition(pos)
                if playing:
                    self.control_bar.player.play()
                publish(
                    Event.STATUS_UPDATE,
                    text=f"Playing Source: {os.path.basename(source_path)}",
                )
            else:
                self.control_bar.btn_ab.setChecked(False)
                QMessageBox.warning(
                    self,
                    "A/B Error",
                    "Could not find original source file for comparison.",
                )
        else:
            wav_preview = os.path.join(
                os.path.dirname(midi_path), "temp_preview.wav"
            )
            target_file = (
                wav_preview if os.path.exists(wav_preview) else midi_path
            )

            publish(
                Event.AUDIO_LOAD_REQUESTED,
                path=target_file,
                analyze=True,
                load_player=True,
            )
            self.control_bar.player.setPosition(pos)
            if playing:
                self.control_bar.player.play()
            publish(Event.STATUS_UPDATE, text=f"Playing MIDI: {base_name}")

    def panic_action(self):
        self.control_bar.player.stop()
        if hasattr(self, "preview_thread") and self.preview_thread.isRunning():
            self.preview_thread.stop()
            self.preview_thread.wait()
        if hasattr(self, "worker") and self.worker.isRunning():
            self.worker.stop()
            self.worker.terminate()
            self.worker.wait()
        if hasattr(self, "chord_thread") and self.chord_thread.isRunning():
            self.chord_thread.terminate()
        self.is_processing = False
        if hasattr(self, "reset_ui_state"):
            self.reset_ui_state()
        publish(Event.STATUS_UPDATE, text="Emergency stop successful")
        QMessageBox.information(
            self, "Panic", "All processes and audio playback stopped."
        )
        publish(Event.CONVERT_BUTTON_STATE, enabled=bool(self.file_list_paths))
