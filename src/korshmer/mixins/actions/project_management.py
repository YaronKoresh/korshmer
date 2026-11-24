import json
import os
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from korshmer.event_bus import Event, publish
from korshmer.mixins.files import audio_file_filter
from korshmer.project_management.project_backup import snapshot_choices
from korshmer.project_management.project_manager import (
    SONG_SUFFIX,
    is_valid_new_name,
)
from korshmer.project_management.song_folder import (
    conversion_kept_in_song,
    opened_editor,
    own_song_path,
    stored_editor,
)
from korshmer.project_management.validators.project_validator import (
    INVALID_NAME_MESSAGE,
)
from korshmer.utils.text_input import get_limited_text

SAVED_DATA_ONLY_MESSAGE = (
    "Files can only be saved within the 'saved_data' directory."
)


SONG_FILTER = "KORSHMER Project (*.kproj)"
UNSAVED_QUESTION = (
    "This song has changes that are not saved.\n\n"
    "Save keeps them. Discard closes without saving."
)
FINGERPRINT_KEYS = ("tempo", "time_signature", "tracks", "notes", "clips")
SAVED_EDITS_QUESTION = (
    "This song has saved edits.\n\n"
    "Yes opens your saved song. No loads the fresh conversion instead."
)


class ExportReport(QObject):
    job_done = Signal(object, bool)
    finished = Signal(int, int, list)

    def __init__(self, total, parent=None):
        super().__init__(parent)
        self.total = total
        self.done = 0
        self.problems = []
        self.job_done.connect(self.note)

    def note(self, job, success):
        if success:
            self.done += 1
        else:
            name = Path(str(job.get("source", ""))).name
            reason = job.get("error") or "the export failed"
            self.problems.append(f"{name}: {reason}")
        if self.done + len(self.problems) == self.total:
            self.finished.emit(self.done, self.total, self.problems)
            self.deleteLater()


def saved_song_in(folder):
    folder = Path(str(folder))
    if not folder.is_dir():
        return None
    songs = sorted(
        folder.glob(f"*{SONG_SUFFIX}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return songs[0] if songs else None


class ProjectManagementActionsMixin:
    def _song_home(self):
        session = getattr(self, "current_session", None)
        folder = str(getattr(session, "output_dir", "") or "")
        if not folder or not os.path.isdir(folder):
            return None
        if not self.project_manager.is_path_within_saved_data(folder):
            return None
        existing = saved_song_in(folder)
        if existing is not None:
            return existing
        name = Path(str(getattr(session, "source_path", ""))).stem
        if not is_valid_new_name(name):
            name = "song"
        return Path(folder) / f"{name}{SONG_SUFFIX}"

    def _editor_fingerprint(self):
        arranger = getattr(self, "arranger", None)
        if arranger is None or not hasattr(arranger, "to_project_dict"):
            return None
        state = arranger.to_project_dict()
        kept = {key: state.get(key) for key in FINGERPRINT_KEYS}
        return json.dumps(kept, sort_keys=True, default=str)

    def _mark_song_saved(self):
        self._saved_fingerprint = self._editor_fingerprint()

    def has_unsaved_changes(self):
        if not self._editor_has_content():
            return False
        saved = getattr(self, "_saved_fingerprint", None)
        return self._editor_fingerprint() != saved

    def confirm_unsaved_changes(self):
        if not self.has_unsaved_changes():
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved Song",
            UNSAVED_QUESTION,
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if answer == QMessageBox.Save:
            return bool(self.save_project_action())
        return answer == QMessageBox.Discard

    def _editor_has_content(self):
        model = getattr(getattr(self, "arranger", None), "model", None)
        return bool(model is not None and (model.notes or model.clips))

    def _capture_song(self, song_path):
        data = self.project_manager.current_project_data
        data.setdefault("name", Path(song_path).stem)
        arranger = getattr(self, "arranger", None)
        if arranger is not None and hasattr(arranger, "to_project_dict"):
            data["editor"] = stored_editor(
                arranger.to_project_dict(), Path(song_path).parent
            )
        session = getattr(self, "current_session", None)
        folder = str(getattr(session, "output_dir", "") or "")
        data["session_dir"] = (
            os.path.relpath(folder, Path(song_path).parent) if folder else ""
        )

    def _apply_song(self, data, song_path):
        from korshmer.audio_processing.stem_session import StemSession

        folder = Path(song_path).parent
        session_dir = folder / str((data or {}).get("session_dir") or "")
        self.current_session = (
            StemSession.find(session_dir)
            if self.project_manager.is_path_within_saved_data(session_dir)
            else None
        )
        editor = (data or {}).get("editor")
        arranger = getattr(self, "arranger", None)
        if arranger is None or not isinstance(editor, dict):
            return False
        arranger.load_project_dict(opened_editor(editor, folder))
        if hasattr(arranger, "set_song_folder"):
            arranger.set_song_folder(
                folder
                if song_path == self.project_manager.current_project_path
                else None
            )
        self._mark_song_saved()
        missing = (
            arranger.missing_clip_names()
            if hasattr(arranger, "missing_clip_names")
            else []
        )
        if missing:
            QMessageBox.warning(
                self,
                "Missing Audio",
                "The audio of these clips was not found in the song folder: "
                + ", ".join(missing),
            )
        if hasattr(self, "_show_editor_tab"):
            self._show_editor_tab()
        return True

    def new_project_action(self):
        name, ok = get_limited_text(self, "New Song", "Song name:")
        if ok and name:
            if self.project_manager.song_path_for(name) is None:
                QMessageBox.warning(
                    self, "Invalid Song Name", INVALID_NAME_MESSAGE
                )
                return
            if not self.confirm_unsaved_changes():
                return
            templates = self.project_manager.list_templates()
            template = None
            if templates:
                template, ok = QInputDialog.getItem(
                    self,
                    "Template",
                    "Select Template:",
                    ["blank"] + templates,
                    0,
                    False,
                )
                if not ok:
                    return

            project = self.project_manager.start_song(name, template)
            if project is None:
                QMessageBox.warning(
                    self, "Invalid Song Name", INVALID_NAME_MESSAGE
                )
                return
            self.current_session = None
            arranger = getattr(self, "arranger", None)
            if arranger is not None and hasattr(arranger, "clear_editor"):
                arranger.clear_editor()
            if arranger is not None and hasattr(arranger, "set_song_folder"):
                arranger.set_song_folder(
                    self.project_manager.song_path_for(name).parent
                )
            self._mark_song_saved()
            publish(
                Event.STATUS_UPDATE,
                text=f"New song: {name}. It is kept in the folder {name}.",
            )

    def _keep_conversion_in(self, song_folder):
        from korshmer.audio_processing.stem_session import StemSession

        session = getattr(self, "current_session", None)
        old_folder = str(getattr(session, "output_dir", "") or "")
        kept = conversion_kept_in_song(old_folder, song_folder)
        if kept is None:
            return
        arranger = getattr(self, "arranger", None)
        if arranger is not None and hasattr(arranger, "repoint_clips"):
            arranger.repoint_clips(old_folder, kept)
        self.current_session = StemSession.find(kept)

    def _save_song_to(self, song_path):
        song_path = own_song_path(
            song_path, self.project_manager.saved_data_dir, SONG_SUFFIX
        )
        if not self.project_manager.is_path_within_saved_data(song_path):
            QMessageBox.warning(self, "Safety Error", SAVED_DATA_ONLY_MESSAGE)
            return False
        arranger = getattr(self, "arranger", None)
        if arranger is not None and hasattr(arranger, "keep_clips_in"):
            try:
                song_path.parent.mkdir(parents=True, exist_ok=True)
                self._keep_conversion_in(song_path.parent)
                arranger.keep_clips_in(song_path.parent)
            except OSError as problem:
                QMessageBox.warning(
                    self,
                    "Save Failed",
                    f"The audio clips could not be copied into the song: {problem}",
                )
                return False
        self._capture_song(song_path)
        if not self.project_manager.save_project(song_path):
            QMessageBox.warning(self, "Save Failed", "Failed to save project")
            return False
        publish(
            Event.STATUS_UPDATE,
            text=f"Song saved: {song_path.name} in {song_path.parent.name}",
        )
        if hasattr(self, "refresh_lists"):
            self.refresh_lists()
        self._mark_song_saved()
        return True

    def save_project_action(self):
        song_path = (
            self.project_manager.current_project_path or self._song_home()
        )
        if song_path is None:
            return self.save_project_as_action()
        return self._save_song_to(song_path)

    def save_project_as_action(self):
        default_path = self._song_home()
        if default_path is None:
            name = self.project_manager.current_project_data.get(
                "name", "Untitled"
            )
            default_path = (
                self.project_manager.song_path_for(name)
                or self.project_manager.saved_data_dir / f"{name}{SONG_SUFFIX}"
            )

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            str(default_path),
            SONG_FILTER,
        )

        if not save_path:
            return False
        return self._save_song_to(save_path)

    def open_song_file(self, song_path):
        song_path = Path(song_path)
        project_data = self.project_manager.load_project_with_fallback(
            song_path
        )
        if not project_data:
            problem = getattr(self.project_manager, "last_load_error", "")
            QMessageBox.warning(
                self,
                "Load Failed",
                f"Failed to load project:\n{problem}"
                if problem
                else "Failed to load project",
            )
            return False
        backup = self.project_manager.last_load_backup
        if backup is not None:
            text = (
                f"Loaded backup {backup.name} "
                f"(original {song_path.name} failed to load)"
            )
        elif self._apply_song(project_data, song_path):
            text = f"Opened song: {song_path.stem}"
        else:
            text = f"Loaded project: {song_path.name}"
        publish(Event.STATUS_UPDATE, text=text)
        return True

    def load_project_action(self):
        if not self.confirm_unsaved_changes():
            return
        project_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Project",
            str(self.project_manager.saved_data_dir),
            "KORSHMER Project (*.kproj *.json)",
        )

        if project_path:
            self.open_song_file(project_path)

    def create_snapshot_action(self):
        if not self.project_manager.current_project_path:
            if not self.save_project_action():
                return
        description, ok = get_limited_text(
            self, "Save a Version", "What changed? (optional):"
        )
        if ok:
            self._capture_song(self.project_manager.current_project_path)
            success = self.project_manager.create_snapshot(description)
            if success:
                publish(Event.STATUS_UPDATE, text="Version saved")
            else:
                QMessageBox.warning(
                    self, "Version Not Saved", "The version could not be saved"
                )

    def save_template_action(self):
        name, ok = get_limited_text(self, "Save Template", "Template Name:")
        if ok and name:
            if not is_valid_new_name(name):
                QMessageBox.warning(
                    self, "Invalid Template Name", INVALID_NAME_MESSAGE
                )
                return
            success = self.project_manager.save_as_template(name)
            if success:
                publish(Event.STATUS_UPDATE, text=f"Template saved: {name}")
            else:
                QMessageBox.warning(
                    self, "Template Save Failed", "Failed to save template"
                )

    def import_audio_files_action(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Audio Files",
            str(self.import_manager.imports_dir),
            audio_file_filter("Audio Files", (".aac",)),
        )

        if file_paths:
            imported = self.import_manager.import_audio_files(
                [Path(p) for p in file_paths]
            )
            publish(Event.STATUS_UPDATE, text=f"Imported {len(imported)} files")

    def batch_export_action(self):
        if not hasattr(self, "output_list") or self.output_list.count() == 0:
            QMessageBox.warning(
                self, "No Files", "No files available for export"
            )
            return

        preset_names = self.export_manager.get_preset_names()
        preset, ok = QInputDialog.getItem(
            self,
            "Export Preset",
            "Select Export Preset:",
            preset_names,
            0,
            False,
        )
        if not ok:
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Directory"
        )
        if not output_dir:
            return

        source_files = []
        for i in range(self.output_list.count()):
            item = self.output_list.item(i)
            file_path = item.data(Qt.UserRole)
            if file_path and os.path.exists(file_path):
                source_files.append(Path(file_path))

        count = self.export_manager.batch_export(
            source_files, preset, Path(output_dir)
        )

        if count > 0:
            report = ExportReport(count, self)
            report.finished.connect(self._show_export_report)
            for job in list(self.export_manager.export_queue)[-count:]:
                if job.get("callback") is None:
                    job["callback"] = report.job_done.emit
            self.export_manager.process_queue()
            publish(
                Event.STATUS_UPDATE, text=f"Queued {count} files for export"
            )
        else:
            QMessageBox.warning(
                self, "Export Failed", "No files were queued for export"
            )

    def _show_export_report(self, done, total, problems):
        publish(Event.STATUS_UPDATE, text=f"Exported {done} of {total} files")
        if problems:
            QMessageBox.warning(
                self,
                "Export Failed",
                "These files could not be exported:\n" + "\n".join(problems),
            )

    def manage_assets_action(self):
        stats = self.asset_manager.get_asset_stats()
        msg = (
            f"Asset Library Statistics:\n\n"
            f"Total Assets: {stats['total_assets']}\n"
            f"Total Size: {stats['total_size'] / (1024 * 1024):.2f} MB\n"
            f"Favorites: {stats['favorites_count']}\n"
            f"Tags: {stats['total_tags']}\n\n"
            f"By Type:\n"
        )
        for asset_type, count in stats["type_counts"].items():
            msg += f"  {asset_type}: {count}\n"

        QMessageBox.information(self, "Asset Manager", msg)

    def restore_snapshot_action(self):
        song_path = self.project_manager.current_project_path
        snapshots = self.project_manager.list_snapshots(
            Path(song_path).stem if song_path else None
        )
        if not snapshots:
            QMessageBox.information(
                self,
                "No Versions",
                "No versions yet. Use File → Save a Version first.",
            )
            return

        choices = snapshot_choices(snapshots)
        snapshot_name, ok = QInputDialog.getItem(
            self,
            "Versions",
            "Go back to this version:",
            list(choices),
            0,
            False,
        )

        if ok and snapshot_name in choices:
            snapshot_path = choices[snapshot_name]
            song_path = self.project_manager.current_project_path
            success = self.project_manager.restore_snapshot(snapshot_path)
            if success:
                self.project_manager.current_project_path = song_path
                if song_path is not None:
                    self._apply_song(
                        self.project_manager.current_project_data, song_path
                    )
                publish(
                    Event.STATUS_UPDATE,
                    text=f"Went back to version: {snapshot_name}",
                )
            else:
                QMessageBox.warning(
                    self,
                    "Version Not Restored",
                    "That version could not be restored",
                )

    def list_backups_action(self):
        backups = self.project_manager.list_backups()
        if not backups:
            QMessageBox.information(self, "No Backups", "No backups available")
            return

        backup_names = []
        for b in backups:
            timestamp = datetime.fromtimestamp(b.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            backup_names.append(f"{b.name} ({timestamp})")

        msg = f"Available Backups ({len(backups)}):\n\n" + "\n".join(
            backup_names
        )
        QMessageBox.information(self, "Project Backups", msg)

    def check_integrity_action(self):
        if not self.project_manager.current_project_path:
            QMessageBox.warning(
                self, "No Project", "No project is currently open"
            )
            return

        is_valid, issues = self.project_manager.check_integrity(
            self.project_manager.current_project_path
        )

        if is_valid:
            QMessageBox.information(
                self, "Integrity Check", "Project integrity: OK"
            )
        else:
            msg = "Project integrity issues found:\n\n" + "\n".join(issues)
            QMessageBox.warning(self, "Integrity Issues", msg)

    def project_stats_action(self):
        stats = self.project_manager.get_project_stats()
        if not stats:
            QMessageBox.warning(
                self, "No Project", "No project is currently open"
            )
            return

        msg = (
            f"Project Statistics:\n\n"
            f"Name: {stats['name']}\n"
            f"Tracks: {stats['tracks']}\n"
            f"Markers: {stats['markers']}\n"
            f"Assets: {stats['assets']}\n"
            f"Created: {stats['created']}\n"
            f"Modified: {stats['modified']}\n"
            f"Version: {stats['version']}\n"
            f"Size: {stats['size'] / (1024 * 1024):.2f} MB\n"
        )

        QMessageBox.information(self, "Project Statistics", msg)
