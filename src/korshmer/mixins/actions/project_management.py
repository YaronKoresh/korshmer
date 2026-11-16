import os
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from korshmer import __version__
from korshmer.event_bus import Event, publish
from korshmer.project_management.project_manager import is_valid_new_name
from korshmer.project_management.validators.project_validator import (
    INVALID_NAME_MESSAGE,
)
from korshmer.utils.text_input import get_limited_text

SAVED_DATA_ONLY_MESSAGE = (
    "Files can only be saved within the 'saved_data' directory."
)


class ProjectManagementActionsMixin:
    def new_project_action(self):
        name, ok = get_limited_text(self, "New Project", "Project Name:")
        if ok and name:
            if not is_valid_new_name(name):
                QMessageBox.warning(
                    self, "Invalid Project Name", INVALID_NAME_MESSAGE
                )
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

            project = self.project_manager.create_project(name, template)
            if project is None:
                QMessageBox.warning(
                    self, "Invalid Project Name", INVALID_NAME_MESSAGE
                )
                return
            publish(Event.STATUS_UPDATE, text=f"Created new project: {name}")

    def save_project_action(self):
        if self.project_manager.current_project_path:
            if not self.project_manager.is_path_within_saved_data(
                self.project_manager.current_project_path
            ):
                QMessageBox.warning(
                    self, "Safety Error", SAVED_DATA_ONLY_MESSAGE
                )
                return
            success = self.project_manager.save_project()
            if success:
                publish(Event.STATUS_UPDATE, text="Project saved successfully")
            else:
                QMessageBox.warning(
                    self, "Save Failed", "Failed to save project"
                )
        else:
            self.save_project_as_action()

    def save_project_as_action(self):
        name = self.project_manager.current_project_data.get("name", "Untitled")
        default_path = self.project_manager.projects_dir / f"{name}.kproj"

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            str(default_path),
            "KORSHMER Project (*.kproj)",
        )

        if save_path:
            if not self.project_manager.is_path_within_saved_data(
                Path(save_path).with_suffix(".kproj")
            ):
                QMessageBox.warning(
                    self, "Safety Error", SAVED_DATA_ONLY_MESSAGE
                )
                return
            success = self.project_manager.save_project(Path(save_path))
            if success:
                publish(
                    Event.STATUS_UPDATE,
                    text=f"Project saved: {Path(save_path).name}",
                )
            else:
                QMessageBox.warning(
                    self, "Save Failed", "Failed to save project"
                )

    def load_project_action(self):
        project_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Project",
            str(self.project_manager.projects_dir),
            "KORSHMER Project (*.kproj *.json)",
        )

        if project_path:
            project_data = self.project_manager.load_project_with_fallback(
                Path(project_path)
            )
            if project_data:
                backup = self.project_manager.last_load_backup
                if backup is not None:
                    text = (
                        f"Loaded backup {backup.name} "
                        f"(original {Path(project_path).name} failed to load)"
                    )
                else:
                    text = f"Loaded project: {Path(project_path).name}"
                publish(Event.STATUS_UPDATE, text=text)
            else:
                QMessageBox.warning(
                    self, "Load Failed", "Failed to load project"
                )

    def create_snapshot_action(self):
        description, ok = get_limited_text(
            self, "Save a Version", "What changed? (optional):"
        )
        if ok:
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
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.aiff *.aif *.aac)",
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
            self.export_manager.process_queue()
            publish(
                Event.STATUS_UPDATE, text=f"Queued {count} files for export"
            )
        else:
            QMessageBox.warning(
                self, "Export Failed", "No files were queued for export"
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
        snapshots = self.project_manager.list_snapshots()
        if not snapshots:
            QMessageBox.information(
                self,
                "No Versions",
                "No versions yet. Use File → Save a Version first.",
            )
            return

        snapshot_names = [s.name for s in snapshots]
        snapshot_name, ok = QInputDialog.getItem(
            self,
            "Versions",
            "Go back to this version:",
            snapshot_names,
            0,
            False,
        )

        if ok and snapshot_name:
            snapshot_path = self.project_manager.snapshots_dir / snapshot_name
            success = self.project_manager.restore_snapshot(snapshot_path)
            if success:
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

    def migrate_project_action(self):
        if not self.project_manager.current_project_path:
            QMessageBox.warning(
                self, "No Project", "No project is currently open"
            )
            return

        reply = QMessageBox.question(
            self,
            "Migrate Project",
            "This will migrate the current project to version "
            f"{__version__}.\nA backup will be created. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            success = self.project_manager.migrate_project(
                self.project_manager.current_project_path, __version__
            )
            if success:
                QMessageBox.information(
                    self, "Migration Complete", "Project migrated successfully"
                )
            else:
                QMessageBox.warning(
                    self, "Migration Failed", "Failed to migrate project"
                )
