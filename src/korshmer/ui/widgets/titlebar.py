from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMenuBar, QPushButton

from korshmer import __version__

from ..theming.color_utils import MAT_BORDER, MAT_HOVER, MAT_TEXT


class KorshmerTitleBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)

        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_TranslucentBackground)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 0, 0)
        layout.setSpacing(0)

        self.menu_bar = QMenuBar()
        self.menu_bar.installEventFilter(self)
        self.menu_bar.setStyleSheet(
            f"""
            QMenuBar {{
                background-color: transparent;
                color: {MAT_TEXT};
                border: none;
            }}
            QMenuBar::item {{
                background-color: transparent;
                padding: 4px 10px;
            }}
            QMenuBar::item:selected {{
                background-color: {MAT_HOVER};
            }}
            QMenu {{
                background-color: #f5f5f5;
                color: #111111;
                border: 1px solid {MAT_BORDER};
            }}
            QMenu::item {{
                padding: 6px 20px;
            }}
            QMenu::item:selected {{
                background-color: {MAT_HOVER};
                color: #111111;
            }}
        """
        )

        file_menu = self.menu_bar.addMenu("File")
        file_menu.addAction(
            "New Project",
            lambda: (
                parent.new_project_action()
                if hasattr(parent, "new_project_action")
                else None
            ),
        )
        file_menu.addAction(
            "Open Project",
            lambda: (
                parent.load_project_action()
                if hasattr(parent, "load_project_action")
                else None
            ),
        )
        file_menu.addAction(
            "Save Project",
            lambda: (
                parent.save_project_action()
                if hasattr(parent, "save_project_action")
                else None
            ),
        )
        file_menu.addAction(
            "Save Project As",
            lambda: (
                parent.save_project_as_action()
                if hasattr(parent, "save_project_as_action")
                else None
            ),
        )
        file_menu.addSeparator()
        file_menu.addAction(
            "Import Audio Files",
            lambda: (
                parent.import_audio_files_action()
                if hasattr(parent, "import_audio_files_action")
                else None
            ),
        )
        file_menu.addAction(
            "Batch Export",
            lambda: (
                parent.batch_export_action()
                if hasattr(parent, "batch_export_action")
                else None
            ),
        )
        file_menu.addSeparator()
        file_menu.addAction("Exit", parent.close)

        project_menu = self.menu_bar.addMenu("Project")
        project_menu.addAction(
            "Create Snapshot",
            lambda: (
                parent.create_snapshot_action()
                if hasattr(parent, "create_snapshot_action")
                else None
            ),
        )
        project_menu.addAction(
            "Restore Snapshot",
            lambda: (
                parent.restore_snapshot_action()
                if hasattr(parent, "restore_snapshot_action")
                else None
            ),
        )
        project_menu.addAction(
            "List Backups",
            lambda: (
                parent.list_backups_action()
                if hasattr(parent, "list_backups_action")
                else None
            ),
        )
        project_menu.addSeparator()
        project_menu.addAction(
            "Save as Template",
            lambda: (
                parent.save_template_action()
                if hasattr(parent, "save_template_action")
                else None
            ),
        )
        project_menu.addAction(
            "Migrate Project",
            lambda: (
                parent.migrate_project_action()
                if hasattr(parent, "migrate_project_action")
                else None
            ),
        )
        project_menu.addSeparator()
        project_menu.addAction(
            "Check Integrity",
            lambda: (
                parent.check_integrity_action()
                if hasattr(parent, "check_integrity_action")
                else None
            ),
        )
        project_menu.addAction(
            "Project Statistics",
            lambda: (
                parent.project_stats_action()
                if hasattr(parent, "project_stats_action")
                else None
            ),
        )
        project_menu.addSeparator()
        project_menu.addAction(
            "Manage Assets",
            lambda: (
                parent.manage_assets_action()
                if hasattr(parent, "manage_assets_action")
                else None
            ),
        )

        version_control_menu = self.menu_bar.addMenu("Version Control")
        version_control_menu.addAction(
            "Initialize Repository",
            lambda: (
                parent.version_control_init_action()
                if hasattr(parent, "version_control_init_action")
                else None
            ),
        )
        version_control_menu.addAction(
            "Commit Changes",
            lambda: (
                parent.version_control_commit_action()
                if hasattr(parent, "version_control_commit_action")
                else None
            ),
        )
        version_control_menu.addAction(
            "View History",
            lambda: (
                parent.version_control_history_action()
                if hasattr(parent, "version_control_history_action")
                else None
            ),
        )
        version_control_menu.addAction(
            "Status",
            lambda: (
                parent.version_control_status_action()
                if hasattr(parent, "version_control_status_action")
                else None
            ),
        )
        version_control_menu.addSeparator()
        version_control_menu.addAction(
            "Create Branch",
            lambda: (
                parent.version_control_create_branch_action()
                if hasattr(parent, "version_control_create_branch_action")
                else None
            ),
        )
        version_control_menu.addAction(
            "Checkout Branch",
            lambda: (
                parent.version_control_checkout_branch_action()
                if hasattr(parent, "version_control_checkout_branch_action")
                else None
            ),
        )

        help_menu = self.menu_bar.addMenu("Help")
        help_menu.addAction(
            "Start Tutorial",
            lambda: (
                parent.start_tutorial_action()
                if hasattr(parent, "start_tutorial_action")
                else None
            ),
        )
        help_menu.addAction(
            "Keyboard Shortcuts",
            lambda: (
                parent._show_help() if hasattr(parent, "_show_help") else None
            ),
        )

        layout.addWidget(self.menu_bar)

        self.lbl_title = QLabel(f"KORSHMER v{__version__}")
        self.lbl_title.setStyleSheet(f"color: {MAT_TEXT};")
        self.lbl_title.setAttribute(Qt.WA_TransparentForMouseEvents)

        layout.addWidget(self.lbl_title)
        layout.addStretch()

        btn_style_base = """
            QPushButton {
                background-color: transparent;
                border: none;
                color: #aaaaaa;
                font-family: 'Segoe UI Symbol';
                font-size: 14px;
                border-radius: 0px;
            }
        """

        style_min = (
            btn_style_base
            + """
            QPushButton:hover {
                background-color: #505050;
                color: #00FFFF;
            }
        """
        )

        style_max = (
            btn_style_base
            + """
            QPushButton:hover {
                background-color: #ffdb58;
                color: #000000;
            }
        """
        )

        style_close = (
            btn_style_base
            + """
            QPushButton:hover {
                background-color: #ff0040;
                color: white;
            }
        """
        )

        self.btn_min = QPushButton("－")
        self.btn_min.setFixedSize(46, 32)
        self.btn_min.setStyleSheet(style_min)
        self.btn_min.clicked.connect(parent.showMinimized)

        self.btn_max = QPushButton("▢")
        self.btn_max.setFixedSize(46, 32)
        self.btn_max.setStyleSheet(style_max)
        self.btn_max.clicked.connect(self.toggle_max)

        self.btn_close = QPushButton("✕")
        self.btn_close.setFixedSize(46, 32)
        self.btn_close.setStyleSheet(style_close)
        self.btn_close.clicked.connect(parent.close)

        layout.addWidget(self.btn_min)
        layout.addWidget(self.btn_max)
        layout.addWidget(self.btn_close)
        layout.setSpacing(0)

        self.parent_window = parent

    def toggle_max(self):
        if self.parent_window.isMaximized():
            self.parent_window.showNormal()
            self.btn_max.setText("▢")
        else:
            self.parent_window.showMaximized()
            self.btn_max.setText("❐")

    def eventFilter(self, obj, event):
        if obj == self.menu_bar:
            if event.type() == QEvent.MouseButtonPress:
                action = self.menu_bar.actionAt(event.pos())
                if action is None:
                    self.parent_window.moving_window = True
                    self.parent_window.drag_pos = (
                        event.globalPosition().toPoint()
                        - self.parent_window.frameGeometry().topLeft()
                    )
                    return True
            elif event.type() == QEvent.MouseMove:
                if (
                    hasattr(self.parent_window, "moving_window")
                    and self.parent_window.moving_window
                    and self.parent_window.drag_pos is not None
                ):
                    self.parent_window.move(
                        event.globalPosition().toPoint()
                        - self.parent_window.drag_pos
                    )
                    return True
            elif event.type() == QEvent.MouseButtonRelease:
                if (
                    hasattr(self.parent_window, "moving_window")
                    and self.parent_window.moving_window
                ):
                    self.parent_window.moving_window = False
                    self.parent_window.drag_pos = None
                    return True
        return super().eventFilter(obj, event)
