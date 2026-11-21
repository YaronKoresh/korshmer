from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMenuBar, QPushButton

from korshmer import __version__

from ..styling import WINDOW_EDGE_WIDTH, WINDOW_RADIUS
from ..theming.color_utils import MAT_BORDER, MAT_HOVER, MAT_TEXT

FILE_MENU = (
    ("New Song", "new_project_action"),
    ("Open Song…", "load_project_action"),
    ("Save Song", "save_project_action"),
    ("Save Song As…", "save_project_as_action"),
    None,
    ("Save a Version", "create_snapshot_action"),
    ("Versions…", "restore_snapshot_action"),
    None,
    ("Import Audio Files…", "import_audio_files_action"),
    ("Export Many Files…", "batch_export_action"),
    None,
)
MORE_MENU = (
    ("Automatic Backups", "list_backups_action"),
    ("Save as Template", "save_template_action"),
    ("Check Song Files", "check_integrity_action"),
    ("Song Statistics", "project_stats_action"),
    ("Manage Samples", "manage_assets_action"),
)
HELP_MENU = (
    ("Find a Tool…", "focus_tool_finder"),
    ("Start Tutorial", "start_tutorial_action"),
    ("Keyboard Shortcuts", "_show_help"),
    ("Customise Shortcuts…", "customise_shortcuts_action"),
)


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

        def run(name):
            def call():
                action = getattr(parent, name, None)
                if action is not None:
                    action()

            return call

        def fill(menu, entries):
            for entry in entries:
                if entry is None:
                    menu.addSeparator()
                else:
                    menu.addAction(entry[0], run(entry[1]))

        file_menu = self.menu_bar.addMenu("File")
        fill(file_menu, FILE_MENU)
        more_menu = file_menu.addMenu("More")
        fill(more_menu, MORE_MENU)
        file_menu.addSeparator()
        file_menu.addAction("Exit", parent.close)

        help_menu = self.menu_bar.addMenu("Help")
        fill(help_menu, HELP_MENU)

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

        corner = max(0, WINDOW_RADIUS - WINDOW_EDGE_WIDTH)
        style_close = (
            btn_style_base
            + f"""
            QPushButton[framed="true"] {{
                border-top-right-radius: {corner}px;
            }}
            QPushButton:hover {{
                background-color: #ff0040;
                color: white;
            }}
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
        self.btn_close.setProperty("framed", True)
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
        else:
            self.parent_window.showMaximized()
        self.sync_max_button()

    def sync_max_button(self):
        window = self.parent_window
        self.btn_max.setText("❐" if window.isMaximized() else "▢")
        framed = not (window.isMaximized() or window.isFullScreen())
        if self.btn_close.property("framed") != framed:
            self.btn_close.setProperty("framed", framed)
            self.btn_close.style().unpolish(self.btn_close)
            self.btn_close.style().polish(self.btn_close)

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
                    from korshmer.core.window_initializer import (
                        drag_window_to,
                    )

                    drag_window_to(
                        self.parent_window, event.globalPosition().toPoint()
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
