from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..theming.color_utils import (
    MAT_ACCENT,
    MAT_BG,
    MAT_BORDER,
    MAT_HOVER,
    MAT_TEXT,
)


class TutorialStep:
    def __init__(
        self,
        title: str,
        description: str,
        target_widget_name: str = None,
        highlight_rect: QRect = None,
    ):
        self.title = title
        self.description = description
        self.target_widget_name = target_widget_name
        self.highlight_rect = highlight_rect


class TutorialOverlay(QWidget):
    next_clicked = Signal()
    prev_clicked = Signal()
    close_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        self.highlight_rect = None
        self.current_step = None

        self.info_panel = QWidget(self)
        self.info_panel.setStyleSheet(
            f"""
            QWidget {{
                background-color: {MAT_BG};
                border: 2px solid {MAT_ACCENT};
                border-radius: 8px;
                padding: 10px;
            }}
            QLabel {{
                color: {MAT_TEXT};
                background: transparent;
                border: none;
            }}
            QPushButton {{
                background-color: {MAT_HOVER};
                color: {MAT_TEXT};
                border: 1px solid {MAT_BORDER};
                border-radius: 4px;
                padding: 8px 16px;
                min-width: 80px;
            }}
            QPushButton:hover {{
                background-color: {MAT_ACCENT};
            }}
            """
        )

        panel_layout = QVBoxLayout(self.info_panel)
        panel_layout.setContentsMargins(15, 15, 15, 15)
        panel_layout.setSpacing(10)

        self.title_label = QLabel()
        self.title_label.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: {MAT_ACCENT};"
        )
        self.title_label.setWordWrap(True)
        panel_layout.addWidget(self.title_label)

        self.desc_label = QLabel()
        self.desc_label.setStyleSheet("font-size: 14px;")
        self.desc_label.setWordWrap(True)
        self.desc_label.setMinimumHeight(60)
        panel_layout.addWidget(self.desc_label)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.prev_clicked.emit)

        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_clicked.emit)

        self.close_btn = QPushButton("Close Tutorial")
        self.close_btn.clicked.connect(self.close_clicked.emit)

        button_layout.addWidget(self.prev_btn)
        button_layout.addWidget(self.next_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.close_btn)

        panel_layout.addLayout(button_layout)

        self.info_panel.setMinimumSize(500, 200)

    def set_step(self, step: TutorialStep, step_num: int, total_steps: int):
        self.current_step = step
        self.highlight_rect = step.highlight_rect

        self.title_label.setText(f"Step {step_num}/{total_steps}: {step.title}")
        self.desc_label.setText(step.description)

        self.prev_btn.setEnabled(step_num > 1)
        self.next_btn.setText("Finish" if step_num == total_steps else "Next")

        self.update()
        self._position_info_panel()

    def _position_info_panel(self):
        if self.parent():
            parent_rect = self.parent().rect()
            panel_x = (parent_rect.width() - self.info_panel.width()) // 2
            panel_y = parent_rect.height() - self.info_panel.height() - 50
            self.info_panel.move(panel_x, panel_y)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.parent():
            self.setGeometry(self.parent().rect())
        self._position_info_panel()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        overlay_color = QColor(0, 0, 0, 180)
        painter.fillRect(self.rect(), overlay_color)

        if self.highlight_rect:
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.fillRect(self.highlight_rect, Qt.transparent)

            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            pen = QPen(QColor(MAT_ACCENT))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawRect(self.highlight_rect)


class TutorialManager:
    def __init__(self, parent_window):
        self.parent_window = parent_window
        self.tutorials = {}
        self.current_tutorial = None
        self.current_step_index = 0
        self.overlay = None

        self._init_default_tutorials()

    def _init_default_tutorials(self):

        intro_steps = [
            TutorialStep(
                title="Welcome to KORSHMER!",
                description=(
                    "Welcome to KORSHMER - your ultra-modern hybrid Audio Workstation! "
                    "This tutorial will guide you through the basic features. "
                    "Click 'Next' to continue."
                ),
            ),
            TutorialStep(
                title="The Main Interface",
                description=(
                    "KORSHMER features a custom hardware-inspired interface with Neon Blue aesthetics. "
                    "The top bar contains the main menu with File, Project, and Version Control options. "
                    "Take a moment to explore the clean, modern design."
                ),
            ),
            TutorialStep(
                title="Audio Files",
                description=(
                    "To get started, import audio files using File → Import Audio Files, "
                    "or drag and drop files directly into the file list area. "
                    "KORSHMER supports WAV, MP3, FLAC, and many other formats."
                ),
            ),
            TutorialStep(
                title="Audio Processing",
                description=(
                    "Once you've loaded an audio file, explore the Audio Tools tab "
                    "to access powerful effects like Smart Mastering, EQ, and more. "
                    "Click on any effect to apply it to your selected audio."
                ),
            ),
            TutorialStep(
                title="MIDI and Music Theory",
                description=(
                    "KORSHMER includes advanced MIDI processing and music theory tools. "
                    "Use the Theory tab to analyze chords, scales, and progressions. "
                    "The BPM control is your master tempo - it overrides everything!"
                ),
            ),
            TutorialStep(
                title="Generate A Song In One Click",
                description=(
                    "Open the EFFECTS tab and the MIDI Generator category, "
                    "pick the length, style, structure and roles, then press "
                    "'Generate → Editor'. Verses, choruses and breaks land "
                    "on separate lanes in the EDITOR, ready to tweak. Switch "
                    "the mode to 'Continue editor' to extend what you "
                    "already wrote."
                ),
            ),
            TutorialStep(
                title="Vocals On The Timeline",
                description=(
                    "Select a song and press 'Separate Stems'. When it "
                    "finishes, load the result into the editor: the vocals "
                    "become a draggable audio clip above the notes and the "
                    "instruments become MIDI lanes. Drag the clip to move it, "
                    "double-click to mute, and press Play Editor to hear "
                    "everything mixed together."
                ),
            ),
            TutorialStep(
                title="Keyboard Shortcuts",
                description=(
                    "Speed up your workflow with keyboard shortcuts:\n"
                    "• F1 - Show help\n"
                    "• Space - Play/Pause\n"
                    "• Ctrl+S - Save project\n"
                    "• Ctrl+B - Build song into the editor\n"
                    "• Ctrl+D / Q / U - Duplicate / Quantize / Humanize notes\n\n"
                    "Press F1 anytime to see the full list!"
                ),
            ),
            TutorialStep(
                title="Tutorial Complete!",
                description=(
                    "You've completed the introduction tutorial! "
                    "You can always restart this tutorial from Help → Start Tutorial. "
                    "Explore, experiment, and create amazing music with KORSHMER!"
                ),
            ),
        ]

        self.tutorials["intro"] = intro_steps

    def start_tutorial(self, tutorial_name="intro"):
        if tutorial_name not in self.tutorials:
            return

        self.current_tutorial = self.tutorials[tutorial_name]
        self.current_step_index = 0

        if not self.overlay:
            self.overlay = TutorialOverlay(self.parent_window)
            self.overlay.next_clicked.connect(self._next_step)
            self.overlay.prev_clicked.connect(self._prev_step)
            self.overlay.close_clicked.connect(self._close_tutorial)

        self._show_current_step()
        self.overlay.show()
        self.overlay.raise_()

    def _show_current_step(self):
        if not self.current_tutorial or self.current_step_index >= len(
            self.current_tutorial
        ):
            return

        step = self.current_tutorial[self.current_step_index]
        self.overlay.set_step(
            step, self.current_step_index + 1, len(self.current_tutorial)
        )

    def _next_step(self):
        self.current_step_index += 1

        if self.current_step_index >= len(self.current_tutorial):
            self._close_tutorial()
        else:
            self._show_current_step()

    def _prev_step(self):
        if self.current_step_index > 0:
            self.current_step_index -= 1
            self._show_current_step()

    def _close_tutorial(self):
        if self.overlay:
            self.overlay.hide()
        self.current_tutorial = None
        self.current_step_index = 0

    def is_tutorial_active(self):
        return self.overlay and self.overlay.isVisible()
