from PySide6.QtCore import QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QRegion
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..theming.color_utils import MAT_PRIMARY

PANEL_WIDTH = 520
PANEL_MARGIN = 24
HIGHLIGHT_PAD = 6
CHECK_MS = 250
DIM = QColor(32, 33, 36, 120)
INK = "#202124"
SOFT_INK = "#5F6368"
HAIRLINE = "#DADCE0"
DONE_GREEN = "#188038"
PANEL_STYLE = f"""
    QWidget#tutorialPanel {{
        background-color: white;
        border: 1px solid {HAIRLINE};
        border-radius: 10px;
    }}
    QLabel {{ background: transparent; border: none; color: {INK}; }}
    QPushButton {{
        background-color: white;
        color: {MAT_PRIMARY};
        border: 1px solid {HAIRLINE};
        border-radius: 6px;
        padding: 7px 16px;
        min-width: 72px;
        font-size: 12px;
    }}
    QPushButton:hover {{ background-color: #F1F3F4; }}
    QPushButton#tutorialNext {{
        background-color: {MAT_PRIMARY};
        color: white;
        border: 1px solid {MAT_PRIMARY};
    }}
    QPushButton#tutorialNext:disabled {{
        background-color: #E8EAED;
        color: #9AA0A6;
        border: 1px solid #E8EAED;
    }}
"""
TASK_WAITING = "Your turn: {task}"
TASK_DONE = "✓ Done. Press Next."


class TutorialStep:
    def __init__(
        self,
        title: str,
        description: str,
        target_widget_name: str = None,
        highlight_rect: QRect = None,
        task: str = "",
        begin=None,
        check=None,
    ):
        self.title = title
        self.description = description
        self.target_widget_name = target_widget_name
        self.highlight_rect = highlight_rect
        self.task = task
        self.begin = begin
        self.check = check
        self.memo = None

    @property
    def interactive(self) -> bool:
        return self.check is not None

    def start(self, window):
        self.memo = self.begin(window) if self.begin is not None else None

    def done(self, window) -> bool:
        if self.check is None:
            return True
        try:
            return bool(self.check(window, self.memo))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False


class TutorialOverlay(QWidget):
    next_clicked = Signal()
    prev_clicked = Signal()
    close_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.highlight_rect = None
        self.current_step = None

        self.info_panel = QWidget(self)
        self.info_panel.setObjectName("tutorialPanel")
        self.info_panel.setStyleSheet(PANEL_STYLE)
        self.info_panel.setFixedWidth(PANEL_WIDTH)

        panel_layout = QVBoxLayout(self.info_panel)
        panel_layout.setContentsMargins(20, 16, 20, 16)
        panel_layout.setSpacing(8)

        self.progress_label = QLabel()
        self.progress_label.setStyleSheet(
            f"font-size: 11px; color: {SOFT_INK};"
        )
        panel_layout.addWidget(self.progress_label)

        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.title_label.setWordWrap(True)
        panel_layout.addWidget(self.title_label)

        self.desc_label = QLabel()
        self.desc_label.setStyleSheet("font-size: 13px;")
        self.desc_label.setWordWrap(True)
        panel_layout.addWidget(self.desc_label)

        self.task_label = QLabel()
        self.task_label.setWordWrap(True)
        panel_layout.addWidget(self.task_label)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close_clicked.emit)
        self.prev_btn = QPushButton("Back")
        self.prev_btn.clicked.connect(self.prev_clicked.emit)
        self.next_btn = QPushButton("Next")
        self.next_btn.setObjectName("tutorialNext")
        self.next_btn.clicked.connect(self.next_clicked.emit)
        button_layout.addWidget(self.close_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.prev_btn)
        button_layout.addWidget(self.next_btn)
        panel_layout.addLayout(button_layout)

    def set_step(self, step: TutorialStep, step_num: int, total_steps: int):
        self.current_step = step
        self.progress_label.setText(f"Step {step_num} of {total_steps}")
        self.title_label.setText(step.title)
        self.desc_label.setText(step.description)
        self.prev_btn.setEnabled(step_num > 1)
        self.next_btn.setText("Finish" if step_num == total_steps else "Next")
        self.set_progress(not step.interactive)
        self.set_highlight(step.highlight_rect)

    def set_progress(self, done: bool):
        step = self.current_step
        self.next_btn.setEnabled(done)
        self.task_label.setVisible(step is not None and step.interactive)
        if step is None or not step.interactive:
            return
        if done:
            self.task_label.setText(TASK_DONE)
            self.task_label.setStyleSheet(
                f"font-size: 13px; font-weight: 600; color: {DONE_GREEN};"
            )
        else:
            self.task_label.setText(TASK_WAITING.format(task=step.task))
            self.task_label.setStyleSheet(
                f"font-size: 13px; font-weight: 600; color: {MAT_PRIMARY};"
            )
        self._fit_panel()

    def set_highlight(self, rect):
        self.highlight_rect = rect
        self._fit_panel()
        self._update_mask()
        self.update()

    def _fit_panel(self):
        layout = self.info_panel.layout()
        height = layout.heightForWidth(PANEL_WIDTH)
        if height <= 0:
            height = layout.sizeHint().height()
        self.info_panel.setFixedHeight(height)
        self._position_info_panel()

    def _position_info_panel(self):
        area = self.rect()
        panel = self.info_panel
        x = (area.width() - panel.width()) // 2
        y = area.height() - panel.height() - PANEL_MARGIN
        hole = self.highlight_rect
        if hole is not None and hole.intersects(
            QRect(x, y, panel.width(), panel.height())
        ):
            y = PANEL_MARGIN
            if hole.intersects(QRect(x, y, panel.width(), panel.height())):
                y = max(PANEL_MARGIN, hole.top() - panel.height() - 12)
        panel.move(max(0, x), max(0, y))

    def _update_mask(self):
        if self.highlight_rect is None:
            self.clearMask()
            return
        self.setMask(QRegion(self.rect()) - QRegion(self.highlight_rect))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_info_panel()
        self._update_mask()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), DIM)
        if self.highlight_rect is not None:
            pen = QPen(QColor(MAT_PRIMARY))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawRoundedRect(
                self.highlight_rect.adjusted(-2, -2, 2, 2), 6, 6
            )


def _tab_index(window, attribute):
    return window.tabs.indexOf(getattr(window, attribute))


def _on_tab(attribute):
    return lambda window, _memo: (
        window.tabs.currentIndex() == _tab_index(window, attribute)
    )


def _intro_steps():
    return [
        TutorialStep(
            "Welcome to KORSHMER",
            "This short tour shows the main places in the app. Most steps "
            "ask you to do one small thing yourself, so it sticks.",
        ),
        TutorialStep(
            "Your songs",
            "Songs you want to convert go in the top list. Everything the "
            "app makes for you appears under My music. Drag an audio file "
            "onto the top list, or use the File button.",
            target_widget_name="input_list",
        ),
        TutorialStep(
            "The editor",
            "Notes and audio clips live in the editor. Snap keeps what you "
            "move on the grid.",
            target_widget_name="arranger.btn_snap",
            task="click Snap to turn it off or on.",
            begin=lambda window: window.arranger.btn_snap.isChecked(),
            check=lambda window, was: (
                window.arranger.btn_snap.isChecked() != was
            ),
        ),
        TutorialStep(
            "Tempo and key",
            "The THEORY tab holds the tempo, the key and the groove of the "
            "song.",
            target_widget_name="tabs",
            task="open the THEORY tab.",
            check=lambda window, _memo: (
                window.tabs.tabText(window.tabs.currentIndex()) == "THEORY"
            ),
        ),
        TutorialStep(
            "Set the tempo",
            "The tempo box sets the speed of the whole song. You can type "
            "a number or use the arrows.",
            target_widget_name="spin_bpm",
            task="change the tempo to any other value.",
            begin=lambda window: window.spin_bpm.value(),
            check=lambda window, was: window.spin_bpm.value() != was,
        ),
        TutorialStep(
            "Effects and tools",
            "Every tool for notes and audio is in the EFFECTS tab.",
            target_widget_name="tabs",
            task="open the EFFECTS tab.",
            check=_on_tab("page_tools"),
        ),
        TutorialStep(
            "Find any tool",
            "You do not need to remember where a tool is. Type what you "
            "want to do, for example 'louder', 'swing' or 'echo'. "
            "Ctrl+K jumps here from anywhere.",
            target_widget_name="edit_tool_search",
            task="type at least three letters in the search box.",
            check=lambda window, _memo: (
                len(window.edit_tool_search.text().strip()) >= 3
            ),
        ),
        TutorialStep(
            "Play and compare",
            "The bar at the bottom plays your music. A/B switches between "
            "your version and the original song, and the Levels readout on "
            "the left shows loudness while it plays.",
            target_widget_name="control_bar",
        ),
        TutorialStep(
            "You are ready",
            "Convert a song with Start Processing, load the result into "
            "the editor and make it yours. You can restart this tour from "
            "Help → Start Tutorial.",
        ),
    ]


class TutorialManager:
    def __init__(self, parent_window):
        self.parent_window = parent_window
        self.tutorials = {"intro": _intro_steps()}
        self.current_tutorial = None
        self.current_step_index = 0
        self.overlay = None
        self.timer = QTimer()
        self.timer.setInterval(CHECK_MS)
        self.timer.timeout.connect(self._watch_step)

    def _target(self, name):
        target = self.parent_window
        for part in (name or "").split("."):
            target = getattr(target, part, None)
            if target is None:
                return None
        return target if name else None

    def _target_rect(self, step):
        if step.highlight_rect is not None and not step.target_widget_name:
            return step.highlight_rect
        widget = self._target(step.target_widget_name)
        if widget is None or not widget.isVisible():
            return None
        corner = widget.mapTo(self.parent_window, QPoint(0, 0))
        rect = QRect(corner, widget.size())
        return rect.adjusted(
            -HIGHLIGHT_PAD, -HIGHLIGHT_PAD, HIGHLIGHT_PAD, HIGHLIGHT_PAD
        ).intersected(self.parent_window.rect())

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
        self.overlay.setGeometry(self.parent_window.rect())
        self.overlay.show()
        self.overlay.raise_()
        self._show_current_step()
        self.timer.start()

    def current_step(self):
        if not self.current_tutorial:
            return None
        if self.current_step_index >= len(self.current_tutorial):
            return None
        return self.current_tutorial[self.current_step_index]

    def _show_current_step(self):
        step = self.current_step()
        if step is None:
            return
        step.start(self.parent_window)
        step.highlight_rect = self._target_rect(step)
        self.overlay.set_step(
            step, self.current_step_index + 1, len(self.current_tutorial)
        )

    def _watch_step(self):
        step = self.current_step()
        if step is None or self.overlay is None:
            return
        if self.overlay.size() != self.parent_window.size():
            self.overlay.setGeometry(self.parent_window.rect())
        rect = self._target_rect(step)
        if rect != self.overlay.highlight_rect:
            step.highlight_rect = rect
            self.overlay.set_highlight(rect)
        if step.interactive and not self.overlay.next_btn.isEnabled():
            if step.done(self.parent_window):
                self.overlay.set_progress(True)

    def _next_step(self):
        if self.overlay is not None and not self.overlay.next_btn.isEnabled():
            return
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
        self.timer.stop()
        if self.overlay:
            self.overlay.hide()
        self.current_tutorial = None
        self.current_step_index = 0

    def is_tutorial_active(self):
        return bool(self.overlay and self.overlay.isVisible())
