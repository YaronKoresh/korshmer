from PySide6.QtCore import QEasingCurve, QPropertyAnimation

from korshmer.ui.styling import BPM_SPINBOX_HIGHLIGHTED, BPM_SPINBOX_NORMAL


class BpmAnimator:
    def __init__(self, widget):
        self.widget = widget
        self.highlight_animation = None
        self.reset_animation = None

    def flash(self):
        if self.highlight_animation:
            self.highlight_animation.stop()
        if self.reset_animation:
            self.reset_animation.stop()

        if self.highlight_animation is None:
            self.highlight_animation = QPropertyAnimation(
                self.widget, b"styleSheet"
            )
            self.highlight_animation.setDuration(150)
            self.highlight_animation.setEasingCurve(QEasingCurve.OutCubic)
            self.highlight_animation.finished.connect(self._start_reset)
        self.highlight_animation.setStartValue(BPM_SPINBOX_NORMAL)
        self.highlight_animation.setEndValue(BPM_SPINBOX_HIGHLIGHTED)
        self.highlight_animation.start()

        if self.reset_animation is None:
            self.reset_animation = QPropertyAnimation(
                self.widget, b"styleSheet"
            )
            self.reset_animation.setDuration(300)
            self.reset_animation.setEasingCurve(QEasingCurve.InCubic)
        self.reset_animation.setStartValue(BPM_SPINBOX_HIGHLIGHTED)
        self.reset_animation.setEndValue(BPM_SPINBOX_NORMAL)

    def _start_reset(self):
        if self.reset_animation:
            self.reset_animation.start()
