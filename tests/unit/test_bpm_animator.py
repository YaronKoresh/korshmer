import subprocess
import sys

from PySide6.QtCore import QAbstractAnimation, QEasingCurve
from PySide6.QtWidgets import QSpinBox

from korshmer.ui.animations import BpmAnimator
from korshmer.ui.styling import BPM_SPINBOX_HIGHLIGHTED, BPM_SPINBOX_NORMAL


def test_import_does_not_load_main_module():
    code = (
        "import sys; import korshmer.ui.animations.bpm_animator; "
        "sys.exit(1 if 'korshmer.__main__' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False)
    assert result.returncode == 0


def test_flash_configures_animations(mock_qt_app):
    spin = QSpinBox()
    animator = BpmAnimator(spin)
    assert animator.highlight_animation is None
    assert animator.reset_animation is None

    animator.flash()

    highlight = animator.highlight_animation
    reset = animator.reset_animation
    assert highlight.targetObject() is spin
    assert bytes(highlight.propertyName()) == b"styleSheet"
    assert highlight.duration() == 150
    assert highlight.easingCurve().type() == QEasingCurve.OutCubic
    assert highlight.startValue() == BPM_SPINBOX_NORMAL
    assert highlight.endValue() == BPM_SPINBOX_HIGHLIGHTED
    assert highlight.state() == QAbstractAnimation.Running

    assert reset.targetObject() is spin
    assert bytes(reset.propertyName()) == b"styleSheet"
    assert reset.duration() == 300
    assert reset.easingCurve().type() == QEasingCurve.InCubic
    assert reset.startValue() == BPM_SPINBOX_HIGHLIGHTED
    assert reset.endValue() == BPM_SPINBOX_NORMAL
    assert reset.state() == QAbstractAnimation.Stopped


def test_flash_reuses_animations_and_restarts(mock_qt_app):
    spin = QSpinBox()
    animator = BpmAnimator(spin)
    animator.flash()
    highlight = animator.highlight_animation
    reset = animator.reset_animation

    animator._start_reset()
    assert reset.state() == QAbstractAnimation.Running

    animator.flash()
    assert animator.highlight_animation is highlight
    assert animator.reset_animation is reset
    assert highlight.state() == QAbstractAnimation.Running
    assert reset.state() == QAbstractAnimation.Stopped


def test_rapid_flashes_start_reset_once(mock_qt_app):
    spin = QSpinBox()
    animator = BpmAnimator(spin)
    for _ in range(5):
        animator.flash()

    reset_starts = []
    animator.reset_animation.stateChanged.connect(
        lambda new, old: (
            reset_starts.append(new)
            if new == QAbstractAnimation.Running
            else None
        )
    )
    animator.highlight_animation.setCurrentTime(150)

    assert len(reset_starts) == 1
    assert animator.reset_animation.state() == QAbstractAnimation.Running
