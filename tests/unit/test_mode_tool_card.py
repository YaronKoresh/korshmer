import sys

import pytest
from PySide6.QtWidgets import QApplication, QComboBox, QLabel

if not QApplication.instance():
    QApplication(sys.argv)

from korshmer.ui.widgets.material.mode_tool import ModeToolCard, ToolMode


def _card(calls, single=False):
    shared = QComboBox()
    only_a = QLabel("a")
    only_b = QLabel("b")
    modes = [
        ToolMode(
            "Alpha",
            lambda: calls.append("alpha"),
            (shared, only_a),
            "alpha help",
        ),
        ToolMode(
            "Beta", lambda: calls.append("beta"), (shared, only_b), "beta help"
        ),
        ToolMode("Gamma", lambda: calls.append("gamma"), (), ""),
    ]
    if single:
        modes = modes[:1]
    card = ModeToolCard("Title", modes, help_text="base help")
    return card, shared, only_a, only_b


def test_the_apply_button_runs_the_chosen_mode():
    calls = []
    card, *_ = _card(calls)
    card.button.click()
    card.set_mode("Beta")
    card.button.click()
    card.set_mode("Gamma")
    card.button.click()
    assert calls == ["alpha", "beta", "gamma"]


def test_only_the_parameters_of_the_chosen_mode_are_shown():
    calls = []
    card, shared, only_a, only_b = _card(calls)
    assert not shared.isHidden() and not only_a.isHidden() and only_b.isHidden()
    card.set_mode("Beta")
    assert not shared.isHidden() and only_a.isHidden() and not only_b.isHidden()
    card.set_mode("Gamma")
    assert shared.isHidden() and only_a.isHidden() and only_b.isHidden()


def test_help_follows_the_mode_with_a_fallback():
    card, *_ = _card([])
    assert card.help_button.toolTip() == "alpha help"
    card.set_mode("Beta")
    assert card.help_button.toolTip() == "beta help"
    card.set_mode("Gamma")
    assert card.help_button.toolTip() == "base help"


def test_mode_labels_and_current_mode():
    card, *_ = _card([])
    assert card.mode_labels() == ["Alpha", "Beta", "Gamma"]
    assert card.current_mode().label == "Alpha"


def test_a_single_mode_hides_the_combo_box():
    card, *_ = _card([], single=True)
    assert card.combo_mode.isHidden()


def test_several_modes_show_the_combo_box():
    card, *_ = _card([])
    assert not card.combo_mode.isHidden()


def test_a_card_needs_modes_with_unique_labels():
    with pytest.raises(ValueError):
        ModeToolCard("t", [])
    with pytest.raises(ValueError):
        ModeToolCard(
            "t", [ToolMode("x", lambda: None), ToolMode("x", lambda: None)]
        )


def test_apply_returns_the_action_result():
    card = ModeToolCard("t", [ToolMode("x", lambda: 7)])
    assert card.apply() == 7
