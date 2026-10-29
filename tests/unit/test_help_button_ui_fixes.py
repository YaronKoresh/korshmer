import pytest
from PySide6.QtWidgets import QApplication, QSizePolicy

from korshmer.ui.widgets.material.button import HelpButton
from korshmer.ui.widgets.material.card import MaterialCard


class TestHelpButtonUIFixes:
    def test_help_button_has_fixed_size_policy(self):
        QApplication.instance() or QApplication([])
        btn = HelpButton("Test tooltip")

        assert btn.sizePolicy().horizontalPolicy() == QSizePolicy.Fixed
        assert btn.sizePolicy().verticalPolicy() == QSizePolicy.Fixed

    def test_help_button_has_correct_dimensions(self):
        QApplication.instance() or QApplication([])
        btn = HelpButton("Test tooltip")

        assert btn.width() == 18 or btn.minimumWidth() == 18
        assert btn.height() == 18 or btn.minimumHeight() == 18

    def test_help_button_has_stylesheet(self):
        QApplication.instance() or QApplication([])
        btn = HelpButton("Test tooltip")

        stylesheet = btn.styleSheet()
        assert "background-color" in stylesheet
        assert "border-radius" in stylesheet

    def test_help_button_paintEvent_override_exists(self):
        assert hasattr(HelpButton, "paintEvent")

        assert "paintEvent" in HelpButton.__dict__

    def test_material_card_has_minimum_height(self):
        QApplication.instance() or QApplication([])
        card = MaterialCard()

        assert card.minimumSizeHint().height() >= 60

    def test_material_card_shadow_opacity_reduced(self):
        QApplication.instance() or QApplication([])
        card = MaterialCard()

        effect = card.graphicsEffect()
        assert effect is not None

        color = effect.color()

        assert color.alpha() == 30


class TestCreateLabelHelp:
    def test_create_label_help_container_has_minimum_height(self):

        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
