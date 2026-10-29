from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QStackedWidget,
    QWidget,
)

from .toolbox import ToolboxMixin

__all__ = ["ToolsPageMixin", "ToolboxMixin"]


class ToolsPageMixin(ToolboxMixin):
    def _create_page_tools(self):
        page = QWidget()
        main_layout = QHBoxLayout(page)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        self.list_categories = QListWidget()
        self.list_categories.setFixedWidth(200)
        self.list_categories.setFocusPolicy(Qt.NoFocus)
        self.stacked_content = QStackedWidget()
        categories = [
            ("Polish & Analyze", self._create_category_polish()),
            ("Rhythm & Timing", self._create_category_timing()),
            ("Harmony & Voicing", self._create_category_voicing()),
            ("MIDI Generator", self._create_midi_generator_panel()),
            ("Dynamics", self._create_category_dynamics()),
            ("Audio Tools", self._create_category_audio()),
        ]
        for name, widget in categories:
            self.list_categories.addItem(name)
            self.stacked_content.addWidget(widget)
        self.list_categories.setCurrentRow(0)
        self.list_categories.currentRowChanged.connect(
            self.stacked_content.setCurrentIndex
        )
        main_layout.addWidget(self.list_categories)
        main_layout.addWidget(self.stacked_content)
        return page
