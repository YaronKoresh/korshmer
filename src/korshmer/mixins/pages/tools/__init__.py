from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from korshmer.ui.theming.color_utils import MAT_PRIMARY
from korshmer.ui.tool_index import collect_tools, search_tools

from .toolbox import ToolboxMixin

__all__ = ["ToolsPageMixin", "ToolboxMixin"]

TOOL_SEARCH_HINT = "Find a tool…  (Ctrl+K)"
TOOL_SEARCH_HELP = (
    "Type what you want to do, for example 'transpose', 'swing' or "
    "'reverb'. Press Enter to open the first result."
)
NO_TOOL_FOUND = "No tool matches. Try another word."
TOOL_COLUMN_WIDTH = 240
INK = "#202124"
SOFT_INK = "#5F6368"
HAIRLINE = "#DADCE0"
RESULT_HEIGHT = 46
SEARCH_STYLE = f"""
    QLineEdit {{
        border: 1px solid {HAIRLINE};
        border-radius: 16px;
        padding: 6px 12px;
        font-size: 13px;
        background: white;
        color: {INK};
    }}
    QLineEdit:focus {{
        border: 2px solid {MAT_PRIMARY};
        padding: 5px 11px;
    }}
"""
RESULTS_STYLE = f"""
    QListWidget {{
        border: 1px solid {HAIRLINE};
        border-radius: 8px;
        background: white;
        outline: none;
    }}
    QListWidget::item {{
        border-bottom: 1px solid {HAIRLINE};
        padding: 0px;
    }}
    QListWidget::item:hover, QListWidget::item:selected {{
        background: #E8F0FE;
    }}
"""
RESULT_TEXT = (
    "<div style='font-size:13px; font-weight:600; color:{text};'>{name}</div>"
    "<div style='font-size:11px; color:{soft};'>{place}</div>"
)


def _result_label(name, place):
    label = QLabel(
        RESULT_TEXT.format(text=INK, soft=SOFT_INK, name=name, place=place)
    )
    label.setTextFormat(Qt.RichText)
    label.setContentsMargins(12, 4, 8, 4)
    label.setAttribute(Qt.WA_TransparentForMouseEvents)
    label.setStyleSheet("background: transparent;")
    return label


class ToolsPageMixin(ToolboxMixin):
    def _create_page_tools(self):
        page = QWidget()
        main_layout = QHBoxLayout(page)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        self.edit_tool_search = QLineEdit()
        self.edit_tool_search.setPlaceholderText(TOOL_SEARCH_HINT)
        self.edit_tool_search.setToolTip(TOOL_SEARCH_HELP)
        self.edit_tool_search.setClearButtonEnabled(True)
        self.edit_tool_search.setMinimumHeight(34)
        self.edit_tool_search.setStyleSheet(SEARCH_STYLE)
        self.list_tool_results = QListWidget()
        self.list_tool_results.setStyleSheet(RESULTS_STYLE)
        self.list_tool_results.setVisible(False)
        self.list_categories = QListWidget()
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
        self.tool_entries = collect_tools(categories)
        self.list_categories.setCurrentRow(0)
        self.list_categories.currentRowChanged.connect(
            self.stacked_content.setCurrentIndex
        )
        self.edit_tool_search.textChanged.connect(self._show_tool_results)
        self.edit_tool_search.returnPressed.connect(self._open_first_tool)
        self.list_tool_results.itemClicked.connect(self._open_tool_item)
        self.list_tool_results.itemActivated.connect(self._open_tool_item)

        column = QWidget()
        column.setFixedWidth(TOOL_COLUMN_WIDTH)
        column_layout = QVBoxLayout(column)
        column_layout.setContentsMargins(0, 0, 0, 0)
        column_layout.setSpacing(8)
        column_layout.addWidget(self.edit_tool_search)
        column_layout.addWidget(self.list_categories, 1)
        column_layout.addWidget(self.list_tool_results, 1)
        main_layout.addWidget(column)
        main_layout.addWidget(self.stacked_content)
        self.page_tools = page
        return page

    def _show_tool_results(self, text):
        found = search_tools(self.tool_entries, text)
        searching = bool(str(text).strip())
        self.list_tool_results.clear()
        for entry in found:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, self.tool_entries.index(entry))
            item.setData(Qt.AccessibleTextRole, entry.name)
            item.setToolTip(entry.help)
            item.setSizeHint(QSize(TOOL_COLUMN_WIDTH - 24, RESULT_HEIGHT))
            self.list_tool_results.addItem(item)
            place = entry.category
            if entry.group not in (entry.name, entry.category):
                place = f"{entry.category}  ›  {entry.group}"
            self.list_tool_results.setItemWidget(
                item, _result_label(entry.name, place)
            )
        if found:
            self.list_tool_results.setCurrentRow(0)
        if searching and not found:
            item = QListWidgetItem(NO_TOOL_FOUND)
            item.setFlags(Qt.NoItemFlags)
            self.list_tool_results.addItem(item)
        self.list_tool_results.setVisible(searching)
        self.list_categories.setVisible(not searching)

    def _open_first_tool(self):
        found = search_tools(self.tool_entries, self.edit_tool_search.text())
        if found:
            self.open_tool(found[0])

    def _open_tool_item(self, item):
        index = item.data(Qt.UserRole)
        if index is not None:
            self.open_tool(self.tool_entries[int(index)])

    def open_tool(self, entry):
        self.edit_tool_search.clear()
        self.list_categories.setCurrentRow(entry.page)
        card = entry.card
        if card is None:
            return
        card.set_mode(entry.name)
        page = self.stacked_content.widget(entry.page)
        if isinstance(page, QScrollArea):
            page.ensureWidgetVisible(card)
        card.button.setFocus()
        card.highlight()

    def focus_tool_finder(self):
        tabs = getattr(self, "tabs", None)
        if tabs is not None:
            tabs.setCurrentWidget(self.page_tools)
        self.edit_tool_search.setFocus()
        self.edit_tool_search.selectAll()
