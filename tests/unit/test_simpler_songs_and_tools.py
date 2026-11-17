import sys

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMenu,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

if not QApplication.instance():
    app = QApplication(sys.argv)

import korshmer.project_management as project_management
from korshmer.mixins.pages.tools import NO_TOOL_FOUND, ToolsPageMixin
from korshmer.services.file_management import FileManagementService
from korshmer.ui.tool_index import ToolEntry, collect_tools, search_tools
from korshmer.ui.widgets.material.adaptive import make_scrollable
from korshmer.ui.widgets.material.mode_tool import ModeToolCard, ToolMode
from korshmer.ui.widgets.titlebar import KorshmerTitleBar


def _page(*cards):
    page = QWidget()
    layout = QVBoxLayout(page)
    for card in cards:
        layout.addWidget(card)
    return make_scrollable(page)


class _Host(QWidget, ToolsPageMixin):
    def __init__(self):
        super().__init__()
        self.ran = []
        self.tabs = QTabWidget()
        self.tabs.addTab(QLabel("editor"), "EDITOR")
        self.tabs.addTab(self._create_page_tools(), "EFFECTS")

    def _card(self, title, *names, **keywords):
        modes = [
            ToolMode(
                name,
                lambda name=name: self.ran.append(name),
                (),
                f"Help for {name}",
                keywords.get(name, ""),
            )
            for name in names
        ]
        return ModeToolCard(title, modes)

    def _create_category_polish(self):
        return _page(self._card("Edit", "Transpose", "Snap to scale"))

    def _create_category_timing(self):
        return _page(self._card("Timing", "Groove", "Thin out"))

    def _create_category_voicing(self):
        return _page(self._card("Voicing", "Harmonize"))

    def _create_midi_generator_panel(self):
        return _page()

    def _create_category_dynamics(self):
        return _page(self._card("Dynamics", "Crescendo", Crescendo="bigger"))

    def _create_category_audio(self):
        return _page(self._card("Audio effect", "Reverb", "Transient shaper"))


def _results(host):
    rows = host.list_tool_results
    return [
        rows.item(i).data(Qt.AccessibleTextRole) or rows.item(i).text()
        for i in range(rows.count())
    ]


def test_every_tool_mode_is_indexed_with_its_place():
    host = _Host()
    names = [entry.name for entry in host.tool_entries]
    assert names == [
        "Transpose",
        "Snap to scale",
        "Groove",
        "Thin out",
        "Harmonize",
        "MIDI Generator",
        "Crescendo",
        "Reverb",
        "Transient shaper",
    ]
    groove = host.tool_entries[2]
    assert (groove.category, groove.group, groove.page) == (
        "Rhythm & Timing",
        "Timing",
        1,
    )
    assert groove.help == "Help for Groove"
    assert host.tool_entries[5].card is None


def test_search_ranks_names_before_keywords_and_help():
    host = _Host()
    found = search_tools(host.tool_entries, "tra")
    assert [entry.name for entry in found[:2]] == [
        "Transpose",
        "Transient shaper",
    ]
    assert [e.name for e in search_tools(host.tool_entries, "swing")] == [
        "Groove"
    ]
    assert [e.name for e in search_tools(host.tool_entries, "bigger")] == [
        "Crescendo"
    ]
    assert [e.name for e in search_tools(host.tool_entries, "voicing")] == [
        "Harmonize"
    ]
    assert (
        search_tools(host.tool_entries, "help for reverb")[0].name == "Reverb"
    )
    assert search_tools(host.tool_entries, "   ") == []
    assert search_tools(host.tool_entries, "zzzz") == []
    assert len(search_tools(host.tool_entries, "e", limit=2)) == 2
    plain = [ToolEntry("A", "A", "Echo box", "", 0)]
    assert search_tools(plain, "box")[0].label == "Echo box  ·  A"
    assert collect_tools([]) == []


def test_typing_shows_results_and_opening_one_goes_to_the_tool():
    host = _Host()
    assert host.list_tool_results.isHidden()
    host.edit_tool_search.setText("swing")
    assert _results(host) == ["Groove"]
    assert not host.list_tool_results.isHidden()
    assert host.list_categories.isHidden()

    host._open_tool_item(host.list_tool_results.item(0))
    card = host.tool_entries[2].card
    assert host.stacked_content.currentIndex() == 1
    assert host.list_categories.currentRow() == 1
    assert card.current_mode().label == "Groove"
    assert host.edit_tool_search.text() == ""
    assert not host.list_categories.isHidden()
    card.apply()
    assert host.ran == ["Groove"]

    host.edit_tool_search.setText("zzzz")
    assert _results(host) == [NO_TOOL_FOUND]
    assert host.list_tool_results.item(0).flags() == Qt.NoItemFlags
    host._open_first_tool()
    assert host.stacked_content.currentIndex() == 1

    host.edit_tool_search.setText("midi gen")
    host._open_first_tool()
    assert host.stacked_content.currentIndex() == 3


def test_the_finder_shortcut_target_brings_up_the_tools_page():
    host = _Host()
    host.tabs.setCurrentIndex(0)
    host.edit_tool_search.setText("old")
    host.focus_tool_finder()
    assert host.tabs.currentWidget() is host.page_tools
    assert host.edit_tool_search.selectedText() == "old"
    assert "Ctrl+K" in host.edit_tool_search.placeholderText()


class _Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.called = []

    def __getattr__(self, name):
        if name.endswith("_action") or name in (
            "focus_tool_finder",
            "_show_help",
        ):
            return lambda: self.called.append(name)
        raise AttributeError(name)


def test_the_menu_has_one_file_menu_with_versions_and_no_git():
    window = _Window()
    bar = KorshmerTitleBar(window)
    menus = {
        action.text(): action.menu()
        for action in bar.menu_bar.actions()
        if action.menu() is not None
    }
    assert list(menus) == ["File", "Help"]
    file_items = [a.text() for a in menus["File"].actions() if a.text()]
    assert file_items[:6] == [
        "New Song",
        "Open Song…",
        "Save Song",
        "Save Song As…",
        "Save a Version",
        "Versions…",
    ]
    everything = [
        a.text() for menu in bar.findChildren(QMenu) for a in menu.actions()
    ]
    for gone in ("Commit Changes", "Create Branch", "Initialize Repository"):
        assert gone not in everything
    by_text = {
        a.text(): a for menu in bar.findChildren(QMenu) for a in menu.actions()
    }
    for text in (
        "Save a Version",
        "Versions…",
        "Find a Tool…",
        "Song Statistics",
    ):
        by_text[text].trigger()
    assert window.called == [
        "create_snapshot_action",
        "restore_snapshot_action",
        "focus_tool_finder",
        "project_stats_action",
    ]


def test_git_version_control_is_gone_from_the_package():
    assert not hasattr(project_management, "VersionControlManager")
    with pytest.raises(ImportError):
        __import__("korshmer.project_management.version_control")


def test_the_music_list_hides_internal_folders_and_reads_like_a_musician(
    tmp_path,
):
    for name in (
        "papit",
        "exports",
        "Processed_MIDI",
        "snapshots",
        "backups",
        "projects",
        "tempo_aligned",
        "logs",
        "_temp_work",
    ):
        (tmp_path / name).mkdir()
    (tmp_path / "idea.mid").write_bytes(b"MThd")
    shown = {
        name for name, _ in FileManagementService.list_output_files(tmp_path)
    }
    assert shown == {"papit", "exports", "Processed_MIDI", "idea.mid"}

    name = FileManagementService.display_name
    assert name("2026_11_16__14_52_37__179484075721215") == "16 Nov 2026, 14:52"
    assert name("2026_11_16__14_52_37") == "16 Nov 2026, 14:52"
    assert name("2026_13_45__99_99_99") == "2026_13_45__99_99_99"
    assert name("Processed_MIDI") == "Edited MIDI"
    assert name("exports") == "Exported files"
    assert name("papit") == "papit"
    kept = FileManagementService.list_output_files(tmp_path, skip_names=set())
    assert "snapshots" in {n for n, _ in kept}
