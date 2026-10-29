from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QInputDialog,
    QMessageBox,
)

from korshmer.midi_processing.score_notation import (
    Articulation,
    Clef,
    Dynamic,
    Ornament,
    ScoreNote,
    ScoreRenderer,
)
from korshmer.ui.viewmodels.score_notation_vm import ScoreNotationViewModel
from korshmer.ui.widgets.material import MAT_BG, MAT_PRIMARY, MAT_TEXT


class ScoreNoteItem(QGraphicsRectItem):
    def __init__(self, note, x_pos, y_pos, width=20, height=8):
        super().__init__(0, 0, width, height)
        self.setPos(x_pos, y_pos)
        self.score_note = note

        self.setBrush(QBrush(QColor(MAT_PRIMARY)))
        self.setPen(QPen(Qt.black, 1))

        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)

    def paint(self, painter, _option, widget=None):
        rect = self.rect()

        if self.isSelected():
            painter.setBrush(QBrush(QColor(MAT_PRIMARY).lighter(150)))
            painter.setPen(QPen(Qt.white, 2))
        else:
            painter.setBrush(self.brush())
            painter.setPen(self.pen())

        painter.drawEllipse(rect)

        if self.score_note.articulations:
            painter.setFont(QFont("Tahoma", 10))
            painter.setPen(QPen(Qt.white))
            y_offset = -12
            for articulation in self.score_note.articulations:
                symbol = self._get_articulation_symbol(articulation)
                painter.drawText(rect.x(), rect.y() + y_offset, symbol)
                y_offset -= 10

    def _get_articulation_symbol(self, articulation):
        symbols = {
            Articulation.STACCATO: "\u00b7",
            Articulation.ACCENT: ">",
            Articulation.TENUTO: "\u2500",
            Articulation.MARCATO: "^",
            Articulation.FERMATA: "\U0001d110",
        }
        return symbols.get(articulation, "")


class ScoreNotationWidget(QGraphicsView):
    note_selected = Signal(ScoreNote)
    note_modified = Signal(ScoreNote)

    def __init__(self, parent=None, context_menu_manager=None):
        super().__init__(parent)
        self.vm = ScoreNotationViewModel()

        self.context_menu_manager = context_menu_manager
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.note_items = []
        self.selected_notes = []

        self.renderer = ScoreRenderer()

        self.setMinimumHeight(300)
        self.setBackgroundBrush(QBrush(QColor(MAT_BG)))

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.TextAntialiasing)

        self._draw_initial_staff()

    @property
    def measures(self):
        return self.vm.measures

    @property
    def clef(self):
        return self.vm.clef

    @clef.setter
    def clef(self, value):
        self.vm.set_clef(value)

    def _draw_initial_staff(self):
        staff_line_y_positions = [100, 120, 140, 160, 180]
        staff_width = 2000

        for y in staff_line_y_positions:
            line = QGraphicsLineItem(0, y, staff_width, y)
            line.setPen(QPen(QColor(MAT_TEXT), 1))
            self.scene.addItem(line)

        clef_text = QGraphicsTextItem("\U0001d11e")
        clef_text.setDefaultTextColor(QColor(MAT_TEXT))
        clef_text.setFont(QFont("Tahoma", 60))
        clef_text.setPos(10, 80)
        self.scene.addItem(clef_text)

    def set_measures(self, measures):
        self.vm.set_measures(measures)
        self._render_measures()

    def _render_measures(self):
        self.scene.clear()
        self.note_items.clear()
        self._draw_initial_staff()

        x_offset = 100

        for measure in self.vm.measures:
            for note in measure.notes:
                x_pos, y_pos = self.vm.note_position(note, x_offset)
                note_item = ScoreNoteItem(note, x_pos, y_pos)
                self.scene.addItem(note_item)
                self.note_items.append(note_item)

            x_offset += 200

    def contextMenuEvent(self, event):
        if self.context_menu_manager is None:
            return

        self.selected_notes = [
            item.score_note
            for item in self.scene.selectedItems()
            if isinstance(item, ScoreNoteItem)
        ]

        menu = self.context_menu_manager.create_score_editor_menu(self)
        menu.exec(event.globalPos())

    def _require_selection(self, action_name):
        if not self.selected_notes:
            QMessageBox.warning(
                self, "No Selection", f"Please select notes to {action_name}."
            )
            return False
        return True

    def on_score_editor_transpose_up(self):
        if not self._require_selection("transpose"):
            return
        modified = self.vm.transpose_notes(self.selected_notes, 1)
        for note in modified:
            self.note_modified.emit(note)
        self._render_measures()

    def on_score_editor_transpose_down(self):
        if not self._require_selection("transpose"):
            return
        modified = self.vm.transpose_notes(self.selected_notes, -1)
        for note in modified:
            self.note_modified.emit(note)
        self._render_measures()

    def on_score_editor_add_articulation(self):
        if not self._require_selection("add articulation"):
            return

        articulations = [art.value for art in Articulation]
        selected, ok = QInputDialog.getItem(
            self,
            "Add Articulation",
            "Select articulation:",
            articulations,
            0,
            False,
        )

        if ok and selected:
            modified = self.vm.add_articulation(self.selected_notes, selected)
            for note in modified:
                self.note_modified.emit(note)
            self._render_measures()

    def on_score_editor_change_dynamic(self):
        if not self._require_selection("change dynamics"):
            return

        dynamics = [dyn.value for dyn in Dynamic]
        selected, ok = QInputDialog.getItem(
            self,
            "Change Dynamic",
            "Select dynamic marking:",
            dynamics,
            3,
            False,
        )

        if ok and selected:
            modified = self.vm.change_dynamic(self.selected_notes, selected)
            for note in modified:
                self.note_modified.emit(note)
            self._render_measures()

    def on_score_editor_add_ornament(self):
        if not self._require_selection("add ornament"):
            return

        ornaments = [orn.value for orn in Ornament]
        selected, ok = QInputDialog.getItem(
            self, "Add Ornament", "Select ornament:", ornaments, 0, False
        )

        if ok and selected:
            modified = self.vm.add_ornament(self.selected_notes, selected)
            for note in modified:
                self.note_modified.emit(note)
            self._render_measures()

    def on_score_editor_toggle_tie(self):
        if not self._require_selection("toggle tie"):
            return
        modified = self.vm.toggle_tie(self.selected_notes)
        for note in modified:
            self.note_modified.emit(note)
        self._render_measures()

    def on_score_editor_change_clef(self):
        clefs = [clef.value for clef in Clef]
        selected, ok = QInputDialog.getItem(
            self, "Change Clef", "Select clef:", clefs, 0, False
        )

        if ok and selected:
            self.vm.set_clef(selected)
            self.renderer.clef = self.vm.clef
            self._render_measures()
