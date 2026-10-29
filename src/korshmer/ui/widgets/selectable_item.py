from PySide6.QtWidgets import QWidget


class SelectableItem(QWidget):
    def __init__(self, list_widget, item_ref, parent=None):
        super().__init__(parent)
        self.list_widget = list_widget
        self.item_ref = item_ref

    def mousePressEvent(self, event):
        self.list_widget.setCurrentItem(self.item_ref)
        self.list_widget.itemPressed.emit(self.item_ref)
        self.list_widget.itemClicked.emit(self.item_ref)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.list_widget.itemDoubleClicked.emit(self.item_ref)
