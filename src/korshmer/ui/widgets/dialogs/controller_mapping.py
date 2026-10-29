import logging

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
)

from korshmer.midi.controller_mapping import (
    ACTION_NAMES,
    DEFAULT_CONTROLLER_TYPE,
    ControllerMapping,
    detect_controller_type,
    list_input_device_names,
)
from korshmer.midi.mapping_store import ControllerMappingStore
from korshmer.ui.widgets.material.button import MaterialButton

GENERIC_DEVICE = "Generic controller"


LEARN_POLL_MS = 50


class ControllerMappingDialog(QDialog):
    def __init__(
        self,
        device_names=None,
        store: ControllerMappingStore | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Controller Mapping")
        self.resize(480, 520)
        self.store = store if store is not None else ControllerMappingStore()

        names = list(
            device_names
            if device_names is not None
            else list_input_device_names()
        )
        for saved in self.store.device_names():
            if saved not in names:
                names.append(saved)
        if not names:
            names = [GENERIC_DEVICE]
        self._mappings: dict[str, ControllerMapping] = {}
        self._learn_port = None
        self._learn_timer = QTimer(self)
        self._learn_timer.timeout.connect(self._read_learn_port)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Controller:"))
        self.combo_device = QComboBox()
        self.combo_device.addItems(names)
        layout.addWidget(self.combo_device)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["CC", "Action"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        self.label_status = QLabel("")
        layout.addWidget(self.label_status)

        row = QHBoxLayout()
        self.btn_add = MaterialButton("Add", role="tonal")
        self.btn_delete = MaterialButton("Delete", role="tonal")
        self.btn_learn = MaterialButton("Learn", role="tonal")
        self.btn_reset = MaterialButton("Reset", role="tonal")
        for button in (
            self.btn_add,
            self.btn_delete,
            self.btn_learn,
            self.btn_reset,
        ):
            row.addWidget(button)
        layout.addLayout(row)

        row = QHBoxLayout()
        self.btn_save = MaterialButton("Save", role="primary")
        self.btn_close = MaterialButton("Close", role="tonal")
        row.addWidget(self.btn_save)
        row.addWidget(self.btn_close)
        layout.addLayout(row)

        self.btn_add.clicked.connect(lambda: self.add_row())
        self.btn_delete.clicked.connect(self.delete_selected_row)
        self.btn_learn.clicked.connect(self.toggle_learn)
        self.btn_reset.clicked.connect(self.reset_to_defaults)
        self.btn_save.clicked.connect(self.save)
        self.btn_close.clicked.connect(self.reject)
        self.combo_device.currentTextChanged.connect(self._show_device)

        self._shown_device = None
        self._show_device(self.combo_device.currentText())

    @property
    def device_name(self) -> str:
        return self.combo_device.currentText()

    def mapping_for(self, device_name: str) -> ControllerMapping:
        mapping = self._mappings.get(device_name)
        if mapping is None:
            mapping = ControllerMapping(
                DEFAULT_CONTROLLER_TYPE
                if device_name == GENERIC_DEVICE
                else detect_controller_type(device_name)
            )
            self.store.apply(device_name, mapping)
            self._mappings[device_name] = mapping
        return mapping

    def _show_device(self, device_name: str) -> None:
        if self._shown_device is not None:
            self._commit_rows(self._shown_device, quiet=True)
        self._shown_device = device_name
        self._fill_table(self.mapping_for(device_name).cc_actions)

    def _fill_table(self, table: dict[int, str]) -> None:
        self.table.setRowCount(0)
        for cc, action in sorted(table.items()):
            self.add_row(cc, action)

    def add_row(self, cc: int = 0, action: str | None = None) -> None:
        index = self.table.rowCount()
        self.table.insertRow(index)
        spin = QSpinBox()
        spin.setRange(0, 127)
        spin.setValue(cc)
        combo = QComboBox()
        combo.addItems(ACTION_NAMES)
        if action in ACTION_NAMES:
            combo.setCurrentText(action)
        self.table.setCellWidget(index, 0, spin)
        self.table.setCellWidget(index, 1, combo)

    def delete_selected_row(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def rows(self) -> list[tuple[int, str]]:
        return [
            (
                self.table.cellWidget(r, 0).value(),
                self.table.cellWidget(r, 1).currentText(),
            )
            for r in range(self.table.rowCount())
        ]

    def _commit_rows(self, device_name: str, quiet: bool = False) -> bool:
        rows = self.rows()
        seen = set()
        for cc, _ in rows:
            if cc in seen:
                if not quiet:
                    self.label_status.setText(f"CC {cc} appears twice.")
                return False
            seen.add(cc)
        self.mapping_for(device_name).replace_cc_actions(dict(rows))
        return True

    def reset_to_defaults(self) -> None:
        mapping = self.mapping_for(self.device_name)
        mapping.reset_cc_actions()
        self._fill_table(mapping.cc_actions)
        self.label_status.setText("Built-in table restored (not saved yet).")

    def save(self) -> bool:
        name = self.device_name
        if not self._commit_rows(name):
            return False
        try:
            self.store.save(name, self.mapping_for(name))
        except OSError as e:
            logging.warning(f"Could not save controller mapping: {e}")
            QMessageBox.warning(
                self, "Controller Mapping", f"Save failed:\n{e}"
            )
            return False
        self.label_status.setText(f"Saved mapping for {name}.")
        return True

    def learn_cc(self, controller) -> bool:
        row = self.table.currentRow()
        if row < 0:
            row = 0 if self.table.rowCount() else -1
        if row < 0:
            self.add_row(int(controller))
            return True
        try:
            self.table.cellWidget(row, 0).setValue(int(controller))
        except (TypeError, ValueError):
            return False
        return True

    def toggle_learn(self) -> None:
        if self._learn_timer.isActive():
            self._stop_learn()
            return
        try:
            import mido

            self._learn_port = mido.open_input(
                None if self.device_name == GENERIC_DEVICE else self.device_name
            )
        except Exception:
            self.label_status.setText(
                "Learn is not available: no MIDI input could be opened."
            )
            return
        self._learn_timer.start(LEARN_POLL_MS)
        self.btn_learn.setText("Stop learning")
        self.label_status.setText("Move a control on the controller...")

    def _stop_learn(self) -> None:
        self._learn_timer.stop()
        port, self._learn_port = self._learn_port, None
        if port is not None:
            try:
                port.close()
            except Exception:
                logging.debug("closing the learn port failed", exc_info=True)
        self.btn_learn.setText("Learn")

    def _read_learn_port(self) -> None:
        port = self._learn_port
        if port is None:
            return
        for message in port.iter_pending():
            if message.type == "control_change":
                self.learn_cc(message.control)
                self._stop_learn()
                return

    def done(self, result):
        self._stop_learn()
        super().done(result)
