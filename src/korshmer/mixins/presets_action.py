from korshmer.event_bus import Event, publish
from korshmer.services.preset_service import PresetService, validate_preset_name
from korshmer.utils.text_input import get_limited_text


class PresetsMixin:
    def load_presets(self):
        return PresetService.load_all()

    def save_presets(self):
        PresetService.save_all(self.presets)

    def apply_preset(self, name):
        if name not in self.presets:
            return
        p = self.presets[name]
        self.combo_octave.setCurrentText(p.get("octave", "Original"))
        self.combo_root.setCurrentText(p.get("root", "Auto"))
        self.combo_scale.setCurrentText(p.get("scale", "Auto (Detect)"))
        self.combo_grid.setCurrentText(p.get("grid", "1/8"))
        self.slider_quantize.setValue(p.get("quantize", 100))
        self.slider_swing.setValue(p.get("swing", 0))
        vm = PresetService.normalize_velocity_mode(p.get("vel_mode", "Fixed"))
        self.combo_velocity.setCurrentText(vm)
        self.slider_vel_val.setValue(p.get("vel_val", 80))
        self.rad_merged.setChecked(p.get("export", "Separate") == "Merged")
        self.rad_separate.setChecked(p.get("export", "Separate") != "Merged")
        self.chk_cc.setChecked(p.get("cc", True))
        self.chk_gm.setChecked(p.get("gm", True))
        self.chk_wav32.setChecked(p.get("wav32", True))
        stems = p.get("stems", [])
        self.btn_drums.setChecked("drums" in stems)
        self.btn_bass.setChecked("bass" in stems)
        self.btn_other.setChecked("other" in stems)
        self.btn_vocals.setChecked("vocals" in stems)
        self.btn_harmonic.setChecked("guitar" in stems or "piano" in stems)

    def _collect_stems(self):
        s = []
        if self.btn_drums.isChecked():
            s.append("drums")
        if self.btn_bass.isChecked():
            s.append("bass")
        if self.btn_other.isChecked():
            s.append("other")
        if self.btn_vocals.isChecked():
            s.append("vocals")
        if self.btn_harmonic.isChecked():
            s.append("guitar")
            s.append("piano")
        return s

    def save_preset_dialog(self):
        name, ok = get_limited_text(self, "Save Preset", "New Preset Name:")
        if not ok:
            return
        name = validate_preset_name(name)
        if name is None:
            publish(
                Event.STATUS_UPDATE,
                text="Preset not saved: name is blank or reserved ('Default')",
            )
            return
        data = PresetService.build_preset_data(
            octave=self.combo_octave.currentText(),
            root=self.combo_root.currentText(),
            scale=self.combo_scale.currentText(),
            grid=self.combo_grid.currentText(),
            quantize=self.slider_quantize.value(),
            swing=self.slider_swing.value(),
            vel_mode=PresetService.normalize_velocity_mode(
                self.combo_velocity.currentText()
            ),
            vel_val=self.slider_vel_val.value(),
            export_mode="Merged" if self.rad_merged.isChecked() else "Separate",
            cc=self.chk_cc.isChecked(),
            gm=self.chk_gm.isChecked(),
            wav32=self.chk_wav32.isChecked(),
            stems=self._collect_stems(),
        )
        self.presets[name] = data
        self.save_presets()
        self.combo_presets.blockSignals(True)
        self.combo_presets.clear()
        self.combo_presets.addItems(list(self.presets.keys()))
        self.combo_presets.setCurrentText(name)
        self.combo_presets.blockSignals(False)

    def apply_preset_default(self):
        defaults = PresetService.get_default()
        self.apply_preset_data(defaults)
        publish(Event.STATUS_UPDATE, text="Settings reset to default")

    def apply_preset_data(self, p):
        self.combo_octave.setCurrentText(p.get("octave", "Original"))
        self.combo_root.setCurrentText(p.get("root", "Auto"))
        self.combo_scale.setCurrentText(p.get("scale", "Auto (Detect)"))
        self.combo_grid.setCurrentText(p.get("grid", "1/8"))
        self.slider_quantize.setValue(p.get("quantize", 100))
        self.slider_swing.setValue(p.get("swing", 0))
        vm = PresetService.normalize_velocity_mode(p.get("vel_mode", "Fixed"))
        self.combo_velocity.setCurrentText(vm)
        self.slider_vel_val.setValue(p.get("vel_val", 80))
        self.rad_merged.setChecked(p.get("export", "Separate") == "Merged")
        self.rad_separate.setChecked(p.get("export", "Separate") != "Merged")
        self.chk_cc.setChecked(p.get("cc", True))
        self.chk_gm.setChecked(p.get("gm", True))
        self.chk_wav32.setChecked(p.get("wav32", True))
        stems = p.get("stems", [])
        self.btn_drums.setChecked("drums" in stems)
        self.btn_bass.setChecked("bass" in stems)
        self.btn_other.setChecked("other" in stems)
        self.btn_vocals.setChecked("vocals" in stems)
        self.btn_harmonic.setChecked("guitar" in stems or "piano" in stems)
