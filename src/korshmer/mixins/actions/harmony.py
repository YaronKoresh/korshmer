from .base import BaseActionsMixin


class ArpHarmonyActionsMixin(BaseActionsMixin):
    def run_arpeggiator_action(self):
        from korshmer.midi_processors import process_arpeggiator

        path = self.get_selected_midi()
        if not path:
            return
        pattern = self.combo_arp_pattern.currentText()
        rate = self.combo_arp_rate.currentText()
        current_bpm = float(self.spin_bpm.value())
        self._run_action(
            process_arpeggiator,
            path,
            pattern,
            rate,
            current_bpm,
            success_msg="Arpeggiator Complete",
        )

    def run_harmony_action(self):
        from korshmer.analysis import detect_midi_key
        from korshmer.midi_processors import process_smart_harmony

        path = self.get_selected_midi()
        if not path:
            return
        root = self.combo_root.currentText()
        scale = self.combo_scale.currentText()
        h_type = self.combo_harm_type.currentText()
        if "Auto" in root or "Auto" in scale:
            r, s = detect_midi_key(path)
            if "Auto" in root:
                root = r
            if "Auto" in scale:
                scale = s
        self._run_action(
            process_smart_harmony,
            path,
            root,
            scale,
            h_type,
            success_msg=f"Harmony ({h_type}) Complete",
        )
