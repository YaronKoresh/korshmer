import os
import secrets
import sys

from .base import BaseActionsMixin


class ModifiersActionsMixin(BaseActionsMixin):
    def run_ghosts(self):
        from korshmer.midi_processors import process_ghosts

        p = self.get_selected_midi()
        current_bpm = float(self.spin_bpm.value())
        if p:
            self._run_action(
                process_ghosts,
                p,
                0.3,
                current_bpm,
                success_msg="Ghost Notes Added",
            )

    def run_lofi(self):
        from korshmer.midi_processors import process_lofi

        p = self.get_selected_midi()
        if p:
            self._run_action(
                process_lofi, p, success_msg="Lo-Fi Velocity Applied"
            )

    def run_voice_leading(self):
        from korshmer.midi_processors import process_voice_leading

        p = self.get_selected_midi()
        if p:
            self._run_action(
                process_voice_leading, p, success_msg="Voice Leading Complete"
            )

    def run_lfo(self):
        from korshmer.midi_processors import process_lfo

        p = self.get_selected_midi()
        if p:
            f = self.slider_lfo_freq.value() / 10.0
            a = self.slider_lfo_amt.value()
            self._run_action(process_lfo, p, f, a, success_msg="LFO Complete")

    def run_strum(self):
        from korshmer.midi_processors import process_strum

        p = self.get_selected_midi()
        if p:
            c = self.combo_strum_curve.currentText()
            self._run_action(
                process_strum,
                p,
                0.04,
                "Down",
                c,
                success_msg="Strumming Complete",
            )

    def run_humanize_action(self):
        from korshmer.midi_processors import process_humanize

        path = self.get_selected_midi()
        if not path:
            return
        t = self.slider_human_time.value() / 100.0
        v = self.slider_human_vel.value()
        self._run_action(
            process_humanize, path, t, v, success_msg="Humanize Complete"
        )

    def run_snap_action(self):
        from korshmer.analysis import detect_midi_key
        from korshmer.midi_processors import process_scale_snap

        path = self.get_selected_midi()
        if not path:
            return
        root = self.combo_root.currentText()
        scale = self.combo_scale.currentText()
        if "Auto" in root or "Auto" in scale:
            r, s = detect_midi_key(path)
            if "Auto" in root:
                root = r
            if "Auto" in scale:
                scale = s

        grid_str = self.combo_grid.currentText()

        quant_strength = self.slider_quantize.value() / 100.0
        swing_val = self.slider_swing.value() / 100.0
        current_bpm = float(self.spin_bpm.value())

        self._run_action(
            process_scale_snap,
            path,
            root,
            scale,
            grid_str,
            quant_strength,
            swing_val,
            current_bpm,
            success_msg="Snap Complete",
        )

    def run_inversions_action(self):
        from korshmer.midi_processors import process_inversions

        path = self.get_selected_midi()
        if not path:
            return
        self._run_action(
            process_inversions, path, success_msg="Inversions Complete"
        )

    def run_length_action(self):
        from korshmer.midi_processors import process_legato_staccato

        path = self.get_selected_midi()
        if not path:
            return
        mode = self.combo_len_mode.currentText()
        val = 0.5 if mode == "Staccato" else 0.0
        self._run_action(
            process_legato_staccato,
            path,
            mode,
            val,
            success_msg=f"{mode} Complete",
        )

    def run_split_action(self):
        from korshmer.midi_processors import process_split_pitch

        path = self.get_selected_midi()
        if not path:
            return
        split_name = self.spin_split.currentText()
        octave = int(split_name[1])
        split_val = (octave + 1) * 12
        self._run_action(
            process_split_pitch,
            path,
            split_val,
            success_msg="Pitch Split Complete",
        )

    def run_double_action(self):
        from korshmer.midi_processors import process_doubling

        path = self.get_selected_midi()
        if not path:
            return
        oct_shift = -1 if "-1" in self.combo_oct.currentText() else 1
        self._run_action(
            process_doubling,
            path,
            oct_shift,
            success_msg="Octave Double Complete",
        )

    def run_vel_comp_action(self):
        from korshmer.midi_processors import process_velocity_compress

        path = self.get_selected_midi()
        if not path:
            return
        min_v = self.slider_vel_min.value()
        max_v = self.slider_vel_max.value()
        self._run_action(
            process_velocity_compress,
            path,
            min_v,
            max_v,
            success_msg="Velocity Compress Complete",
        )

    def run_remap_action(self):
        from korshmer.midi_processors import process_drum_remap

        path = self.get_selected_midi()
        if not path:
            return
        map_name = self.combo_drum_map.currentText()
        self._run_action(
            process_drum_remap,
            path,
            map_name,
            success_msg=f"Drum Remap to {map_name} Complete",
        )

    def run_sustain(self):
        from korshmer.midi_processors import process_sustain

        p = self.get_selected_midi()
        if p:
            self._run_action(
                process_sustain, p, success_msg="Sustain Pedal Complete"
            )

    def run_open_voice(self):
        from korshmer.midi_processors import process_open_voice

        p = self.get_selected_midi()
        if p:
            self._run_action(
                process_open_voice, p, success_msg="Open Voicing Complete"
            )

    def run_mono(self):
        from korshmer.midi_processors import process_mono

        p = self.get_selected_midi()
        if p:
            m = self.combo_mono.currentText()
            self._run_action(
                process_mono, p, m, success_msg=f"Force Mono ({m}) Complete"
            )

    def run_transpose(self):
        from korshmer.midi_processors import process_transpose

        p = self.get_selected_midi()
        if p:
            v = int(self.spin_trans.currentText())
            self._run_action(
                process_transpose,
                p,
                v,
                success_msg=f"Transpose by {v} Complete",
            )

    def run_polish_midi(self):
        from korshmer.midi_processors import process_polish

        p = self.get_selected_midi()
        if p:
            target = self.combo_polish_target.currentText()
            ms = int(self.combo_polish_ms.currentText().replace("ms", ""))
            self._run_action(
                process_polish,
                p,
                target,
                ms,
                success_msg=f"Polish ({target}) Complete",
            )

    def run_create_variation(self):
        from korshmer.midi_processors import process_variation

        p = self.get_selected_midi()
        if p:
            strength = self.combo_variation_strength.currentText()
            seed = secrets.randbelow(10**6)
            self._run_action(
                process_variation,
                p,
                strength,
                seed,
                success_msg=f"Variation ({strength}) Complete",
            )

    def run_cleanup(self):
        from korshmer.midi_processors import process_cleanup

        p = self.get_selected_midi()
        if p:
            ms = int(self.spin_ms.currentText().replace("ms", ""))
            self._run_action(
                process_cleanup, p, ms, success_msg="Cleanup Complete"
            )

    def run_crescendo(self):
        from korshmer.midi_processors import process_crescendo

        p = self.get_selected_midi()
        if p:
            v1_text = self.spin_v1.currentText()
            v2_text = self.spin_v2.currentText()
            map_v = {"pp": 30, "mp": 60, "p": 40, "f": 100, "ff": 127}
            v1_label = v1_text.split(" ")[0]
            v2_label = v2_text.split(" ")[0]
            v1 = map_v.get(v1_label, 40)
            v2 = map_v.get(v2_label, 100)
            self._run_action(
                process_crescendo, p, v1, v2, success_msg="Crescendo Complete"
            )

    def run_retrograde(self):
        from korshmer.midi_processors import process_retrograde

        p = self.get_selected_midi()
        if p:
            self._run_action(
                process_retrograde, p, success_msg="Retrograde Complete"
            )

    def run_mirror(self):
        from korshmer.midi_processors import process_mirror

        p = self.get_selected_midi()
        if p:
            self._run_action(
                process_mirror, p, 60, success_msg="Mirror Inversion Complete"
            )

    def run_drop2(self):
        from korshmer.midi_processors import process_drop2

        p = self.get_selected_midi()
        if p:
            self._run_action(
                process_drop2, p, success_msg="Drop 2 Voicing Complete"
            )

    def run_limit(self):
        from korshmer.midi_processors import process_limit_range

        p = self.get_selected_midi()
        if p:
            min_p_name = self.spin_r_min.currentText()
            max_p_name = self.spin_r_max.currentText()

            def name_to_midi(name):
                octave = int(name[1])
                return (octave + 1) * 12

            min_p = name_to_midi(min_p_name)
            max_p = name_to_midi(max_p_name)
            self._run_action(
                process_limit_range,
                p,
                min_p,
                max_p,
                success_msg="Pitch Range Limit Complete",
            )

    def run_rand_vel(self):
        from korshmer.midi_processors import process_rand_vel

        p = self.get_selected_midi()
        if p:
            self._run_action(
                process_rand_vel,
                p,
                60,
                110,
                success_msg="Randomize Velocity Complete",
            )

    def run_rhythm(self):
        from korshmer.midi_processors import process_rhythm

        p = self.get_selected_midi()
        if p:
            self._run_action(
                process_rhythm, p, success_msg="Extract Rhythm Complete"
            )

    def run_speed(self, ratio):
        from korshmer.midi_processors import process_speed

        p = self.get_selected_midi()
        if p:
            self._run_action(
                process_speed, p, ratio, success_msg="Time Stretch Complete"
            )

    def run_delay(self):
        from korshmer.midi_processors import process_midi_delay

        p = self.get_selected_midi()
        if p:
            self._run_action(
                process_midi_delay,
                p,
                0.3,
                0.6,
                success_msg="MIDI Delay Complete",
            )

    def run_thin(self):
        from korshmer.midi_processors import process_density

        p = self.get_selected_midi()
        if p:
            self._run_action(
                process_density, p, 0.4, success_msg="Thin Density Complete"
            )

    def run_remove_gaps(self):
        from korshmer.midi_processors import process_remove_silence

        p = self.get_selected_midi()
        if p:
            self._run_action(
                process_remove_silence,
                p,
                success_msg="Remove Empty Bars Complete",
            )

    def run_groove(self):
        from korshmer.midi_processors import process_groove

        p = self.get_selected_midi()
        if p:
            t = self.combo_groove.currentText()
            current_bpm = float(self.spin_bpm.value())
            self._run_action(
                process_groove,
                p,
                t,
                current_bpm,
                success_msg=f"Groove Apply ({t}) Complete",
            )

    def run_merge_glitch(self):
        from korshmer.midi_processors import process_merge_glitch

        p = self.get_selected_midi()
        if p:
            self._run_action(
                process_merge_glitch, p, success_msg="Merge Glitches Complete"
            )

    def run_inv_vel(self):
        from korshmer.midi_processors import process_invert_vel

        p = self.get_selected_midi()
        if p:
            self._run_action(
                process_invert_vel, p, success_msg="Invert Velocity Complete"
            )

    def run_smart_loop(self):
        from korshmer.midi_processors import process_smart_loop

        p = self.get_selected_midi()
        if p:
            b = int(self.combo_loop.currentText().split()[0])
            current_bpm = float(self.spin_bpm.value())
            self._run_action(
                process_smart_loop,
                p,
                b,
                current_bpm,
                success_msg=f"Smart Loop ({b} Bars) Complete",
            )

    def run_ratchet(self):
        from korshmer.midi_processors import process_ratchet

        p = self.get_selected_midi()
        if p:
            d = int(self.combo_ratchet.currentText().replace("x", ""))
            self._run_action(
                process_ratchet, p, d, success_msg=f"Ratchet (x{d}) Complete"
            )

    def run_detune(self):
        from korshmer.midi_processors import process_microtuning

        p = self.get_selected_midi()
        if p:
            self._run_action(
                process_microtuning,
                p,
                3000,
                success_msg="Microtuning/Detune Complete",
            )

    def run_vel_ramp(self):
        from korshmer.midi_processors import process_vel_ramp

        p = self.get_selected_midi()
        if p:
            self._run_action(
                process_vel_ramp,
                p,
                40,
                120,
                success_msg="Velocity Ramp Complete",
            )

    def run_chord_detection(self):
        from korshmer.midi_processors import process_chord_detection

        p = self.get_selected_midi()
        if p:

            def open_result(result):
                if result and sys.platform == "win32":
                    os.startfile(result)

            self._run_action(
                process_chord_detection,
                p,
                success_msg="Chord Detection Complete",
                on_done=open_result,
            )

    def run_advanced_quantize(self):
        from korshmer.midi_processors import process_advanced_quantize

        p = self.get_selected_midi()
        if p:
            self._run_action(
                process_advanced_quantize,
                p,
                success_msg="Advanced Quantization Complete",
            )

    def run_arpeggiator_adv(self):
        from korshmer.midi_processors import process_arpeggiator_advanced

        p = self.get_selected_midi()
        if p:
            self._run_action(
                process_arpeggiator_advanced,
                p,
                success_msg="Advanced Arpeggiator Complete",
            )

    def run_scale_detection(self):
        from korshmer.midi_processors import process_scale_detection

        p = self.get_selected_midi()
        if p:

            def open_result(result):
                if result and sys.platform == "win32":
                    os.startfile(result)

            self._run_action(
                process_scale_detection,
                p,
                success_msg="Scale Detection Complete",
                on_done=open_result,
            )
