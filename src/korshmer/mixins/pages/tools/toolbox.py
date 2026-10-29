from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from korshmer.constants import DRUM_MAPPINGS, HELP_TEXTS
from korshmer.midi.polish import TARGETS
from korshmer.midi.variation import STRENGTHS
from korshmer.ui.widgets.material.adaptive import make_scrollable
from korshmer.ui.widgets.material.button import MaterialButton
from korshmer.ui.widgets.material.card import MaterialCard
from korshmer.ui.widgets.material.mode_tool import ModeToolCard, ToolMode
from korshmer.ui.widgets.material.slider import MaterialSlider


def _combo(items, current=None, attr_owner=None, attr=None):
    box = QComboBox()
    box.addItems(list(items))
    if current is not None:
        box.setCurrentText(current)
    if attr_owner is not None:
        setattr(attr_owner, attr, box)
    return box


def _help(key):
    return HELP_TEXTS.get(key, "")


class ToolboxMixin:
    def _tool_page(self, *cards):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(10)
        for card in cards:
            layout.addWidget(card)
        layout.addStretch()
        return make_scrollable(page)

    def _create_category_polish(self):
        self.combo_polish_target = _combo(TARGETS, "Everything")
        self.combo_polish_ms = _combo(["30ms", "50ms", "100ms"], "50ms")
        polish = ModeToolCard(
            "Polish MIDI",
            [
                ToolMode(
                    "Polish",
                    self.run_polish_midi,
                    (
                        self.combo_polish_target,
                        self.combo_polish_ms,
                    ),
                    "Smooths the selected MIDI in one pass: removes glitches "
                    "and repeated notes, leads chord changes with the "
                    "nearest voicing and joins the notes of a lead. "
                    "Drums are never changed.",
                )
            ],
            button_text="Polish MIDI",
        )

        self.spin_ms = _combo(["30ms", "50ms", "100ms"], "50ms")
        clean = ModeToolCard(
            "Clean up",
            [
                ToolMode(
                    "Remove short notes",
                    self.run_cleanup,
                    (self.spin_ms,),
                    _help("clean"),
                ),
                ToolMode(
                    "Remove empty bars",
                    self.run_remove_gaps,
                    (),
                    "Closes silent gaps between bars.",
                ),
                ToolMode(
                    "Merge glitch notes",
                    self.run_merge_glitch,
                    (),
                    _help("glitch"),
                ),
            ],
        )

        self.spin_trans = _combo(
            [f"{i:+d}" for i in range(-12, 13) if i != 0], "+1"
        )
        self.spin_r_min = _combo(["C1", "C2", "C3"], "C2")
        self.spin_r_max = _combo(["C5", "C6", "C7"], "C6")
        self.combo_drum_map = _combo(list(DRUM_MAPPINGS.keys()))
        self.combo_variation_strength = _combo(list(STRENGTHS), "Medium")
        edit = ModeToolCard(
            "Edit",
            [
                ToolMode(
                    "Transpose",
                    self.run_transpose,
                    (self.spin_trans,),
                    _help("transpose"),
                ),
                ToolMode(
                    "Snap to scale",
                    self.run_snap_action,
                    (),
                    _help("snap"),
                ),
                ToolMode(
                    "Invert chords",
                    self.run_inversions_action,
                    (),
                    _help("inv"),
                ),
                ToolMode(
                    "Limit range",
                    self.run_limit,
                    (self.spin_r_min, self.spin_r_max),
                    _help("limit"),
                ),
                ToolMode(
                    "Remap drums",
                    self.run_remap_action,
                    (self.combo_drum_map,),
                    _help("remap"),
                ),
                ToolMode(
                    "Create variation",
                    self.run_create_variation,
                    (self.combo_variation_strength,),
                    "Saves a new MIDI that is built on the selected one: the "
                    "first notes of every second bar and the longest notes "
                    "stay, the rest moves along the key with passing notes, "
                    "shifted starts and new chord voicings. Drums are never "
                    "changed. Stronger settings change more.",
                ),
            ],
        )

        self.combo_chord_precision = _combo(
            ["Every Beat", "Half Bar", "1 Bar", "2 Bars"], "1 Bar"
        )
        analyze = ModeToolCard(
            "Analyze",
            [
                ToolMode("Statistics", self.run_stats),
                ToolMode("Rhythm", self.run_rhythm),
                ToolMode("Count instruments", self.run_instrument_counter),
                ToolMode(
                    "Detect chords",
                    self.run_chord_detection,
                    (),
                    "Detect chords in MIDI",
                ),
                ToolMode(
                    "Detect scales",
                    self.run_scale_detection,
                    (),
                    "Detect musical scales",
                ),
                ToolMode(
                    "Chord analysis (audio)",
                    self.generate_chords_action,
                    (self.combo_chord_precision,),
                ),
                ToolMode("Piano roll image", self.run_piano_roll_img),
                ToolMode("Frequency spectrum", self.run_spectrum_action),
            ],
            button_text="Run",
            role="tonal",
        )
        return self._tool_page(polish, clean, edit, analyze)

    def _create_category_timing(self):
        self.combo_loop = _combo(["4 Bars", "8 Bars"])
        self.combo_groove = _combo(
            ["MPC Swing", "Laid Back", "Rushing", "Drunken"], "MPC Swing"
        )
        self.combo_ratchet = _combo(["x2", "x3", "x4", "x8"], "x4")
        self.combo_strum_curve = _combo(["Linear", "Accelerate"], "Linear")
        timing = ModeToolCard(
            "Timing",
            [
                ToolMode(
                    "Smart loop",
                    self.run_smart_loop,
                    (self.combo_loop,),
                    _help("loop"),
                ),
                ToolMode(
                    "Speed x2 (faster)",
                    lambda: self.run_speed(0.5),
                    (),
                    _help("speed"),
                ),
                ToolMode(
                    "Speed x0.5 (slower)",
                    lambda: self.run_speed(2.0),
                    (),
                    _help("speed"),
                ),
                ToolMode(
                    "Groove",
                    self.run_groove,
                    (self.combo_groove,),
                    _help("groove"),
                ),
                ToolMode(
                    "Ratchet",
                    self.run_ratchet,
                    (self.combo_ratchet,),
                    _help("ratchet"),
                ),
                ToolMode(
                    "Advanced quantize",
                    self.run_advanced_quantize,
                    (),
                    "Grid quantization with beat-aware strength",
                ),
                ToolMode(
                    "Strum chords",
                    self.run_strum,
                    (self.combo_strum_curve,),
                    _help("strum"),
                ),
                ToolMode("MIDI delay", self.run_delay, (), _help("delay")),
                ToolMode("Thin out", self.run_thin, (), _help("thin")),
            ],
        )

        self.combo_arp_pattern = _combo(["Up", "Down", "Up-Down", "Random"])
        self.combo_arp_rate = _combo(["1/8", "1/16", "1/32", "1/4"], "1/16")
        arp = ModeToolCard(
            "Arpeggiator",
            [
                ToolMode(
                    "Arpeggiate",
                    self.run_arpeggiator_action,
                    (self.combo_arp_pattern, self.combo_arp_rate),
                    _help("arp"),
                ),
                ToolMode(
                    "Advanced arpeggio",
                    self.run_arpeggiator_adv,
                    (),
                    "Advanced arpeggiator",
                ),
            ],
        )
        return self._tool_page(timing, arp)

    def _create_category_voicing(self):
        self.combo_harm_type = _combo(
            ["Triads", "7ths", "Power (5th)", "Octaves"]
        )
        self.combo_mono = _combo(["High", "Low"], "High")
        self.spin_split = _combo(["C3", "C4", "C5"], "C4")
        self.combo_oct = _combo(["+1 Octave", "-1 Octave"], "-1 Octave")
        voicing = ModeToolCard(
            "Voicing",
            [
                ToolMode(
                    "Harmonize",
                    self.run_harmony_action,
                    (self.combo_harm_type,),
                    _help("harm"),
                ),
                ToolMode(
                    "Voice lead",
                    self.run_voice_leading,
                    (),
                    _help("voice_lead"),
                ),
                ToolMode(
                    "Open voicing", self.run_open_voice, (), _help("open")
                ),
                ToolMode("Drop 2", self.run_drop2, (), _help("drop2")),
                ToolMode(
                    "Force mono",
                    self.run_mono,
                    (self.combo_mono,),
                    _help("mono"),
                ),
                ToolMode(
                    "Split by pitch",
                    self.run_split_action,
                    (self.spin_split,),
                    _help("split"),
                ),
                ToolMode(
                    "Double octave",
                    self.run_double_action,
                    (self.combo_oct,),
                    _help("double"),
                ),
            ],
        )
        return self._tool_page(voicing)

    def _create_category_dynamics(self):
        self.slider_human_time = MaterialSlider(
            "Time Amt", _help("human"), max_val=20, default=5
        )
        self.slider_human_vel = MaterialSlider(
            "Vel Amt", _help("human"), max_val=20, default=10
        )
        self.spin_v1 = _combo(["p (40)", "mp (60)", "f (100)"], "p (40)")
        self.spin_v2 = _combo(["f (100)", "ff (127)", "p (40)"], "f (100)")
        self.slider_lfo_freq = MaterialSlider(
            "Freq (x0.1)", _help("lfo"), max_val=100, default=10
        )
        self.slider_lfo_amt = MaterialSlider(
            "Amount", _help("lfo"), max_val=63, default=20
        )
        dynamics = ModeToolCard(
            "Dynamics",
            [
                ToolMode(
                    "Humanize",
                    self.run_humanize_action,
                    (self.slider_human_time, self.slider_human_vel),
                    _help("human"),
                ),
                ToolMode("Velocity ramp", self.run_vel_ramp, (), _help("ramp")),
                ToolMode(
                    "Crescendo",
                    self.run_crescendo,
                    (self.spin_v1, self.spin_v2),
                    _help("cresc"),
                ),
                ToolMode(
                    "Invert velocity", self.run_inv_vel, (), _help("inv_vel")
                ),
                ToolMode("Lo-Fi velocity", self.run_lofi, (), _help("lofi")),
                ToolMode("Random velocity", self.run_rand_vel),
                ToolMode(
                    "LFO",
                    self.run_lfo,
                    (self.slider_lfo_freq, self.slider_lfo_amt),
                    _help("lfo"),
                ),
            ],
        )
        return self._tool_page(dynamics)

    def _create_category_audio(self):
        self.spin_eq_freq = _combo(
            ["100Hz", "500Hz", "1000Hz", "5000Hz", "10000Hz"], "1000Hz"
        )
        self.slider_eq_gain = MaterialSlider(
            "Gain", "EQ Gain in dB", min_val=-12, max_val=12, default=0
        )
        self.combo_filt_type = _combo(["High Pass", "Low Pass"], "High Pass")
        self.combo_filt_hz = _combo(
            ["100Hz", "500Hz", "1000Hz", "5000Hz"], "1000Hz"
        )
        self.spin_gate = _combo(["-20dB", "-30dB", "-40dB"], "-30dB")
        self.slider_transient = MaterialSlider(
            "Shaper", _help("transient"), max_val=90, default=10
        )
        effect = ModeToolCard(
            "Audio effect",
            [
                ToolMode(
                    "Parametric EQ",
                    self.run_parametric_eq,
                    (self.spin_eq_freq, self.slider_eq_gain),
                    "Frequency-specific gain control",
                ),
                ToolMode(
                    "Compressor",
                    self.run_compressor,
                    (),
                    "Dynamic range compression",
                ),
                ToolMode("Reverb", self.run_reverb, (), "Add room ambience"),
                ToolMode("Delay", self.run_delay_fx, (), "Echo effect"),
                ToolMode("Chorus", self.run_chorus, (), "Thickening effect"),
                ToolMode("Limiter", self.run_limiter, (), "Prevent clipping"),
                ToolMode(
                    "Filter",
                    self.run_audio_filter,
                    (self.combo_filt_type, self.combo_filt_hz),
                    _help("filter"),
                ),
                ToolMode(
                    "Gate",
                    self.run_audio_gate,
                    (self.spin_gate,),
                    _help("gate"),
                ),
                ToolMode(
                    "Transient shaper",
                    self.run_transient,
                    (self.slider_transient,),
                    _help("transient"),
                ),
            ],
        )

        self.spin_pitch = _combo(
            [f"{i:+d}" for i in range(-12, 13) if i != 0], "+1"
        )
        self.spin_slice_db = _combo(["15dB", "20dB", "25dB"], "20dB")
        fix = ModeToolCard(
            "Audio fix",
            [
                ToolMode("Normalize", self.run_audio_norm, (), _help("norm")),
                ToolMode(
                    "Pitch shift",
                    self.run_audio_pitch_action,
                    (self.spin_pitch,),
                    _help("pitch"),
                ),
                ToolMode(
                    "Mono mix", self.run_audio_mono, (), _help("mono_mix")
                ),
                ToolMode(
                    "Reverse", self.run_audio_reverse, (), _help("rev_aud")
                ),
                ToolMode(
                    "Slice by silence",
                    self.run_slicer,
                    (self.spin_slice_db,),
                    _help("slice"),
                ),
            ],
            role="tonal",
        )

        mastering = ModeToolCard(
            "Mastering",
            [
                ToolMode(
                    "Smart mastering",
                    self.run_smart_mastering,
                    (),
                    "Multiband compression & limiting",
                ),
                ToolMode(
                    "Range EQ",
                    self.run_range_eq_action,
                    (),
                    "Dynamic focused equalization",
                ),
            ],
        )

        models = MaterialCard()
        models_layout = QVBoxLayout(models)
        models_layout.setContentsMargins(10, 10, 10, 10)
        models_layout.addWidget(QLabel("<b>Model tools</b>"))
        row = QHBoxLayout()
        b_midi = MaterialButton("Audio to MIDI", role="primary")
        b_midi.setToolTip("Convert audio to MIDI with a neural model")
        b_midi.clicked.connect(self.run_audio_to_midi_conversion)
        b_stems = MaterialButton("Stem Separation", role="tonal")
        b_stems.setToolTip(_help("separate"))
        b_stems.clicked.connect(self.run_stem_separation)
        row.addWidget(b_midi)
        row.addWidget(b_stems)
        models_layout.addLayout(row)

        controller = MaterialButton("Controller Mapping…", role="tonal")
        controller.setToolTip("Assign MIDI CC numbers to actions")
        controller.clicked.connect(self.open_controller_mapping_dialog)
        return self._tool_page(effect, fix, mastering, models, controller)
