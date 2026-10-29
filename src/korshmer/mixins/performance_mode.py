from korshmer.ui.widgets.performance_panel import PerformancePanel

PERFORMANCE_TEMPO_STEP = 5


class PerformanceModeMixin:
    _performance_panel = None
    _performance_mode = False

    def is_performance_mode(self):
        return self._performance_mode

    def toggle_performance_mode(self):
        if self._performance_mode:
            self._performance_panel.hide()
            self.splitter.show()
            self._performance_mode = False
            self._announce_performance_mode("Performance mode off")
            return False

        if self._performance_panel is None:
            self._performance_panel = self._build_performance_panel()
        self.splitter.hide()
        self._performance_panel.show()
        self._performance_mode = True
        self._announce_performance_mode("Performance mode on")
        return True

    def _build_performance_panel(self):
        panel = PerformancePanel(self)
        panel.play_pause_clicked.connect(self._handle_play_pause)
        panel.stop_clicked.connect(self._performance_stop)
        panel.tempo_up_clicked.connect(
            lambda: self._performance_step_tempo(PERFORMANCE_TEMPO_STEP)
        )
        panel.tempo_down_clicked.connect(
            lambda: self._performance_step_tempo(-PERFORMANCE_TEMPO_STEP)
        )
        panel.previous_scene_clicked.connect(
            lambda: self._performance_move_scene("previous")
        )
        panel.next_scene_clicked.connect(
            lambda: self._performance_move_scene("next")
        )
        self.main_layout.insertWidget(1, panel, 1)
        panel.hide()
        return panel

    def _performance_stop(self):
        control_bar = getattr(self, "control_bar", None)
        if control_bar is not None:
            control_bar.stop_audio()

    def _performance_step_tempo(self, delta):
        spin = self.spin_bpm
        spin.setValue(spin.value() + delta)

    def _performance_move_scene(self, direction):
        setlist = getattr(self, "setlist", None)
        mover = getattr(setlist, direction, None)
        if mover is None:
            self._announce_performance_mode("No setlist loaded")
            return None
        return mover()

    def _announce_performance_mode(self, message):
        status_bar = getattr(self, "enhanced_status_bar", None)
        if status_bar is not None:
            status_bar.update_action(message)
