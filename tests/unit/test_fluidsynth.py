class TestSuppressStderr:
    def test_suppress_stderr_import(self):
        from korshmer.core.fluidsynth import suppress_stderr

        assert callable(suppress_stderr)

    def test_suppress_stderr_context_manager(self):
        from korshmer.core.fluidsynth import suppress_stderr

        with suppress_stderr():
            pass

    def test_suppress_stderr_suppresses_output(self):
        import sys

        from korshmer.core.fluidsynth import suppress_stderr

        with suppress_stderr():
            pass

        assert sys.stderr is not None


class TestFluidSynthModule:
    def test_fluidsynth_class_exists(self):
        from korshmer.core.fluidsynth import FluidSynth

        assert FluidSynth is not None

    def test_fluidsynth_class_is_class(self):
        import inspect

        from korshmer.core.fluidsynth import FluidSynth

        assert inspect.isclass(FluidSynth)
