import os
import subprocess
import sys
import textwrap

BROKEN_AUDIO_SYSTEM = textwrap.dedent(
    """
    class PortAudioError(Exception):
        pass

    raise PortAudioError(
        "Error initializing PortAudio: Unanticipated host error "
        "'PulseAudio_Initialize: Can't connect to server'"
    )
    """
)

CHECK = textwrap.dedent(
    """
    from korshmer.core import audio_engine, looper
    from korshmer.ui.widgets.arranger import timeline
    from korshmer.utils.audio_host import list_output_devices

    sd = audio_engine.sd
    assert timeline.sd is sd and looper is not None
    assert list_output_devices() == []
    for opening in (sd.OutputStream, sd.InputStream, sd.Stream, sd.query_devices):
        try:
            opening()
        except sd.PortAudioError as error:
            assert "PulseAudio_Initialize" in str(error)
        else:
            raise AssertionError("audio opened without an audio system")
    sd.default.reset()
    print("loaded")
    """
)


def test_the_app_loads_and_reports_audio_errors_when_the_audio_system_cannot_start(
    tmp_path,
):
    (tmp_path / "sounddevice.py").write_text(BROKEN_AUDIO_SYSTEM)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), *sys.path])
    env["QT_QPA_PLATFORM"] = "offscreen"

    run = subprocess.run(
        [sys.executable, "-c", CHECK],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip().endswith("loaded")
