import contextlib
import logging
import os
import platform
import sys
from ctypes import CDLL, c_char_p, c_double, c_int, c_void_p
from ctypes.util import find_library
from pathlib import Path

import numpy as np

from korshmer.midi.message_validator import (
    validate_cc,
    validate_channel,
    validate_note_message,
    validate_pitch_bend,
)
from korshmer.utils.worker_count import compute_worker_count

_stderr_suppression_count = 0


@contextlib.contextmanager
def suppress_stderr():
    global _stderr_suppression_count

    should_suppress = False
    _stderr_suppression_count = getattr(suppress_stderr, "_count", 0)
    _stderr_suppression_count += 1
    suppress_stderr._count = _stderr_suppression_count
    should_suppress = _stderr_suppression_count == 1

    if not should_suppress:
        try:
            yield
        finally:
            _stderr_suppression_count -= 1
            suppress_stderr._count = _stderr_suppression_count
        return

    old_stderr = None
    devnull = None
    try:
        if sys.platform == "win32":
            null_file = "NUL"
        else:
            null_file = "/dev/null"

        try:
            stderr_fd = sys.stderr.fileno()
            old_stderr = os.dup(stderr_fd)
            devnull = open(null_file, "w")
            os.dup2(devnull.fileno(), stderr_fd)
        except (AttributeError, OSError) as e:
            logging.debug(f"Could not suppress stderr: {e}")

        yield

    finally:
        _stderr_suppression_count -= 1
        suppress_stderr._count = _stderr_suppression_count

        if old_stderr is not None:
            try:
                stderr_fd = sys.stderr.fileno()
                os.dup2(old_stderr, stderr_fd)
            except OSError:
                pass
            try:
                os.close(old_stderr)
            except OSError:
                pass

        if devnull is not None:
            try:
                devnull.close()
            except Exception:
                pass


class FluidSynth:
    def __init__(
        self,
        lib_dir: str | None = None,
        soundfont_path: str | None = None,
        driver: str | None = None,
        device: str | None = None,
        gain: float = 0.2,
        samplerate: int = 44100,
    ) -> None:
        self.settings = None
        self.synth = None
        self.audio_driver = None

        if soundfont_path:
            sf_path = Path(soundfont_path).resolve()
            if not sf_path.exists():
                raise FileNotFoundError(f"SoundFont file not found: {sf_path}")
            if not sf_path.is_file():
                raise FileNotFoundError(
                    f"SoundFont path is not a regular file: {sf_path}"
                )

        try:
            self._lib = self._load_library(lib_dir)
            self._set_prototypes()

            with suppress_stderr():
                self.settings = self._lib.new_fluid_settings()
                if not self.settings:
                    raise RuntimeError("Failed to create FluidSynth settings")

                self._lib.fluid_settings_setnum(
                    self.settings, b"synth.sample-rate", float(samplerate)
                )
                self._lib.fluid_settings_setnum(
                    self.settings, b"synth.gain", float(gain)
                )
                self._lib.fluid_settings_setint(
                    self.settings, b"synth.polyphony", 65535
                )
                self._lib.fluid_settings_setint(
                    self.settings,
                    b"synth.cpu-cores",
                    compute_worker_count("synth"),
                )

                self.synth = self._lib.new_fluid_synth(self.settings)
                if not self.synth:
                    raise RuntimeError(
                        "Failed to create FluidSynth synthesizer"
                    )

            if soundfont_path:
                if self.sf_load(soundfont_path) == -1:
                    raise RuntimeError(
                        f"Failed to load SoundFont: {soundfont_path}"
                    )
            if driver:
                self.start_audio(driver, device)
        except Exception as e:
            logging.error(f"FluidSynth initialization failed: {e}")
            raise

    def _load_library(self, user_lib_dir):
        logging.debug("Loading FluidSynth library")
        system = platform.system()
        lib_names = {
            "Windows": [
                "libfluidsynth-3.dll",
                "libfluidsynth-2.dll",
                "fluidsynth.dll",
            ],
            "Linux": [
                "libfluidsynth.so.3",
                "libfluidsynth.so.2",
                "libfluidsynth.so",
            ],
            "Darwin": ["libfluidsynth.3.dylib", "libfluidsynth.dylib"],
        }
        possible_names = lib_names.get(system, ["libfluidsynth.so"])

        if user_lib_dir:
            try:
                user_lib_dir = Path(user_lib_dir).resolve()
                if user_lib_dir.exists():
                    self._add_to_path(str(user_lib_dir))
                    for name in possible_names:
                        dll_path = user_lib_dir / name
                        if dll_path.exists():
                            try:
                                lib = CDLL(str(dll_path))
                                logging.info(
                                    f"Loaded FluidSynth from {dll_path}"
                                )
                                return lib
                            except OSError as e:
                                logging.debug(f"Could not load {dll_path}: {e}")
            except Exception as e:
                logging.debug(f"Error processing user_lib_dir: {e}")

        for name in possible_names:
            try:
                lib = CDLL(name)
                logging.info(f"Loaded FluidSynth library: {name}")
                return lib
            except OSError:
                continue

        found_lib = find_library("fluidsynth")
        if found_lib:
            try:
                lib = CDLL(found_lib)
                logging.info(f"Loaded FluidSynth via find_library: {found_lib}")
                return lib
            except OSError as e:
                logging.debug(f"Could not load {found_lib}: {e}")

        if system == "Windows":
            standard_paths = [
                Path(os.getenv("ProgramFiles", "C:\\Program Files"))
                / "FluidSynth"
                / "bin",
                Path(os.getenv("ProgramFiles(x86)", "C:\\Program Files (x86)"))
                / "FluidSynth"
                / "bin",
            ]
            for path in standard_paths:
                if path.exists():
                    self._add_to_path(str(path))
                    for name in possible_names:
                        full_path = path / name
                        if full_path.exists():
                            try:
                                lib = CDLL(str(full_path))
                                logging.info(
                                    f"Loaded FluidSynth from {full_path}"
                                )
                                return lib
                            except OSError as e:
                                logging.debug(
                                    f"Could not load {full_path}: {e}"
                                )

        error_msg = (
            "FluidSynth library not found. Please install FluidSynth:\n"
            "Windows: Download from https://github.com/FluidSynth/fluidsynth/releases\n"
            "Linux: sudo apt-get install libfluidsynth3 (Ubuntu/Debian) or equivalent\n"
            "macOS: brew install fluid-synth"
        )
        logging.error(error_msg)
        raise ImportError(error_msg)

    def _add_to_path(self, path):
        if path not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + path
        if platform.system() == "Windows" and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(path)
            except (OSError, AttributeError):
                pass

    def _set_prototypes(self):
        l = self._lib
        l.new_fluid_settings.restype = c_void_p
        l.fluid_settings_setnum.argtypes = [c_void_p, c_char_p, c_double]
        l.fluid_settings_setint.argtypes = [c_void_p, c_char_p, c_int]
        l.fluid_settings_setstr.argtypes = [c_void_p, c_char_p, c_char_p]
        l.delete_fluid_settings.argtypes = [c_void_p]
        l.new_fluid_synth.argtypes = [c_void_p]
        l.new_fluid_synth.restype = c_void_p
        l.delete_fluid_synth.argtypes = [c_void_p]
        l.fluid_synth_sfload.argtypes = [c_void_p, c_char_p, c_int]
        l.fluid_synth_sfload.restype = c_int
        l.fluid_synth_sfunload.argtypes = [c_void_p, c_int, c_int]
        l.new_fluid_audio_driver.argtypes = [c_void_p, c_void_p]
        l.new_fluid_audio_driver.restype = c_void_p
        l.delete_fluid_audio_driver.argtypes = [c_void_p]
        l.fluid_synth_noteon.argtypes = [c_void_p, c_int, c_int, c_int]
        l.fluid_synth_noteoff.argtypes = [c_void_p, c_int, c_int]
        l.fluid_synth_cc.argtypes = [c_void_p, c_int, c_int, c_int]
        l.fluid_synth_pitch_bend.argtypes = [c_void_p, c_int, c_int]
        l.fluid_synth_program_change.argtypes = [c_void_p, c_int, c_int]
        l.fluid_synth_bank_select.argtypes = [c_void_p, c_int, c_int]
        l.fluid_synth_program_select.argtypes = [
            c_void_p,
            c_int,
            c_int,
            c_int,
            c_int,
        ]
        l.fluid_synth_system_reset.argtypes = [c_void_p]
        l.fluid_synth_set_reverb.argtypes = [
            c_void_p,
            c_double,
            c_double,
            c_double,
            c_double,
        ]
        l.fluid_synth_set_chorus.argtypes = [
            c_void_p,
            c_int,
            c_double,
            c_double,
            c_double,
            c_int,
        ]
        try:
            l.fluid_synth_write_float.argtypes = [
                c_void_p,
                c_int,
                c_void_p,
                c_int,
                c_int,
                c_void_p,
                c_int,
                c_int,
            ]
            l.fluid_synth_write_float.restype = c_int
        except AttributeError:
            pass

    def start_audio(self, driver=None, device=None):
        if self.audio_driver:
            return
        if driver:
            self._lib.fluid_settings_setstr(
                self.settings, b"audio.driver", driver.encode()
            )
        if device:
            self._lib.fluid_settings_setstr(
                self.settings, b"audio.driver.device", device.encode()
            )
        self.audio_driver = self._lib.new_fluid_audio_driver(
            self.settings, self.synth
        )

    def get_samples(self, samples):
        left_buf = np.empty(samples, dtype=np.float32)
        right_buf = np.empty(samples, dtype=np.float32)
        l_ptr = left_buf.ctypes.data_as(c_void_p)
        r_ptr = right_buf.ctypes.data_as(c_void_p)
        self._lib.fluid_synth_write_float(
            self.synth, samples, l_ptr, 0, 1, r_ptr, 0, 1
        )
        return left_buf, right_buf

    def sf_load(self, path):
        try:
            path = Path(path).resolve()

            if not path.exists():
                logging.error(f"SoundFont file not found: {path}")
                return -1

            if not path.is_file():
                logging.error(
                    f"SoundFont path is not a regular file "
                    f"(is it a directory?): {path}"
                )
                return -1

            if path.stat().st_size < 1024:
                logging.error(
                    f"SoundFont file too small (likely corrupted): {path}"
                )
                return -1

            try:
                with open(path, "rb") as f:
                    header = f.read(4)
                    if header != b"RIFF":
                        logging.warning(
                            f"SoundFont may be corrupted (invalid RIFF header): {path}"
                        )
            except OSError as e:
                logging.error(f"Could not read SoundFont file: {e}")
                return -1

            result = self._lib.fluid_synth_sfload(
                self.synth, str(path).encode("utf-8"), 1
            )
            if result == -1:
                logging.error(f"FluidSynth failed to load SoundFont: {path}")
            else:
                logging.info(f"Successfully loaded SoundFont: {path}")
            return result
        except Exception as e:
            logging.error(f"Error loading SoundFont {path}: {e}")
            return -1

    def note_on(self, channel, key, velocity):
        channel, key, velocity = validate_note_message(channel, key, velocity)
        self._lib.fluid_synth_noteon(self.synth, channel, key, velocity)

    def note_off(self, channel, key):
        channel, key, _ = validate_note_message(channel, key)
        self._lib.fluid_synth_noteoff(self.synth, channel, key)

    def cc(self, channel, ctrl, val):
        channel = validate_channel(channel)
        ctrl, val = validate_cc(ctrl, val)
        self._lib.fluid_synth_cc(self.synth, channel, ctrl, val)

    def pitch_bend(self, channel, val):
        channel = validate_channel(channel)
        val = validate_pitch_bend(val)
        self._lib.fluid_synth_pitch_bend(self.synth, channel, val)

    def program_change(self, channel, program):
        self._lib.fluid_synth_program_change(self.synth, channel, program)

    def program_select(self, channel, sfid, bank, preset):
        self._lib.fluid_synth_program_select(
            self.synth, channel, sfid, bank, preset
        )

    def set_reverb(self, roomsize=0.5, damping=0.5, width=0.5, level=0.9):
        self._lib.fluid_synth_set_reverb(
            self.synth, roomsize, damping, width, level
        )

    def set_chorus(self, num=3, level=2.0, speed=0.3, depth=8.0, type_int=0):
        self._lib.fluid_synth_set_chorus(
            self.synth, num, level, speed, depth, type_int
        )

    def close(self):
        if self.audio_driver:
            self._lib.delete_fluid_audio_driver(self.audio_driver)
            self.audio_driver = None
        if self.synth:
            self._lib.delete_fluid_synth(self.synth)
            self.synth = None
        if self.settings:
            self._lib.delete_fluid_settings(self.settings)
            self.settings = None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()

    def __del__(self):
        self.close()
