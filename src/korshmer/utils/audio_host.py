import json
import logging
import sys
import threading
import types
import weakref
from functools import cache
from typing import Any

from korshmer.constants import (
    AUDIO_FALLBACK_DEVICE_TRIES,
    AUDIO_HOST_FILE,
    MAX_PLAYBACK_BUFFER_SIZE,
    MIN_PLAYBACK_BUFFER_SIZE,
)

HOST_DEFAULT = "default"
HOST_ASIO = "asio"
HOST_COREAUDIO = "coreaudio"
HOST_JACK = "jack"
HOST_CHOICES = (HOST_DEFAULT, HOST_ASIO, HOST_COREAUDIO, HOST_JACK)
HOST_KEY = "host"
OUTPUT_DEVICE_KEY = "output_device"
BUFFER_SIZE_KEY = "buffer_size"


def _buffer_size_choices() -> tuple:
    sizes = []
    size = MIN_PLAYBACK_BUFFER_SIZE
    while size <= MAX_PLAYBACK_BUFFER_SIZE:
        sizes.append(size)
        size *= 2
    return tuple(sizes)


BUFFER_SIZE_CHOICES = _buffer_size_choices()


_HOST_LABELS = {
    HOST_ASIO: "ASIO",
    HOST_COREAUDIO: "CoreAudio",
    HOST_JACK: "JACK",
}
_HOST_API_MATCH = {
    HOST_ASIO: "asio",
    HOST_COREAUDIO: "core audio",
    HOST_JACK: "jack",
}

_logger = logging.getLogger(__name__)
_lock = threading.Lock()
_preference = None
_missing_devices_told: set = set()


def asio_supported() -> bool:
    return sys.platform == "win32"


def coreaudio_supported() -> bool:
    return sys.platform == "darwin"


def jack_supported() -> bool:
    return sys.platform.startswith("linux")


def host_supported(host: str) -> bool:
    if host == HOST_ASIO:
        return asio_supported()
    if host == HOST_COREAUDIO:
        return coreaudio_supported()
    if host == HOST_JACK:
        return jack_supported()
    return True


def available_hosts() -> tuple:
    return tuple(h for h in HOST_CHOICES if host_supported(h))


def _normalize(value) -> str:
    if isinstance(value, str) and value.strip().lower() in HOST_CHOICES:
        return value.strip().lower()
    return HOST_DEFAULT


def _read_settings() -> dict:
    try:
        if AUDIO_HOST_FILE.exists():
            data = json.loads(AUDIO_HOST_FILE.read_text("utf-8"))
            if isinstance(data, dict):
                return data
    except (OSError, ValueError) as e:
        _logger.warning(f"Could not read audio settings: {e}")
    return {}


def _write_setting(key: str, value) -> None:
    settings = _read_settings()
    settings[key] = value
    try:
        AUDIO_HOST_FILE.parent.mkdir(parents=True, exist_ok=True)
        AUDIO_HOST_FILE.write_text(json.dumps(settings), "utf-8")
    except OSError as e:
        _logger.warning(f"Could not save audio settings: {e}")


def _valid_buffer_size(value):
    if type(value) is int and value in BUFFER_SIZE_CHOICES:
        return value
    return None


def _valid_device_name(value):
    return value if isinstance(value, str) and value.strip() else None


def get_buffer_size_preference():
    return _valid_buffer_size(_read_settings().get(BUFFER_SIZE_KEY))


def set_buffer_size_preference(frames):
    frames = _valid_buffer_size(frames)
    with _lock:
        _write_setting(BUFFER_SIZE_KEY, frames)
    return frames


def get_output_device_preference():
    return _valid_device_name(_read_settings().get(OUTPUT_DEVICE_KEY))


def set_output_device_preference(name):
    name = _valid_device_name(name)
    with _lock:
        _write_setting(OUTPUT_DEVICE_KEY, name)
    return name


def _unavailable_sounddevice(reason: str) -> types.ModuleType:
    module = types.ModuleType("sounddevice")

    class PortAudioError(Exception):
        pass

    class CallbackStop(Exception):
        pass

    def unavailable(*_args, **_kwargs):
        raise PortAudioError(reason)

    module.__dict__.update(
        PortAudioError=PortAudioError,
        CallbackStop=CallbackStop,
        OutputStream=unavailable,
        InputStream=unavailable,
        Stream=unavailable,
        query_devices=unavailable,
        query_hostapis=unavailable,
        default=types.SimpleNamespace(device=(None, None), reset=lambda: None),
    )
    return module


@cache
def sounddevice_module() -> Any:
    try:
        import sounddevice
    except Exception as e:
        _logger.warning(f"Audio devices are unavailable: {e}")
        return _unavailable_sounddevice(str(e))
    return sounddevice


def list_output_devices(sd_module=None) -> list:
    if sd_module is None:
        sd_module = sounddevice_module()
    try:
        apis = [str(api.get("name", "")) for api in sd_module.query_hostapis()]
        found = []
        for index, info in enumerate(sd_module.query_devices()):
            if info.get("max_output_channels", 0) <= 0:
                continue
            api = info.get("hostapi", -1)
            name = str(info.get("name", f"Device {index}"))
            if isinstance(api, int) and 0 <= api < len(apis) and apis[api]:
                name = f"{name} ({apis[api]})"
            found.append((index, name))
    except Exception as e:
        _logger.warning(f"Could not list audio output devices: {e}")
        return []
    return found


def _saved_output_device(sd_module):
    name = get_output_device_preference()
    if name is None:
        return None, None
    for index, label in list_output_devices(sd_module):
        if label == name:
            return index, None
    _logger.warning(f"Saved audio output {name!r} was not found")
    if name in _missing_devices_told:
        return None, None
    _missing_devices_told.add(name)
    return None, (
        f"Audio output '{name}' is not connected. "
        "Playing on the default output instead."
    )


def apply_audio_preferences(engine, sd_module=None):
    if hasattr(engine, "set_preferred_buffer_size"):
        engine.set_preferred_buffer_size(get_buffer_size_preference())
    if not hasattr(engine, "set_output_device"):
        return None
    index, notice = _saved_output_device(sd_module)
    engine.set_output_device(index)
    return notice


def get_host_preference() -> str:
    global _preference
    with _lock:
        if _preference is None:
            _preference = _normalize(_read_settings().get(HOST_KEY))
        preference = _preference
    if not host_supported(preference):
        return HOST_DEFAULT
    return preference


def set_host_preference(value: str) -> str:
    global _preference
    value = _normalize(value)
    if not host_supported(value):
        value = HOST_DEFAULT
    with _lock:
        _preference = value
        _write_setting(HOST_KEY, value)
    return value


def reset_host_preference_cache() -> None:
    global _preference
    with _lock:
        _preference = None
        _missing_devices_told.clear()


def find_host_device(sd_module, host: str, kind: str = "output"):
    needle = _HOST_API_MATCH.get(host)
    if needle is None:
        return None
    try:
        for api in sd_module.query_hostapis():
            if needle not in str(api.get("name", "")).lower():
                continue
            index = api.get(f"default_{kind}_device", -1)
            if isinstance(index, int) and index >= 0:
                return index
    except Exception as e:
        _logger.warning(f"Could not query audio host APIs: {e}")
    return None


def find_asio_output_device(sd_module):
    return find_host_device(sd_module, HOST_ASIO, "output")


def find_asio_input_device(sd_module):
    return find_host_device(sd_module, HOST_ASIO, "input")


def find_coreaudio_output_device(sd_module):
    return find_host_device(sd_module, HOST_COREAUDIO, "output")


def find_coreaudio_input_device(sd_module):
    return find_host_device(sd_module, HOST_COREAUDIO, "input")


def find_jack_output_device(sd_module):
    return find_host_device(sd_module, HOST_JACK, "output")


def find_jack_input_device(sd_module):
    return find_host_device(sd_module, HOST_JACK, "input")


def _open_stream(sd_module, stream_name: str, kind: str, kwargs):
    host = get_host_preference()
    if host != HOST_DEFAULT:
        label = _HOST_LABELS[host]
        factory = getattr(sd_module, stream_name)
        device = find_host_device(sd_module, host, kind)
        if device is None:
            _logger.warning(
                f"{label} selected but no {label} {kind} device found; "
                "using the default audio device"
            )
        else:
            try:
                return factory(device=device, **kwargs)
            except sd_module.PortAudioError as e:
                _logger.warning(
                    f"Could not open {label} device {device}: {e}; "
                    "using the default audio device"
                )
    return getattr(sd_module, stream_name)(**kwargs)


def open_output_stream(sd_module, **kwargs):
    stream = _open_stream(sd_module, "OutputStream", "output", kwargs)
    _remember_output_stream(stream)
    return stream


def _close_quietly(stream) -> None:
    try:
        stream.close()
    except Exception as e:
        _logger.debug(f"Could not close failed stream: {e}")


def _default_output_index(sd_module):
    try:
        value = sd_module.default.device
        if isinstance(value, (tuple, list)) or hasattr(value, "__getitem__"):
            value = value[1]
    except Exception:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _spare_output_devices(sd_module, channels, skip):
    try:
        devices = sd_module.query_devices()
        found = []
        for index, info in enumerate(devices):
            if index == skip:
                continue
            if info.get("max_output_channels", 0) >= channels:
                found.append(index)
    except Exception as e:
        _logger.warning(f"Could not list audio output devices: {e}")
        return []
    return found[:AUDIO_FALLBACK_DEVICE_TRIES]


def _start_on_spare_device(sd_module, kwargs):
    channels = kwargs.get("channels", 1)
    if not isinstance(channels, int) or isinstance(channels, bool):
        channels = 1
    spare = _spare_output_devices(
        sd_module, channels, _default_output_index(sd_module)
    )
    for index in spare:
        options = dict(kwargs)
        options["device"] = index
        stream = None
        try:
            stream = sd_module.OutputStream(**options)
            _remember_output_stream(stream)
            stream.start()
        except (sd_module.PortAudioError, ValueError) as e:
            _logger.warning(f"Audio output device {index} failed: {e}")
            if stream is not None:
                _close_quietly(stream)
            continue
        _logger.warning(f"Playing on audio output device {index} instead")
        return stream
    return None


def open_and_start_output_stream(sd_module, **kwargs):
    try:
        return _open_and_start_on_chosen_host(sd_module, **kwargs)
    except sd_module.PortAudioError:
        if "device" in kwargs:
            raise
        stream = _start_on_spare_device(sd_module, kwargs)
        if stream is None:
            raise
        return stream


def _open_and_start_on_chosen_host(sd_module, **kwargs):
    host = get_host_preference()
    try:
        stream = open_output_stream(sd_module, **kwargs)
    except (sd_module.PortAudioError, ValueError) as e:
        if host == HOST_DEFAULT:
            raise
        _logger.warning(
            f"Could not open {_HOST_LABELS[host]} stream: {e}; "
            "using the default audio device"
        )
        stream = sd_module.OutputStream(**kwargs)
        _remember_output_stream(stream)
    try:
        stream.start()
        return stream
    except sd_module.PortAudioError as e:
        _close_quietly(stream)
        if host == HOST_DEFAULT:
            raise
        _logger.warning(
            f"Could not start {_HOST_LABELS[host]} stream: {e}; "
            "using the default audio device"
        )
    stream = sd_module.OutputStream(**kwargs)
    _remember_output_stream(stream)
    try:
        stream.start()
    except BaseException:
        _close_quietly(stream)
        raise
    return stream


def open_input_stream(sd_module, **kwargs):
    return _open_stream(sd_module, "InputStream", "input", kwargs)


def open_duplex_stream(sd_module, **kwargs):
    host = get_host_preference()
    if host != HOST_DEFAULT:
        label = _HOST_LABELS[host]
        in_dev = find_host_device(sd_module, host, "input")
        out_dev = find_host_device(sd_module, host, "output")
        if in_dev is None or out_dev is None:
            _logger.warning(
                f"{label} selected but no {label} input/output device pair "
                "found; using the default audio devices"
            )
        else:
            try:
                return sd_module.Stream(device=(in_dev, out_dev), **kwargs)
            except sd_module.PortAudioError as e:
                _logger.warning(
                    f"Could not open {label} devices {in_dev}/{out_dev}: {e}; "
                    "using the default audio devices"
                )
    return sd_module.Stream(**kwargs)


_output_ref = None


def _remember_output_stream(stream) -> None:
    global _output_ref
    try:
        _output_ref = weakref.ref(stream)
    except TypeError:
        _output_ref = None


def _clean_latency(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if 0.0 < value < float("inf") else 0.0


def _latency_seconds(stream, index: int) -> float:
    try:
        value = stream.latency
        if isinstance(value, (tuple, list)):
            value = value[index]
    except Exception:
        return 0.0
    return _clean_latency(value)


def stream_input_latency(stream) -> float:
    return _latency_seconds(stream, 0)


def active_output_latency() -> float:
    ref = _output_ref
    stream = ref() if ref is not None else None
    if stream is None:
        return 0.0
    try:
        if getattr(stream, "closed", False) or not getattr(
            stream, "active", False
        ):
            return 0.0
    except Exception:
        return 0.0
    return _latency_seconds(stream, 1)


def compensation_frames(
    input_latency: float, output_latency: float, samplerate: int
) -> int:
    total = _clean_latency(input_latency) + _clean_latency(output_latency)
    return int(round(total * samplerate))
