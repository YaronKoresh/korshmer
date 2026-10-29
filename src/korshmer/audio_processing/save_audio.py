import os

import numpy as np
import soundfile as sf


def save_audio(
    path: str,
    y: np.ndarray,
    sr: int = 44100,
    file_format: str = "MP3",
    *,
    bits: int = 32,
    bitrate: int = 320,
) -> None:
    ceiling_db = -0.1
    lin_amp = 10 ** (ceiling_db / 20.0)

    y = np.clip(y * lin_amp, -lin_amp, lin_amp)

    base_path = os.path.splitext(path)[0]

    if file_format.lower() == "mp3":
        import lameenc

        final_path = f"{base_path}.mp3"

        INT16_MAX = np.iinfo(np.int16).max
        y_scaled = (y * INT16_MAX).astype(np.int16)

        if y_scaled.ndim == 1:
            channels = 1
            y_interleaved = y_scaled
        else:
            y_interleaved = np.ascontiguousarray(
                y_scaled.T
                if y_scaled.shape[0] < y_scaled.shape[1]
                else y_scaled
            )
            channels = y_interleaved.shape[1]

        encoder = lameenc.Encoder()
        encoder.set_bit_rate(bitrate)
        encoder.set_in_sample_rate(sr)
        encoder.set_channels(channels)
        encoder.set_quality(2)

        mp3_data = encoder.encode(y_interleaved.tobytes())
        mp3_data += encoder.flush()

        with open(final_path, "wb") as f:
            f.write(mp3_data)
        return

    final_path = f"{base_path}.wav"
    if bits == 16:
        INT16_MIN = np.iinfo(np.int16).min
        INT16_MAX = np.iinfo(np.int16).max
        y_scaled = y * float(INT16_MAX)

        tpdf = (
            np.random.uniform(-1.0, 1.0, y_scaled.shape)
            + np.random.uniform(-1.0, 1.0, y_scaled.shape)
        ) * 0.5

        y_dithered = y_scaled + tpdf
        y = np.clip(np.rint(y_dithered), INT16_MIN, INT16_MAX).astype(np.int16)

    elif bits == 32:
        y = y.astype(np.float32)

    elif bits == 64:
        y = y.astype(np.float64)

    else:
        raise ValueError(f"Unsupported bit depth: {bits}. Use 16, 32, or 64.")

    if y.ndim == 2 and y.shape[0] < y.shape[1]:
        y = y.T

    subtype_map = {16: "PCM_16", 32: "FLOAT", 64: "DOUBLE"}
    sf.write(final_path, y, sr, subtype=subtype_map[bits], format="RF64")
