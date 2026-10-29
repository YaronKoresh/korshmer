import contextlib
import gc
import tempfile
import traceback

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from PySide6.QtGui import QImage

from korshmer.constants import SPECTRUM_MAX_DURATION
from korshmer.utils.filesystem import is_midi_path

matplotlib.use("Agg")

plt.ioff()


@contextlib.contextmanager
def matplotlib_figure(*args, **kwargs):
    fig = None
    ax = None
    try:
        fig, ax = plt.subplots(*args, **kwargs)
        yield fig, ax
    finally:
        if fig is not None:
            plt.close(fig)

            del fig
        if ax is not None:
            del ax


def create_spectrum_visualization(audio_path: str) -> str | None:
    try:
        y, sr = librosa.load(
            audio_path, sr=None, duration=SPECTRUM_MAX_DURATION
        )
        n_fft = 16384
        hop_length = 512
        stft_result = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft_result)
        avg_magnitude = np.mean(magnitude, axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        magnitude_db = librosa.amplitude_to_db(avg_magnitude, ref=np.max)

        del y, stft_result, magnitude, avg_magnitude

        with matplotlib_figure(figsize=(10, 6), facecolor="#f0f0f0") as (
            fig,
            ax,
        ):
            ax.set_facecolor("white")
            ax.fill_between(
                freqs,
                magnitude_db,
                y2=np.min(magnitude_db) - 1,
                color="#7c3aed",
                alpha=0.8,
                zorder=2,
            )
            ax.plot(freqs, magnitude_db, color="#4c2a8c", linewidth=1, zorder=3)
            ax.set_xscale("log")
            ax.set_xlim(5, sr / 2)
            ax.set_ylim(np.min(magnitude_db) - 1, np.max(magnitude_db) + 5)
            xticks = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
            xtick_labels = [
                "10",
                "20",
                "50",
                "100",
                "200",
                "500",
                "1k",
                "2k",
                "5k",
                "10k",
                "20k",
            ]
            valid_ticks = [x for x in xticks if x < sr / 2]
            valid_labels = [
                label
                for x, label in zip(xticks, xtick_labels, strict=False)
                if x < sr / 2
            ]
            ax.set_xticks(valid_ticks)
            ax.set_xticklabels(valid_labels)
            ax.grid(
                True, which="both", ls="--", color="gray", alpha=0.6, zorder=1
            )
            ax.set_title(
                "Frequency Analysis", color="black", fontsize=14, pad=15
            )
            ax.set_xlabel("Frequency (Hz)", color="black")
            ax.set_ylabel("Amplitude (dB)", color="black")
            ax.tick_params(colors="black", which="both")
            audible_mask = freqs > 1
            if np.any(audible_mask):
                peak_idx = np.argmax(magnitude_db[audible_mask])
                peak_freq = freqs[audible_mask][peak_idx]
                peak_db = magnitude_db[audible_mask][peak_idx]
                peak_text = f"Peak: {peak_freq:.0f} Hz at {peak_db:.1f} dB"
                ax.text(
                    0.98,
                    0.95,
                    peak_text,
                    transform=ax.transAxes,
                    color="black",
                    ha="right",
                    va="top",
                    fontsize=11,
                    fontweight="bold",
                )
            fig.tight_layout()
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as tmp:
                temp_path = tmp.name
            fig.savefig(temp_path, facecolor=fig.get_facecolor(), dpi=300)

        del freqs, magnitude_db

        return temp_path
    except Exception as e:
        print(f"Error creating spectrum: {e}")
        return None
    finally:
        gc.collect()


def calculate_waveform_features(
    audio_path: str,
) -> tuple[np.ndarray | None, float]:
    try:
        if is_midi_path(audio_path):
            return None, 0
        y, sr = librosa.load(audio_path, sr=8000, mono=True)
        hop_length = int(sr / 60)
        if hop_length < 16:
            hop_length = 16
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        if rms.max() > 0:
            rms /= rms.max()
        rms = np.sqrt(rms)
        duration = librosa.get_duration(y=y, sr=sr)

        del y

        return rms, duration
    except Exception:
        traceback.print_exc()
        return None, 0


def create_spectrogram_image(path: str) -> QImage:
    try:
        y, sr = sf.read(path, frames=8000000, always_2d=True, dtype="float32")
        y = y.mean(axis=1) if y.shape[1] > 1 else y.flatten()
        step = max(1, int(sr // 22050))
        y = y[::step]
        sr = sr // step
        hop_length = int(len(y) / 600)
        if hop_length < 256:
            hop_length = 256
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, fmax=8000, hop_length=hop_length
        )
        S_dB = librosa.power_to_db(S, ref=np.max)
        S_min, S_max = S_dB.min(), S_dB.max()

        del S, y
        if S_max == S_min:
            S_max = S_min + 1e-6
        S_norm = 255 * (S_dB - S_min) / (S_max - S_min)
        S_norm = np.flipud(S_norm.astype(np.uint8))

        del S_dB

        h, w = S_norm.shape
        t = np.linspace(0, 1, 256)
        lut_r = (255 + (26 - 255) * t).astype(np.uint8)
        lut_g = (255 + (115 - 255) * t).astype(np.uint8)
        lut_b = (255 + (232 - 255) * t).astype(np.uint8)
        rgb = np.dstack((lut_r[S_norm], lut_g[S_norm], lut_b[S_norm])).tobytes()

        result = QImage(rgb, w, h, w * 3, QImage.Format_RGB888).copy()

        del S_norm, t, lut_r, lut_g, lut_b

        gc.collect()

        return result
    except Exception:
        return QImage()
