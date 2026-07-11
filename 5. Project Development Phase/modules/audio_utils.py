"""
modules/audio_utils.py

Librosa/SoundFile-backed audio ingestion for VBCUA.

Responsibilities:
    - load_audio_signal:        Load + validate + resample raw audio.
    - extract_raw_audio_metrics: Produce the 4 raw metrics that feed
                                  scoring_engine.evaluate_understanding()
                                  and populate the audio_feature table.
    - save_waveform:             Render a dark-themed waveform PNG for the
                                  dashboard and PDF report.

This module NEVER computes point values — only raw signal metrics.
scoring_engine.py owns all thresholding logic.
"""

import logging
import os
from pathlib import Path
from typing import TypedDict

import librosa
import librosa.display
import matplotlib
import numpy as np
import soundfile as sf

matplotlib.use("Agg")  # non-interactive backend, safe for Streamlit's server thread
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

DEFAULT_TARGET_SR = 16000
MIN_DURATION_SEC = 1.0
SILENCE_AMPLITUDE_EPS = 1e-6   # below this, a signal is treated as effectively empty
PAUSE_THRESHOLD_RATIO = 0.10   # silent frame = RMS below 10% of max RMS
DEFAULT_FRAME_LENGTH = 2048
DEFAULT_HOP_LENGTH = 512


class AudioLoadError(ValueError):
    """Raised when an audio file fails validation (too short, silent, unreadable)."""


class RawAudioMetrics(TypedDict):
    pause_ratio: float
    rms_energy: float
    zero_crossing_rate: float
    duration_sec: float


def load_audio_signal(
    file_path: str | Path,
    target_sr: int = DEFAULT_TARGET_SR,
) -> tuple[np.ndarray, int]:
    """
    Loads a .wav or .mp3 file, downmixes to mono, and resamples to target_sr.

    Args:
        file_path: Path to the audio file (.wav or .mp3).
        target_sr: Uniform sample rate to resample to (default 16kHz, matching
                    Whisper's expected input rate).

    Returns:
        (y, sr): y is a 1-D float32 numpy array, sr is the resulting sample rate.

    Raises:
        AudioLoadError: if the file cannot be decoded, is shorter than
            MIN_DURATION_SEC, or the decoded signal is empty/silent.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise AudioLoadError(f"Audio file not found: {file_path}")

    try:
        # librosa.load handles both wav/mp3 via soundfile/audioread backends
        # and performs the resample to target_sr in one pass.
        y, sr = librosa.load(str(file_path), sr=target_sr, mono=True)
    except Exception as exc:
        # Covers corrupted files, unsupported/exotic codecs, truncated headers, etc.
        logger.exception("Failed to decode audio file: %s", file_path)
        raise AudioLoadError(f"Could not decode audio file '{file_path.name}': {exc}") from exc

    if y is None or y.size == 0:
        raise AudioLoadError(f"Decoded signal is empty for file: {file_path.name}")

    duration_sec = len(y) / sr
    if duration_sec < MIN_DURATION_SEC:
        raise AudioLoadError(
            f"Audio too short: {duration_sec:.2f}s (minimum {MIN_DURATION_SEC}s required)."
        )

    if np.max(np.abs(y)) < SILENCE_AMPLITUDE_EPS:
        raise AudioLoadError(f"Audio signal is completely silent: {file_path.name}")

    logger.info(
        "Loaded audio '%s': duration=%.2fs sr=%dHz samples=%d",
        file_path.name, duration_sec, sr, len(y),
    )
    return y, sr


def extract_raw_audio_metrics(
    y: np.ndarray,
    sr: int,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> RawAudioMetrics:
    """
    Computes raw signal-level metrics for the audio_feature table.

    Args:
        y: Mono audio signal (as returned by load_audio_signal).
        sr: Sample rate of y.
        frame_length: Frame size used for framewise RMS analysis.
        hop_length: Hop size between frames.

    Returns:
        RawAudioMetrics dict with keys: pause_ratio, rms_energy,
        zero_crossing_rate, duration_sec. These are RAW metrics —
        pass them into scoring_engine.evaluate_understanding() for
        point conversion; this function does not classify or score.

    Raises:
        AudioLoadError: if y is empty (defensive check — should already
            have been caught by load_audio_signal).
    """
    if y is None or y.size == 0:
        raise AudioLoadError("Cannot extract metrics from an empty signal array.")

    # Framewise RMS energy across the clip
    rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    max_rms = float(np.max(rms_frames)) if rms_frames.size else 0.0

    if max_rms <= 0.0:
        # Defensive fallback; load_audio_signal's silence check should prevent this.
        logger.warning("Max RMS is zero during metric extraction; treating clip as fully silent.")
        pause_ratio = 1.0
    else:
        silence_threshold = PAUSE_THRESHOLD_RATIO * max_rms
        silent_frame_count = int(np.sum(rms_frames < silence_threshold))
        pause_ratio = silent_frame_count / len(rms_frames)

    rms_energy = float(np.mean(rms_frames)) if rms_frames.size else 0.0

    zcr_frames = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_length, hop_length=hop_length
    )[0]
    zero_crossing_rate = float(np.mean(zcr_frames)) if zcr_frames.size else 0.0

    duration_sec = float(len(y) / sr)

    metrics: RawAudioMetrics = {
        "pause_ratio": round(pause_ratio, 6),
        "rms_energy": round(rms_energy, 6),
        "zero_crossing_rate": round(zero_crossing_rate, 6),
        "duration_sec": round(duration_sec, 3),
    }

    logger.info(
        "Extracted audio metrics: pause_ratio=%.4f rms_energy=%.6f zcr=%.6f duration=%.2fs",
        metrics["pause_ratio"], metrics["rms_energy"],
        metrics["zero_crossing_rate"], metrics["duration_sec"],
    )
    return metrics


def save_waveform(
    file_path: str | Path,
    output_img_path: str | Path = "assets/waveform.png",
    target_sr: int = DEFAULT_TARGET_SR,
) -> str:
    """
    Renders a dark-themed waveform plot for the given audio file and saves it as PNG.

    Args:
        file_path: Path to the source audio file.
        output_img_path: Destination PNG path (parent dirs created if needed).
        target_sr: Sample rate to load at before plotting.

    Returns:
        The output_img_path as a string, for storage in session_state / DB.

    Raises:
        AudioLoadError: propagated from load_audio_signal if the source
            file is invalid.
    """
    y, sr = load_audio_signal(file_path, target_sr=target_sr)

    output_img_path = Path(output_img_path)
    output_img_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10, 3), dpi=120)

        librosa.display.waveshow(y, sr=sr, ax=ax, color="#2ecc71", linewidth=0.8)

        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        ax.set_xlabel("Time (s)", color="#c9d1d9", fontsize=9)
        ax.set_ylabel("Amplitude", color="#c9d1d9", fontsize=9)
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.grid(True, color="#30363d", linewidth=0.4, alpha=0.5)
        fig.tight_layout()

        fig.savefig(output_img_path, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

    except Exception:
        plt.close("all")
        logger.exception("Failed to render/save waveform image for: %s", file_path)
        raise

    logger.info("Waveform saved to: %s", output_img_path)
    return str(output_img_path)
