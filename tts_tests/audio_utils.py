"""Shared audio utilities."""

import numpy as np
from scipy.signal import resample_poly
from math import gcd


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample audio from src_rate to dst_rate."""
    if src_rate == dst_rate:
        return audio
    g = gcd(src_rate, dst_rate)
    up = dst_rate // g
    down = src_rate // g
    return resample_poly(audio, up, down).astype(np.float32)


def normalise(audio: np.ndarray) -> np.ndarray:
    """Peak-normalise audio to [-1, 1]."""
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32)


def trim_silence(audio: np.ndarray, sr: int,
                 threshold_db: float = -40.0,
                 min_silence_ms: int = 100) -> np.ndarray:
    """Trim leading and trailing silence from audio.

    Args:
        audio: float32 audio array.
        sr: Sample rate.
        threshold_db: Silence threshold in dB below peak.
        min_silence_ms: Minimum silence duration to consider as silence.
    """
    if len(audio) == 0:
        return audio
    threshold = 10 ** (threshold_db / 20) * np.abs(audio).max()
    if threshold == 0:
        return audio

    window = int(sr * min_silence_ms / 1000)
    if window < 1:
        window = 1

    above = np.abs(audio) > threshold
    if not above.any():
        # Everything is below threshold — return original rather than empty
        return audio

    indices = np.where(above)[0]
    start = max(0, indices[0] - window)
    end = min(len(audio), indices[-1] + window)

    return audio[start:end]


def cap_duration(audio: np.ndarray, sr: int, max_seconds: float) -> np.ndarray:
    """Truncate audio to a maximum duration."""
    max_samples = int(sr * max_seconds)
    if len(audio) > max_samples:
        return audio[:max_samples]
    return audio
