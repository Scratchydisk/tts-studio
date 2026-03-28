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
