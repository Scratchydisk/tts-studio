"""Bridge between the caption pipeline and the TTS model registry."""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

from tts_tests import registry
from tts_tests.caption.config import CaptionTTSConfig
from tts_tests.config import get_device

logger = logging.getLogger(__name__)


def generate_speech_for_segments(
    segments: list[dict],
    tts_config: CaptionTTSConfig,
    output_dir: Path,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Generate speech WAV files for each caption segment.

    Enriches each segment dict with:
      - speech_path: Path to the generated WAV file.
      - speech_duration: Duration of the generated audio in seconds.
      - padded_duration: speech_duration + silence_before + silence_after.

    Segments whose *padded_duration* exceeds the original screen duration are
    flagged with ``needs_freeze = True`` and a ``freeze_extra`` value.

    Returns the enriched segments list.
    """
    device = get_device()
    model = registry.load_model(tts_config.model_id, device=device)

    ref_audio = Path(tts_config.reference_audio) if tts_config.reference_audio else None
    total = len(segments)
    timeline: list[dict] = []

    for i, segment in enumerate(segments):
        entry = {**segment, "screen_duration": segment["end"] - segment["start"]}

        if not segment.get("text"):
            entry["has_speech"] = False
            entry["speech_path"] = None
            entry["speech_duration"] = 0.0
            entry["padded_duration"] = 0.0
            entry["needs_freeze"] = False
            timeline.append(entry)
            if on_progress is not None:
                on_progress(i + 1, total)
            continue

        try:
            result = model.generate(
                text=segment["text"],
                voice=tts_config.voice,
                reference_audio=ref_audio,
                reference_text=tts_config.reference_text,
            )
        except Exception as e:
            logger.warning("TTS failed for segment %d: %s", i, e)
            entry["has_speech"] = False
            entry["speech_path"] = None
            entry["speech_duration"] = 0.0
            entry["padded_duration"] = 0.0
            entry["needs_freeze"] = False
            timeline.append(entry)
            if on_progress is not None:
                on_progress(i + 1, total)
            continue

        if result.audio is None or len(result.audio) == 0:
            logger.warning("Empty audio for segment %d", i)
            entry["has_speech"] = False
            entry["speech_path"] = None
            entry["speech_duration"] = 0.0
            entry["padded_duration"] = 0.0
            entry["needs_freeze"] = False
            timeline.append(entry)
            if on_progress is not None:
                on_progress(i + 1, total)
            continue

        wav_path = output_dir / f"speech_{i:03d}.wav"
        sf.write(str(wav_path), result.audio, result.sample_rate)

        padded = tts_config.silence_before + result.duration + tts_config.silence_after

        entry["has_speech"] = True
        entry["speech_path"] = str(wav_path)
        entry["speech_duration"] = result.duration
        entry["padded_duration"] = round(padded, 3)
        entry["needs_freeze"] = padded > entry["screen_duration"]
        if entry["needs_freeze"]:
            entry["freeze_extra"] = round(padded - entry["screen_duration"], 3)

        timeline.append(entry)

        if on_progress is not None:
            on_progress(i + 1, total)

    return timeline


def generate_single_caption(
    text: str,
    tts_config: CaptionTTSConfig,
) -> tuple[np.ndarray, int]:
    """Generate audio for a single caption.

    Returns:
        A tuple of (audio_array, sample_rate).
    """
    device = get_device()
    model = registry.load_model(tts_config.model_id, device=device)

    ref_audio = Path(tts_config.reference_audio) if tts_config.reference_audio else None

    result = model.generate(
        text=text,
        voice=tts_config.voice,
        reference_audio=ref_audio,
        reference_text=tts_config.reference_text,
    )

    return result.audio, result.sample_rate
