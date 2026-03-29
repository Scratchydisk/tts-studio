"""Bridge between the caption pipeline and the TTS model registry."""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

from tts_tests import registry
from tts_tests.audio_utils import cap_duration, trim_silence
from tts_tests.caption.config import CaptionTTSConfig
from tts_tests.config import get_device

logger = logging.getLogger(__name__)


def _speech_path(output_dir: Path, index: int) -> Path:
    """Return the canonical path for a segment's speech WAV."""
    return output_dir / f"speech_{index:03d}.wav"


def generate_speech_for_segments(
    segments: list[dict],
    tts_config: CaptionTTSConfig,
    output_dir: Path,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Generate speech WAV files for each caption segment.

    If a WAV file already exists for a segment (e.g. from a preview), it is
    reused rather than regenerated. This allows pinning good takes.

    Enriches each segment dict with:
      - speech_path: Path to the generated WAV file.
      - speech_duration: Duration of the generated audio in seconds.
      - padded_duration: speech_duration + silence_before + silence_after.

    Segments whose *padded_duration* exceeds the original screen duration are
    flagged with ``needs_freeze = True`` and a ``freeze_extra`` value.

    Returns the enriched segments list.
    """
    device = get_device(tts_config.model_id)
    model = registry.load_model(tts_config.model_id, device=device)

    ref_audio = Path(tts_config.reference_audio) if tts_config.reference_audio else None
    total = len(segments)
    timeline: list[dict] = []

    # Max duration: 3x the screen duration or 30s, whichever is smaller
    global_max = 30.0

    for i, segment in enumerate(segments):
        entry = {**segment, "screen_duration": segment["end"] - segment["start"]}
        screen_dur = entry["screen_duration"]
        max_dur = min(max(screen_dur * 3, 1.0), global_max)

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

        wav_path = _speech_path(output_dir, i)

        # Reuse existing WAV if present (pinned from preview)
        if wav_path.exists():
            try:
                audio, sr = sf.read(str(wav_path), dtype="float32")
                duration = len(audio) / sr
                logger.info("Reusing pinned audio for segment %d (%.1fs)", i, duration)
                padded = tts_config.silence_before + duration + tts_config.silence_after
                entry["has_speech"] = True
                entry["speech_path"] = str(wav_path)
                entry["speech_duration"] = duration
                entry["padded_duration"] = round(padded, 3)
                entry["needs_freeze"] = padded > screen_dur
                if entry["needs_freeze"]:
                    entry["freeze_extra"] = round(padded - screen_dur, 3)
                timeline.append(entry)
                if on_progress is not None:
                    on_progress(i + 1, total)
                continue
            except Exception as e:
                logger.warning("Failed to read pinned audio for segment %d: %s", i, e)

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

        # Quality controls: trim silence and cap duration
        audio = trim_silence(result.audio, result.sample_rate)
        audio = cap_duration(audio, result.sample_rate, max_dur)
        duration = len(audio) / result.sample_rate

        sf.write(str(wav_path), audio, result.sample_rate)

        padded = tts_config.silence_before + duration + tts_config.silence_after

        entry["has_speech"] = True
        entry["speech_path"] = str(wav_path)
        entry["speech_duration"] = duration
        entry["padded_duration"] = round(padded, 3)
        entry["needs_freeze"] = padded > screen_dur
        if entry["needs_freeze"]:
            entry["freeze_extra"] = round(padded - screen_dur, 3)

        timeline.append(entry)

        if on_progress is not None:
            on_progress(i + 1, total)

    return timeline


def generate_single_caption(
    text: str,
    tts_config: CaptionTTSConfig,
    max_retries: int = 3,
) -> tuple[np.ndarray, int]:
    """Generate audio for a single caption, retrying on empty results.

    Returns:
        A tuple of (audio_array, sample_rate).
    """
    device = get_device(tts_config.model_id)
    model = registry.load_model(tts_config.model_id, device=device)

    ref_audio = Path(tts_config.reference_audio) if tts_config.reference_audio else None

    for attempt in range(max_retries):
        try:
            result = model.generate(
                text=text,
                voice=tts_config.voice,
                reference_audio=ref_audio,
                reference_text=tts_config.reference_text,
            )
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning("Generation failed on attempt %d: %s — retrying...", attempt + 1, e)
                continue
            raise

        if result.audio is not None and len(result.audio) > 0:
            audio = trim_silence(result.audio, result.sample_rate)
            if len(audio) > 0:
                return audio, result.sample_rate

        if attempt < max_retries - 1:
            logger.warning("Empty audio on attempt %d, retrying...", attempt + 1)

    raise RuntimeError("Model returned empty audio after multiple attempts. Try a different voice or shorter text.")
