"""Public API for external process integration.

Usage:
    from tts_tests.api import caption_video, generate_tts

    # Generate a narrated video
    result = caption_video("video.mp4", profile="emma")

    # Simple TTS
    audio, sr = generate_tts("Hello world", model_id="kokoro-82m")
"""

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from tts_tests.caption.config import CaptionTTSConfig, Config, OutputConfig
from tts_tests.profiles import VoiceProfile, get_profile

logger = logging.getLogger(__name__)


@dataclass
class CaptionResult:
    output_path: Path
    duration: float
    segments_count: int


def _ensure_registry():
    """Ensure model registry is discovered."""
    from tts_tests import registry

    if not registry.list_models():
        registry.discover()


def _resolve_profile(
    profile: str | None,
    model_id: str | None,
    voice: str | None,
    reference_audio: str | Path | None,
    reference_text: str | None,
) -> tuple[str, str | None, str | None, str | None]:
    """Resolve profile and explicit overrides into final TTS parameters.

    Returns (model_id, voice, reference_audio, reference_text).
    Explicit non-None parameters override the profile.
    """
    if profile is not None:
        vp: VoiceProfile = get_profile(profile)
        resolved_model = model_id if model_id is not None else vp.model_id
        resolved_voice = voice if voice is not None else vp.voice
        resolved_ref_audio = (
            str(reference_audio) if reference_audio is not None else vp.reference_audio
        )
        resolved_ref_text = (
            reference_text if reference_text is not None else vp.reference_text
        )
    else:
        resolved_model = model_id if model_id is not None else "kokoro-82m"
        resolved_voice = voice
        resolved_ref_audio = str(reference_audio) if reference_audio is not None else None
        resolved_ref_text = reference_text

    return resolved_model, resolved_voice, resolved_ref_audio, resolved_ref_text


def caption_video(
    video_path: str | Path,
    srt_path: str | Path | None = None,
    profile: str | None = None,
    model_id: str | None = None,
    voice: str | None = None,
    reference_audio: str | Path | None = None,
    reference_text: str | None = None,
    output_path: str | Path | None = None,
    output_format: str = "mkv",
    silence_before: float = 0.3,
    silence_after: float = 0.5,
    from_stage: str | None = None,
    keep_intermediates: bool = False,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> CaptionResult:
    """Convert a captioned video into a narrated video.

    Either ``profile`` or ``model_id`` must be provided.  If *profile* is
    given, it overrides model_id/voice/reference_audio/reference_text.
    If neither is given, defaults to kokoro-82m.
    """
    _ensure_registry()

    resolved_model, resolved_voice, resolved_ref_audio, resolved_ref_text = (
        _resolve_profile(profile, model_id, voice, reference_audio, reference_text)
    )

    tts_config = CaptionTTSConfig(
        model_id=resolved_model,
        voice=resolved_voice,
        reference_audio=resolved_ref_audio,
        reference_text=resolved_ref_text,
        silence_before=silence_before,
        silence_after=silence_after,
    )
    config = Config(
        tts=tts_config,
        output=OutputConfig(format=output_format),
    )

    video_path = Path(video_path)
    srt = Path(srt_path) if srt_path is not None else None
    out = Path(output_path) if output_path is not None else None

    from tts_tests.caption.pipeline import run_pipeline
    from tts_tests.caption.srt_parser import parse_srt_file

    # Count segments for the result — quick parse if SRT is available
    srt_for_count = srt or video_path.with_suffix(".srt")
    segments_count = 0
    if srt_for_count.exists():
        try:
            segments_count = len(parse_srt_file(srt_for_count))
        except Exception:
            pass

    result_path = run_pipeline(
        video_path=video_path,
        srt_path=srt,
        config=config,
        output_path=out,
        from_stage=from_stage,
        keep_intermediates=keep_intermediates,
        on_progress=on_progress,
    )

    # Get output duration via ffprobe
    duration = _get_media_duration(result_path)

    return CaptionResult(
        output_path=result_path,
        duration=duration,
        segments_count=segments_count,
    )


def caption_batch(
    input_dir: str | Path,
    video_pattern: str = "*.mp4",
    profile: str | None = None,
    model_id: str | None = None,
    voice: str | None = None,
    reference_audio: str | Path | None = None,
    reference_text: str | None = None,
    output_format: str = "mkv",
    silence_before: float = 0.3,
    silence_after: float = 0.5,
    keep_intermediates: bool = False,
) -> list[CaptionResult]:
    """Process all video+SRT pairs in a directory."""
    _ensure_registry()

    resolved_model, resolved_voice, resolved_ref_audio, resolved_ref_text = (
        _resolve_profile(profile, model_id, voice, reference_audio, reference_text)
    )

    tts_config = CaptionTTSConfig(
        model_id=resolved_model,
        voice=resolved_voice,
        reference_audio=resolved_ref_audio,
        reference_text=resolved_ref_text,
        silence_before=silence_before,
        silence_after=silence_after,
    )
    config = Config(
        tts=tts_config,
        output=OutputConfig(format=output_format),
    )

    input_dir = Path(input_dir)

    from tts_tests.caption.batch import process_directory

    raw_results = process_directory(
        input_dir=input_dir,
        config=config,
        video_pattern=video_pattern,
        keep_intermediates=keep_intermediates,
    )

    results: list[CaptionResult] = []
    for entry in raw_results:
        if entry["success"] and entry["output"] is not None:
            output_path = Path(entry["output"])
            duration = _get_media_duration(output_path)
            results.append(
                CaptionResult(
                    output_path=output_path,
                    duration=duration,
                    segments_count=0,
                )
            )
    return results


def generate_tts(
    text: str,
    profile: str | None = None,
    model_id: str | None = None,
    voice: str | None = None,
    reference_audio: str | Path | None = None,
    reference_text: str | None = None,
    output_path: str | Path | None = None,
) -> tuple[np.ndarray, int]:
    """Generate speech from text.  Returns (audio_array, sample_rate).

    If *output_path* is provided, also saves to a WAV file.
    """
    _ensure_registry()

    resolved_model, resolved_voice, resolved_ref_audio, resolved_ref_text = (
        _resolve_profile(profile, model_id, voice, reference_audio, reference_text)
    )

    tts_config = CaptionTTSConfig(
        model_id=resolved_model,
        voice=resolved_voice,
        reference_audio=resolved_ref_audio,
        reference_text=resolved_ref_text,
    )

    from tts_tests.caption.tts_bridge import generate_single_caption

    audio, sample_rate = generate_single_caption(text, tts_config)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), audio, sample_rate)
        logger.info("Saved audio to %s", out)

    return audio, sample_rate


def _get_media_duration(path: Path) -> float:
    """Get the duration of a media file in seconds using ffprobe."""
    import json
    import subprocess

    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception:
        logger.warning("Could not determine duration of %s", path)
        return 0.0
