"""Caption-to-speech pipeline orchestration.

Adapted from cap_to_speech.pipeline for the tts_tests project.
"""

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from tts_tests.caption.config import Config, load_config
from tts_tests.caption.srt_parser import generate_adjusted_srt, parse_srt_file
from tts_tests.caption.tts_bridge import generate_speech_for_segments
from tts_tests.caption.timeline import build_render_timeline
from tts_tests.caption.render import build_audio_track, build_video_filter, render_output

logger = logging.getLogger(__name__)

STAGES = ["parse", "tts", "timeline", "render"]


def check_ffmpeg() -> None:
    """Verify ffmpeg is available on the system.

    Raises RuntimeError if ffmpeg cannot be found.
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install it from https://ffmpeg.org/download.html "
            "and ensure it is on your PATH."
        )


def _find_srt(video_path: Path) -> Path | None:
    """Auto-discover an SRT file alongside the video."""
    candidate = video_path.with_suffix(".srt")
    if candidate.exists():
        return candidate
    return None


def _default_intermediates_dir(video_path: Path) -> Path:
    """Return the default intermediates directory for a given video."""
    return video_path.parent / f"{video_path.stem}_intermediates"


def _get_fps(video_path: Path) -> int:
    """Get video FPS using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    for stream in data["streams"]:
        if stream["codec_type"] == "video":
            r_frame_rate = stream["r_frame_rate"]
            num, den = r_frame_rate.split("/")
            return int(int(num) / int(den))
    return 25  # fallback


def run_pipeline(
    video_path: Path,
    srt_path: Path | None = None,
    config: Config | None = None,
    output_path: Path | None = None,
    from_stage: str | None = None,
    keep_intermediates: bool = False,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> Path:
    """Run the caption-to-speech pipeline. Returns the output video path."""
    check_ffmpeg()

    if config is None:
        config = load_config()

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    # Validate stage
    if from_stage is not None and from_stage not in STAGES:
        raise ValueError(
            f"Unknown stage: '{from_stage}'. Valid stages: {STAGES}"
        )

    if from_stage:
        start_idx = STAGES.index(from_stage)
        stages_to_run = STAGES[start_idx:]
    else:
        stages_to_run = list(STAGES)

    # SRT discovery
    if srt_path is None:
        srt_path = _find_srt(video_path)

    intermediates_dir = _default_intermediates_dir(video_path)
    intermediates_dir.mkdir(parents=True, exist_ok=True)

    total_stages = len(stages_to_run)

    def _progress(stage_name: str, current: int) -> None:
        if on_progress is not None:
            on_progress(stage_name, current, total_stages)

    # --- Stage 1: Parse ---
    stage_num = 0
    if "parse" in stages_to_run:
        stage_num += 1
        logger.info("[%d/%d] Parsing SRT subtitles...", stage_num, total_stages)
        _progress("parse", stage_num)

        if srt_path is None:
            raise FileNotFoundError(
                f"No SRT file found. Provide one explicitly or place "
                f"'{video_path.stem}.srt' alongside the video."
            )
        segments = parse_srt_file(srt_path)
        captions_path = intermediates_dir / "captions.json"
        with open(captions_path, "w") as f:
            json.dump(segments, f, indent=2)
    else:
        captions_path = intermediates_dir / "captions.json"
        if not captions_path.exists():
            raise FileNotFoundError(
                f"Cannot resume from stage — missing intermediate file: {captions_path}"
            )
        with open(captions_path) as f:
            segments = json.load(f)

    # --- Stage 2: TTS ---
    if "tts" in stages_to_run:
        stage_num += 1
        logger.info("[%d/%d] Generating speech...", stage_num, total_stages)
        _progress("tts", stage_num)

        timeline = generate_speech_for_segments(
            segments, config.tts, intermediates_dir,
        )
    else:
        timeline_path = intermediates_dir / "timeline.json"
        if not timeline_path.exists():
            raise FileNotFoundError(
                f"Cannot resume from stage — missing intermediate file: {timeline_path}"
            )
        with open(timeline_path) as f:
            timeline = json.load(f)

    fps = _get_fps(video_path)

    # --- Stage 3: Timeline ---
    if "timeline" in stages_to_run:
        stage_num += 1
        logger.info("[%d/%d] Building timeline...", stage_num, total_stages)
        _progress("timeline", stage_num)

        render_timeline = build_render_timeline(timeline, config.tts, fps=fps)
    else:
        render_timeline = timeline

    # --- Stage 4: Render ---
    result_path: Path | None = None
    if "render" in stages_to_run:
        stage_num += 1
        logger.info("[%d/%d] Rendering output...", stage_num, total_stages)
        _progress("render", stage_num)

        total_duration = sum(e["output_duration"] for e in render_timeline)

        audio_path = intermediates_dir / "combined_audio.wav"
        build_audio_track(
            render_timeline, intermediates_dir, audio_path, total_duration,
        )

        filter_str = build_video_filter(render_timeline, fps=fps)

        # Generate adjusted SRT for subtitle embedding
        adjusted_srt = intermediates_dir / "adjusted.srt"
        generate_adjusted_srt(render_timeline, adjusted_srt)

        fmt = config.output.format

        if output_path is None:
            stem = video_path.stem
            parent = video_path.parent
            output_path = parent / f"{stem}_narrated.{fmt}"

        render_output(
            video_path=video_path,
            audio_path=audio_path,
            filter_str=filter_str,
            output_path=output_path,
            output_format=fmt,
            subtitle_path=adjusted_srt if fmt == "mkv" else None,
        )

        result_path = output_path
        logger.info("Output: %s", result_path)

    if not keep_intermediates:
        shutil.rmtree(intermediates_dir, ignore_errors=True)

    if result_path is None:
        raise RuntimeError(
            "Pipeline completed but no output file was generated "
            "(render stage was not included)."
        )

    return result_path
