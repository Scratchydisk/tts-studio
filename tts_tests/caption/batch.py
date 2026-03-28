"""Batch processing for caption-to-speech pipeline."""

import logging
from pathlib import Path
from typing import Callable

from tts_tests.caption.config import Config
from tts_tests.caption.pipeline import run_pipeline

logger = logging.getLogger(__name__)


def find_video_srt_pairs(
    input_dir: Path,
    video_pattern: str = "*.mp4",
) -> list[tuple[Path, Path]]:
    """Find video files with matching SRT files in a directory.

    Returns list of (video_path, srt_path) tuples.
    Only includes videos that have a matching .srt file (same stem).
    """
    pairs: list[tuple[Path, Path]] = []
    for video_path in sorted(input_dir.glob(video_pattern)):
        srt_path = video_path.with_suffix(".srt")
        if srt_path.exists():
            pairs.append((video_path, srt_path))
        else:
            logger.debug(
                "Skipping %s — no matching SRT file found", video_path.name,
            )
    return pairs


def process_directory(
    input_dir: Path,
    config: Config,
    video_pattern: str = "*.mp4",
    keep_intermediates: bool = False,
    on_file_progress: Callable[[str, int, int], None] | None = None,
) -> list[dict]:
    """Process all video+SRT pairs in a directory.

    Args:
        input_dir: Directory containing video and SRT files
        config: Pipeline configuration
        video_pattern: Glob pattern for video files (e.g. "*.mp4", "*.webm")
        keep_intermediates: Keep intermediate files after rendering
        on_file_progress: Callback(filename, file_index, total_files)

    Returns:
        List of dicts with keys: video, srt, output, success, error
    """
    pairs = find_video_srt_pairs(input_dir, video_pattern)
    if not pairs:
        logger.warning("No video+SRT pairs found in %s", input_dir)
        return []

    total = len(pairs)
    logger.info("Found %d video+SRT pair(s) to process", total)
    results: list[dict] = []

    for idx, (video_path, srt_path) in enumerate(pairs, start=1):
        if on_file_progress is not None:
            on_file_progress(video_path.name, idx, total)

        logger.info(
            "[%d/%d] Processing %s", idx, total, video_path.name,
        )

        entry: dict = {
            "video": video_path,
            "srt": srt_path,
            "output": None,
            "success": False,
            "error": None,
        }

        try:
            output = run_pipeline(
                video_path=video_path,
                srt_path=srt_path,
                config=config,
                keep_intermediates=keep_intermediates,
            )
            entry["output"] = output
            entry["success"] = True
            logger.info("Completed: %s -> %s", video_path.name, output.name)
        except Exception as exc:
            entry["error"] = str(exc)
            logger.error(
                "Failed to process %s: %s", video_path.name, exc,
            )

        results.append(entry)

    succeeded = sum(1 for r in results if r["success"])
    logger.info(
        "Batch complete: %d/%d succeeded", succeeded, total,
    )
    return results
