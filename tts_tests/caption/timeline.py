from tts_tests.caption.config import CaptionTTSConfig


def _snap_to_frame(duration: float, fps: int) -> float:
    """Round a duration to the nearest frame boundary."""
    return round(duration * fps) / fps


def build_render_timeline(
    segments: list[dict], config: CaptionTTSConfig, fps: int = 25,
) -> list[dict]:
    render_entries = []
    current_output_time = 0.0

    for segment in segments:
        entry = {**segment}
        # Snap start and end individually to frame boundaries so the trim
        # filter in ffmpeg produces exactly the expected number of frames.
        # Previously we snapped only the *difference*, which could diverge
        # from ffmpeg's actual frame count by ±1 frame per segment,
        # accumulating into noticeable audio/video drift.
        render_start = _snap_to_frame(segment["start"], fps)
        render_end = _snap_to_frame(segment["end"], fps)
        screen_dur = round(render_end - render_start, 6)
        if screen_dur <= 0:
            screen_dur = 1.0 / fps
            render_end = render_start + screen_dur

        entry["render_start"] = render_start
        entry["render_end"] = render_end

        if not segment["has_speech"]:
            entry["output_duration"] = screen_dur
            entry["freeze_duration"] = 0
            entry["audio_offset"] = None
        elif segment["padded_duration"] <= screen_dur:
            entry["output_duration"] = screen_dur
            entry["freeze_duration"] = 0
            entry["audio_offset"] = round(current_output_time + config.silence_before, 3)
        else:
            freeze = _snap_to_frame(
                segment["padded_duration"] - screen_dur, fps,
            )
            entry["output_duration"] = screen_dur + freeze
            entry["freeze_duration"] = freeze
            entry["audio_offset"] = round(current_output_time + config.silence_before, 3)

        entry["output_start"] = round(current_output_time, 3)
        current_output_time += entry["output_duration"]
        render_entries.append(entry)

    return render_entries
