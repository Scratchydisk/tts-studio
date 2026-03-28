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
        screen_dur = _snap_to_frame(segment["screen_duration"], fps)

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
