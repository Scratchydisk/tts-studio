import subprocess
import wave
from pathlib import Path


def build_audio_track(
    timeline: list[dict],
    intermediates_dir: Path,
    output_path: Path,
    total_duration: float,
) -> None:
    """Build a single WAV audio track by placing speech clips at their offsets."""
    sample_rate = 22050
    sample_width = 2
    channels = 1

    for entry in timeline:
        if entry["has_speech"]:
            wav_path = intermediates_dir / f"speech_{entry['index']:03d}.wav"
            with wave.open(str(wav_path), "rb") as wf:
                sample_rate = wf.getframerate()
                sample_width = wf.getsampwidth()
                channels = wf.getnchannels()
            break

    total_frames = int(total_duration * sample_rate)
    audio_buffer = bytearray(total_frames * sample_width * channels)

    for entry in timeline:
        if not entry["has_speech"]:
            continue

        wav_path = intermediates_dir / f"speech_{entry['index']:03d}.wav"
        with wave.open(str(wav_path), "rb") as wf:
            speech_data = wf.readframes(wf.getnframes())

        offset_frames = int(entry["audio_offset"] * sample_rate)
        offset_bytes = offset_frames * sample_width * channels
        end_bytes = offset_bytes + len(speech_data)

        if end_bytes > len(audio_buffer):
            speech_data = speech_data[: len(audio_buffer) - offset_bytes]
            end_bytes = len(audio_buffer)

        audio_buffer[offset_bytes:end_bytes] = speech_data

    with wave.open(str(output_path), "w") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(audio_buffer))


def build_video_filter(timeline: list[dict], fps: int = 25) -> str:
    """Build an ffmpeg filter_complex string for the video pass."""
    parts = []
    labels = []

    for i, entry in enumerate(timeline):
        label = f"v{i}"
        start = entry["start"]
        end = entry["end"]
        trim = f"[0:v]trim={start}:{end},setpts=PTS-STARTPTS"

        if entry["freeze_duration"] > 0:
            trim += f",fps={fps},tpad=stop_duration={entry['freeze_duration']}:stop_mode=clone"

        parts.append(f"{trim}[{label}]")
        labels.append(f"[{label}]")

    n = len(labels)
    concat = f"{''.join(labels)}concat=n={n}:v=1:a=0[outv]"
    parts.append(concat)

    return ";\n".join(parts)


def render_output(
    video_path: Path,
    audio_path: Path,
    filter_str: str,
    output_path: Path,
    output_format: str = "mkv",
    subtitle_path: Path | None = None,
) -> None:
    """Render the final output by muxing filtered video with audio.

    When subtitle_path is provided and format is mkv, embeds the SRT
    as a soft subtitle stream in the output.
    """
    if output_format == "mkv":
        video_codec = ["-c:v", "libx264", "-preset", "medium", "-crf", "23"]
        audio_codec = ["-c:a", "aac", "-b:a", "128k"]
    elif output_format == "mp4":
        video_codec = ["-c:v", "libx264", "-preset", "medium", "-crf", "23"]
        audio_codec = ["-c:a", "aac", "-b:a", "128k"]
    elif output_format == "webm":
        video_codec = ["-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0"]
        audio_codec = ["-c:a", "libopus", "-b:a", "128k"]
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    inputs = ["-i", str(video_path), "-i", str(audio_path)]
    maps = ["-map", "[outv]", "-map", "1:a"]
    subtitle_codec = []

    if subtitle_path and output_format == "mkv":
        inputs.extend(["-i", str(subtitle_path)])
        maps.extend(["-map", "2:s"])
        subtitle_codec = ["-c:s", "srt", "-metadata:s:s:0", "language=eng"]

    cmd = [
        "ffmpeg",
        *inputs,
        "-filter_complex", filter_str,
        *maps,
        *video_codec,
        *audio_codec,
        *subtitle_codec,
        str(output_path),
        "-y",
    ]

    subprocess.run(cmd, check=True, capture_output=True)
