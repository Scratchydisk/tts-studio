import re
from pathlib import Path


def parse_timecode(tc: str) -> float:
    """Parse SRT timecode '00:01:23,456' to seconds."""
    match = re.match(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})", tc.strip())
    if not match:
        raise ValueError(f"Invalid SRT timecode: '{tc}'")
    h, m, s, ms = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def format_timecode(seconds: float) -> str:
    """Convert seconds to SRT timecode '00:01:23,456'."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    if ms >= 1000:
        ms = 0
        s += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt_file(srt_path: Path) -> list[dict]:
    """Parse an SRT file into caption segments.

    Returns list of dicts with keys: index, text, start, end
    (same format as captions.json from the old OCR stage).
    """
    text = srt_path.read_text(encoding="utf-8-sig")  # handles BOM
    blocks = re.split(r"\n\s*\n", text.strip())

    segments = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue

        # First line is the index (skip it, we re-index)
        # Second line is the timecode
        tc_match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})",
            lines[1].strip(),
        )
        if not tc_match:
            continue

        start = parse_timecode(tc_match.group(1))
        end = parse_timecode(tc_match.group(2))
        caption_text = "\n".join(lines[2:]).strip()

        if not caption_text:
            continue

        segments.append({
            "index": len(segments),
            "text": caption_text,
            "start": round(start, 3),
            "end": round(end, 3),
        })

    return segments


def generate_adjusted_srt(render_timeline: list[dict], output_path: Path) -> None:
    """Write a new SRT file with timing adjusted for freeze frames.

    Uses output_start and output_duration from the render timeline to remap
    original SRT timing to the stretched output video.
    """
    lines = []
    srt_index = 1

    for entry in render_timeline:
        text = entry.get("text", "")
        if not text:
            continue

        start = entry["output_start"]
        end = start + entry["output_duration"]
        lines.append(str(srt_index))
        lines.append(f"{format_timecode(start)} --> {format_timecode(end)}")
        lines.append(text)
        lines.append("")
        srt_index += 1

    output_path.write_text("\n".join(lines), encoding="utf-8")
