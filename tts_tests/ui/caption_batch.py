"""Batch caption processing tab."""

from pathlib import Path

import gradio as gr

from tts_tests.caption.batch import process_directory
from tts_tests.caption.config import CaptionTTSConfig, Config, OutputConfig
from tts_tests.profiles import get_profile, list_profiles


def _get_profile_choices() -> list[tuple[str, str]]:
    """Return profile choices including a Custom option."""
    profiles = list_profiles()
    choices = [(name, name) for name in profiles]
    choices.append(("Custom...", "__custom__"))
    return choices


def _run_batch(
    input_dir: str,
    pattern: str,
    profile_name: str,
    output_format: str,
    progress=gr.Progress(),
) -> tuple[str, str]:
    """Run batch processing on a directory. Returns (log_text, summary_markdown)."""
    if not input_dir or not input_dir.strip():
        return "", "Please enter an input directory path."

    dir_path = Path(input_dir.strip())
    if not dir_path.is_dir():
        return "", f"Directory not found: `{dir_path}`"

    if not pattern or not pattern.strip():
        pattern = "*.mp4"

    # Build config from profile
    if profile_name == "__custom__":
        return "", (
            "Custom voice settings are not supported in batch mode. "
            "Please save a voice profile first and select it here."
        )

    try:
        profile = get_profile(profile_name)
    except KeyError:
        return "", f"Profile **{profile_name}** not found."

    tts_config = CaptionTTSConfig(
        model_id=profile.model_id,
        voice=profile.voice,
        reference_audio=profile.reference_audio,
        reference_text=profile.reference_text,
    )

    config = Config(
        tts=tts_config,
        output=OutputConfig(format=output_format),
    )

    log_lines: list[str] = []

    def on_file_progress(filename: str, idx: int, total: int):
        msg = f"[{idx}/{total}] Processing {filename}..."
        log_lines.append(msg)
        progress(idx / total, desc=msg)

    try:
        results = process_directory(
            input_dir=dir_path,
            config=config,
            video_pattern=pattern.strip(),
            on_file_progress=on_file_progress,
        )
    except Exception as e:
        return "\n".join(log_lines), f"Batch processing failed: {e}"

    # Build results log
    for r in results:
        video_name = Path(r["video"]).name if r["video"] else "unknown"
        if r["success"]:
            out_name = Path(r["output"]).name if r["output"] else "unknown"
            log_lines.append(f"  Completed: {video_name} -> {out_name}")
        else:
            log_lines.append(f"  FAILED: {video_name} — {r['error']}")

    succeeded = sum(1 for r in results if r["success"])
    failed = len(results) - succeeded

    summary_parts = [f"**Batch complete:** {succeeded}/{len(results)} succeeded"]
    if failed > 0:
        summary_parts.append(f", {failed} failed")
    summary = "".join(summary_parts) + "."

    if not results:
        summary = (
            f"No video+SRT pairs found in `{dir_path}` "
            f"matching pattern `{pattern.strip()}`."
        )

    return "\n".join(log_lines), summary


def build_caption_batch_tab():
    profile_choices = _get_profile_choices()

    gr.Markdown("### Batch caption-to-speech processing")
    gr.Markdown(
        "Process all video files in a directory that have matching SRT files. "
        "Each video must have a `.srt` file with the same name alongside it."
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_dir = gr.Textbox(
                label="Input directory",
                placeholder="/path/to/videos/",
            )
            browse_file = gr.File(
                label="Or pick any file in the target directory",
                file_count="single",
                type="filepath",
            )
            file_pattern = gr.Dropdown(
                label="File pattern",
                choices=["*.mp4", "*.webm", "*.mkv", "*.avi", "*.mov"],
                value="*.mp4",
                allow_custom_value=True,
                interactive=True,
            )
            profile_dropdown = gr.Dropdown(
                label="Voice profile",
                choices=profile_choices,
                value=(
                    profile_choices[0][1]
                    if len(profile_choices) > 1
                    else "__custom__"
                ),
                interactive=True,
            )
            output_format = gr.Dropdown(
                label="Output format",
                choices=["mkv", "mp4", "webm"],
                value="mkv",
                interactive=True,
            )
            start_btn = gr.Button("Start Batch", variant="primary", size="lg")

        with gr.Column(scale=2):
            progress_log = gr.Textbox(
                label="Progress log",
                lines=15,
                interactive=False,
            )
            results_summary = gr.Markdown()

    # Auto-fill directory from file picker
    def _dir_from_file(filepath):
        if filepath:
            return str(Path(filepath).parent)
        return gr.update()

    browse_file.change(
        fn=_dir_from_file,
        inputs=[browse_file],
        outputs=[input_dir],
    )

    start_btn.click(
        fn=_run_batch,
        inputs=[input_dir, file_pattern, profile_dropdown, output_format],
        outputs=[progress_log, results_summary],
    )

    return profile_dropdown
