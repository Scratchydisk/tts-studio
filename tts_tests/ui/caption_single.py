"""Single video caption processing tab."""

from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

from tts_tests import registry
from tts_tests.caption.config import CaptionTTSConfig, Config, OutputConfig
from tts_tests.caption.pipeline import run_pipeline
from tts_tests.caption.srt_parser import parse_srt_file
from tts_tests.caption.tts_bridge import generate_single_caption
from tts_tests.config import OUTPUT_DIR, get_device
from tts_tests.profiles import VoiceProfile, get_profile, list_profiles
from tts_tests.ui.shared import get_model_choices


def _get_profile_choices() -> list[tuple[str, str]]:
    """Return profile choices including a Custom option."""
    profiles = list_profiles()
    choices = [(name, name) for name in profiles]
    choices.append(("Custom...", "__custom__"))
    return choices


def _get_voice_choices(model_id: str) -> gr.update:
    """Update voice dropdown based on selected model."""
    for info, _avail in registry.list_models():
        if info.model_id == model_id:
            if info.available_voices:
                return gr.update(
                    choices=info.available_voices,
                    value=info.available_voices[0],
                    visible=True,
                )
            else:
                return gr.update(choices=[], value=None, visible=False)
    return gr.update(choices=[], value=None, visible=False)


def _show_ref_audio(model_id: str) -> gr.update:
    """Show/hide reference audio controls based on model capabilities."""
    for info, _avail in registry.list_models():
        if info.model_id == model_id:
            return gr.update(visible=info.supports_voice_cloning)
    return gr.update(visible=False)


def _on_profile_change(profile_name: str) -> tuple:
    """Show/hide custom controls based on profile selection.

    Returns updates for (custom_group, custom_model, custom_voice,
    custom_ref_group).
    """
    if profile_name == "__custom__":
        return (
            gr.update(visible=True),   # custom_group
            gr.update(),               # custom_model (keep current)
            gr.update(),               # custom_voice (keep current)
            gr.update(),               # custom_ref_group (keep current)
        )
    return (
        gr.update(visible=False),  # custom_group
        gr.update(),               # custom_model
        gr.update(),               # custom_voice
        gr.update(),               # custom_ref_group
    )


def _parse_srt(srt_file) -> pd.DataFrame:
    """Parse an SRT file and return a DataFrame for display."""
    if srt_file is None:
        return pd.DataFrame(columns=["Index", "Start", "End", "Text"])

    srt_path = Path(srt_file)
    if not srt_path.exists():
        return pd.DataFrame(columns=["Index", "Start", "End", "Text"])

    segments = parse_srt_file(srt_path)
    rows = []
    for seg in segments:
        rows.append([
            seg["index"],
            f"{seg['start']:.3f}",
            f"{seg['end']:.3f}",
            seg["text"],
        ])
    return pd.DataFrame(rows, columns=["Index", "Start", "End", "Text"])


def _build_tts_config(
    profile_name: str,
    custom_model: str | None,
    custom_voice: str | None,
    custom_ref_audio: str | None,
    custom_ref_text: str,
    silence_before: float,
    silence_after: float,
) -> CaptionTTSConfig:
    """Build a CaptionTTSConfig from either a profile or custom settings."""
    if profile_name != "__custom__":
        try:
            profile = get_profile(profile_name)
            return CaptionTTSConfig(
                model_id=profile.model_id,
                voice=profile.voice,
                reference_audio=profile.reference_audio,
                reference_text=profile.reference_text,
                silence_before=silence_before,
                silence_after=silence_after,
            )
        except KeyError:
            raise ValueError(f"Profile '{profile_name}' not found.")

    if not custom_model:
        raise ValueError("Please select a model.")

    return CaptionTTSConfig(
        model_id=custom_model,
        voice=custom_voice if custom_voice else None,
        reference_audio=custom_ref_audio if custom_ref_audio else None,
        reference_text=(
            custom_ref_text if custom_ref_text and custom_ref_text.strip() else None
        ),
        silence_before=silence_before,
        silence_after=silence_after,
    )


def _get_intermediates_dir(video_file) -> Path | None:
    """Get the intermediates directory for the current video."""
    if video_file is None:
        return None
    video_path = Path(video_file)
    return video_path.parent / f"{video_path.stem}_intermediates"


def _preview_caption(
    profile_name: str,
    custom_model: str | None,
    custom_voice: str | None,
    custom_ref_audio: str | None,
    custom_ref_text: str,
    caption_table: pd.DataFrame,
    caption_index: int,
    video_file,
    pin_preview: bool,
) -> tuple:
    """Preview audio for a single caption row.

    If pin_preview is True and a video file is loaded, saves the audio to
    the intermediates directory so it will be reused during rendering.

    Returns (audio_tuple, status_text).
    """
    if caption_table is None or caption_table.empty:
        return None, "No captions loaded. Please parse an SRT file first."

    try:
        idx = int(caption_index)
    except (TypeError, ValueError):
        return None, "Please enter a valid caption index."

    if idx < 0 or idx >= len(caption_table):
        return None, f"Index {idx} is out of range (0-{len(caption_table) - 1})."

    text = str(caption_table.iloc[idx]["Text"])
    if not text.strip():
        return None, "Caption text is empty."

    try:
        tts_config = _build_tts_config(
            profile_name, custom_model, custom_voice,
            custom_ref_audio, custom_ref_text,
            silence_before=0.0, silence_after=0.0,
        )
    except ValueError as e:
        return None, str(e)

    try:
        audio, sr = generate_single_caption(text, tts_config)
    except Exception as e:
        return None, f"Generation failed: {e}"

    status = f"Preview for caption {idx}: \"{text[:60]}...\""

    # Pin to intermediates if requested
    if pin_preview and video_file is not None:
        intermediates = _get_intermediates_dir(video_file)
        if intermediates:
            intermediates.mkdir(parents=True, exist_ok=True)
            wav_path = intermediates / f"speech_{idx:03d}.wav"
            import soundfile as sf
            sf.write(str(wav_path), audio, sr)
            status += f" **Pinned** — will be used in render."

    return (sr, audio), status


def _unpin_caption(video_file, caption_index: int) -> str:
    """Remove a pinned preview so it will be regenerated during render."""
    if video_file is None:
        return "No video file loaded."
    try:
        idx = int(caption_index)
    except (TypeError, ValueError):
        return "Invalid caption index."

    intermediates = _get_intermediates_dir(video_file)
    if intermediates:
        wav_path = intermediates / f"speech_{idx:03d}.wav"
        if wav_path.exists():
            wav_path.unlink()
            return f"Caption {idx} unpinned — will be regenerated during render."
    return f"Caption {idx} was not pinned."


def _render_video(
    video_file,
    srt_file,
    profile_name: str,
    custom_model: str | None,
    custom_voice: str | None,
    custom_ref_audio: str | None,
    custom_ref_text: str,
    silence_before: float,
    silence_after: float,
    output_format: str,
    progress=gr.Progress(),
) -> tuple:
    """Render the captioned video. Returns (output_path, status_text)."""
    if video_file is None:
        return None, "Please upload a video file."
    if srt_file is None:
        return None, "Please upload an SRT file."

    video_path = Path(video_file)
    srt_path = Path(srt_file)

    try:
        tts_config = _build_tts_config(
            profile_name, custom_model, custom_voice,
            custom_ref_audio, custom_ref_text,
            silence_before, silence_after,
        )
    except ValueError as e:
        return None, str(e)

    config = Config(
        tts=tts_config,
        output=OutputConfig(format=output_format),
    )

    output_path = OUTPUT_DIR / f"{video_path.stem}_narrated.{output_format}"

    def on_progress(stage: str, current: int, total: int):
        progress(current / total, desc=f"Stage: {stage} ({current}/{total})")

    try:
        result_path = run_pipeline(
            video_path=video_path,
            srt_path=srt_path,
            config=config,
            output_path=output_path,
            on_progress=on_progress,
        )
    except Exception as e:
        return None, f"Rendering failed: {e}"

    return str(result_path), f"Rendering complete. Output saved to `{result_path}`"


def _save_srt(caption_table: pd.DataFrame, srt_file) -> str:
    """Save edited captions back to the SRT file."""
    if caption_table is None or caption_table.empty:
        return "No captions to save."
    if srt_file is None:
        return "No SRT file loaded."

    srt_path = Path(srt_file)

    def _fmt_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for i, row in caption_table.iterrows():
        idx = int(row["Index"]) + 1  # SRT is 1-indexed
        start = float(row["Start"])
        end = float(row["End"])
        text = str(row["Text"])
        lines.append(str(idx))
        lines.append(f"{_fmt_time(start)} --> {_fmt_time(end)}")
        lines.append(text)
        lines.append("")

    srt_path.write_text("\n".join(lines), encoding="utf-8")
    return f"Captions saved to `{srt_path.name}`."


def build_caption_single_tab():
    profile_choices = _get_profile_choices()
    model_choices = get_model_choices()

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("Voice Settings", open=True):
                profile_dropdown = gr.Dropdown(
                    label="Voice profile",
                    choices=profile_choices,
                    value=profile_choices[0][1] if len(profile_choices) > 1 else "__custom__",
                    interactive=True,
                    info="Select a saved voice profile, or choose Custom to configure manually.",
                )

                custom_group = gr.Group(
                    visible=(len(profile_choices) <= 1),
                )
                with custom_group:
                    custom_model = gr.Dropdown(
                        label="Model",
                        choices=model_choices,
                        value=model_choices[0][1] if model_choices else None,
                        interactive=True,
                    )
                    custom_voice = gr.Dropdown(
                        label="Voice preset",
                        choices=[],
                        visible=False,
                        interactive=True,
                    )
                    custom_ref_group = gr.Group(visible=False)
                    with custom_ref_group:
                        custom_ref_audio = gr.Audio(
                            label="Reference audio (for voice cloning)",
                            type="filepath",
                        )
                        custom_ref_text = gr.Textbox(
                            label="Reference audio transcript",
                            placeholder="Transcript of the reference audio...",
                        )

            with gr.Accordion("Output Settings", open=False):
                silence_before = gr.Slider(
                    label="Silence before caption (seconds)",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    value=0.3,
                    info="Padding added before each speech segment.",
                )
                silence_after = gr.Slider(
                    label="Silence after caption (seconds)",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    value=0.5,
                    info="Padding added after each speech segment.",
                )
                output_format = gr.Dropdown(
                    label="Output format",
                    choices=["mkv", "mp4", "webm"],
                    value="mkv",
                    interactive=True,
                )

        with gr.Column(scale=2):
            gr.Markdown("### Input files")
            video_input = gr.File(
                label="Video file — the video to narrate",
                file_types=["video"],
            )
            srt_input = gr.File(
                label="SRT subtitle file — SubRip file with caption timing",
                file_types=[".srt"],
            )
            parse_btn = gr.Button("Parse SRT")
            caption_table = gr.Dataframe(
                headers=["Index", "Start", "End", "Text"],
                interactive=True,
                label="Captions — click a row to select it, edit text directly",
                column_widths=["60px", "70px", "70px", None],
            )
            with gr.Row():
                save_srt_btn = gr.Button(
                    "Save Captions to SRT",
                    variant="secondary",
                    size="sm",
                )
            save_srt_status = gr.Markdown()

            with gr.Accordion("Preview & Pin", open=False):
                with gr.Row():
                    preview_index = gr.Number(
                        label="Caption index",
                        value=0,
                        precision=0,
                    )
                    pin_checkbox = gr.Checkbox(
                        label="Pin for render",
                        value=True,
                        info="Pin this take so it won't be regenerated during render.",
                    )
                with gr.Row():
                    preview_btn = gr.Button("Preview Caption", variant="primary")
                    unpin_btn = gr.Button("Unpin", variant="secondary", size="sm")
                preview_audio = gr.Audio(label="Caption preview", type="numpy")
                preview_status = gr.Markdown()

            with gr.Accordion("Speech tips", open=False):
                gr.Markdown(
                    "**Controlling pauses and style:**\n\n"
                    "Most TTS models respond to natural punctuation:\n"
                    "- **Commas** `,` — short pause\n"
                    "- **Periods** `.` — longer pause\n"
                    "- **Ellipsis** `...` — drawn-out pause\n"
                    "- **Dashes** `—` or `--` — mid-sentence break\n"
                    "- **Question marks / exclamation** — affects intonation\n\n"
                    "**Model-specific features:**\n\n"
                    "- **Dia-1.6B:** Speaker tags `[S1]`, `[S2]` for dialogue. "
                    "Emotion markers like `(laughs)`, `(sighs)` in the text.\n"
                    "- **Chatterbox:** Supports emotion exaggeration control.\n"
                    "- **Zonos:** Fine-grained prosody via 10–30s reference audio.\n"
                    "- **Voice cloning models** (🎤): Match the speaking style "
                    "of your reference audio — a slow, calm reference produces "
                    "slow, calm output.\n\n"
                    "**General tips:**\n"
                    "- Split long sentences across multiple captions\n"
                    "- Use short, clear sentences for best results\n"
                    "- Preview individual captions before rendering"
                )

            gr.Markdown("### Render")
            render_btn = gr.Button(
                "Render Video", variant="primary", size="lg",
            )
            render_status = gr.Markdown()
            output_video = gr.Video(label="Output video")

    # Wire events — profile selection
    profile_dropdown.change(
        fn=_on_profile_change,
        inputs=[profile_dropdown],
        outputs=[custom_group, custom_model, custom_voice, custom_ref_group],
    )

    # Wire events — custom model selection
    custom_model.change(
        fn=_get_voice_choices,
        inputs=[custom_model],
        outputs=[custom_voice],
    )
    custom_model.change(
        fn=_show_ref_audio,
        inputs=[custom_model],
        outputs=[custom_ref_group],
    )

    # Parse SRT
    parse_btn.click(
        fn=_parse_srt,
        inputs=[srt_input],
        outputs=[caption_table],
    )

    # Save edited captions
    save_srt_btn.click(
        fn=_save_srt,
        inputs=[caption_table, srt_input],
        outputs=[save_srt_status],
    )

    # Table row select → set caption index
    def _on_caption_select(evt: gr.SelectData):
        row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        return row_idx

    caption_table.select(
        fn=_on_caption_select,
        outputs=[preview_index],
    )

    # Preview caption
    preview_btn.click(
        fn=_preview_caption,
        inputs=[
            profile_dropdown, custom_model, custom_voice,
            custom_ref_audio, custom_ref_text,
            caption_table, preview_index,
            video_input, pin_checkbox,
        ],
        outputs=[preview_audio, preview_status],
    )

    # Unpin caption
    unpin_btn.click(
        fn=_unpin_caption,
        inputs=[video_input, preview_index],
        outputs=[preview_status],
    )

    # Render video
    render_btn.click(
        fn=_render_video,
        inputs=[
            video_input, srt_input,
            profile_dropdown, custom_model, custom_voice,
            custom_ref_audio, custom_ref_text,
            silence_before, silence_after, output_format,
        ],
        outputs=[output_video, render_status],
    )

    return profile_dropdown
