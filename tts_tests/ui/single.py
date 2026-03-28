"""Single-model generation tab."""

from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from tts_tests import registry
from tts_tests.config import OUTPUT_DIR, get_device


def _get_model_choices() -> list[tuple[str, str]]:
    """Return (display_label, model_id) for the dropdown."""
    choices = []
    for info, available in registry.list_models():
        label = info.name
        icons = ""
        if info.available_voices:
            icons += "\U0001f5e3 "  # speaking head = preset voices
        if info.supports_voice_cloning:
            icons += "\U0001f3a4 "  # microphone = voice cloning
        label = icons + label  # microphone emoji = supports cloning
        if not available:
            label += " (not installed)"
        elif registry.is_remote(info.model_id):
            label += " (remote)"
        else:
            label += " (local)"
        choices.append((label, info.model_id))
    return choices


def _get_model_info(model_id: str) -> str:
    """Return formatted info about the selected model."""
    for info, available in registry.list_models():
        if info.model_id == model_id:
            if not available:
                status = "**Not installed**"
            elif registry.is_remote(info.model_id):
                endpoint = registry.get_endpoint(info.model_id)
                status = f"Remote ({endpoint['url']})"
            else:
                status = "Local"
            voices = ", ".join(info.available_voices[:10]) if info.available_voices else "None (use reference audio)"
            return (
                f"**{info.name}** — {info.description}\n\n"
                f"- Status: {status}\n"
                f"- Voice cloning: {'Yes' if info.supports_voice_cloning else 'No'}\n"
                f"- Preset voices: {voices}\n"
                f"- Est. VRAM: {info.estimated_vram_gb:.1f} GB\n"
                f"- Sample rate: {info.native_sample_rate} Hz"
            )
    return ""


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


def _generate(
    model_id: str,
    text: str,
    voice: str | None,
    ref_audio: str | None,
    ref_text: str,
):
    """Generate audio, yielding status updates."""
    if not text.strip():
        yield None, "Please enter some text."
        return

    yield None, "Loading model (downloading if needed)..."

    try:
        device = get_device()
        model = registry.load_model(model_id, device=device)
    except Exception as e:
        yield None, f"Failed to load model: {e}"
        return

    yield None, "Generating audio..."

    ref_path = Path(ref_audio) if ref_audio else None

    try:
        result = model.generate(
            text=text,
            voice=voice if voice else None,
            reference_audio=ref_path,
            reference_text=ref_text if ref_text and ref_text.strip() else None,
        )
    except Exception as e:
        yield None, f"Generation failed: {e}"
        return

    # Save to file
    out_path = OUTPUT_DIR / f"{model_id}_latest.wav"
    sf.write(str(out_path), result.audio, result.sample_rate)

    status = (
        f"Generated {result.duration:.1f}s of audio in {result.generation_time:.1f}s "
        f"({result.duration / result.generation_time:.1f}x realtime) "
        f"@ {result.sample_rate} Hz"
    )

    yield (result.sample_rate, result.audio), status


def build_playground_tab():
    """Build the playground tab. Returns (save_profile_btn, model, voice, ref_audio, ref_text)
    so main.py can wire the 'Save as Profile' button to the profiles tab."""
    choices = _get_model_choices()

    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                label="Model",
                choices=choices,
                value=choices[0][1] if choices else None,
                interactive=True,
                elem_id="pg_model",
            )
            model_info = gr.Markdown(
                value=_get_model_info(choices[0][1]) if choices else "",
            )
            voice_dropdown = gr.Dropdown(
                label="Voice preset",
                choices=[],
                visible=False,
                interactive=True,
                elem_id="pg_voice",
            )
            ref_audio_group = gr.Group(visible=False)
            with ref_audio_group:
                ref_audio = gr.Audio(
                    label="Reference audio (for voice cloning)",
                    type="filepath",
                )
                ref_text = gr.Textbox(
                    label="Reference audio transcript",
                    placeholder="Transcript of the reference audio...",
                )

        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Text to synthesise",
                placeholder="Enter text here...",
                lines=5,
                value="The quick brown fox jumps over the lazy dog.",
            )
            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary", size="lg")
                cancel_btn = gr.Button("Cancel", variant="stop", size="lg", visible=False)
                unload_btn = gr.Button("Unload Model", variant="secondary", size="sm")
                save_profile_btn = gr.Button("Send to Profile \u2192", variant="secondary", size="sm")
            audio_output = gr.Audio(label="Output", type="numpy")
            status_output = gr.Markdown()

    # Wire events
    model_dropdown.change(
        fn=_get_model_info,
        inputs=[model_dropdown],
        outputs=[model_info],
    )
    model_dropdown.change(
        fn=_get_voice_choices,
        inputs=[model_dropdown],
        outputs=[voice_dropdown],
    )
    model_dropdown.change(
        fn=_show_ref_audio,
        inputs=[model_dropdown],
        outputs=[ref_audio_group],
    )

    gen_event = generate_btn.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[generate_btn, cancel_btn],
    ).then(
        fn=_generate,
        inputs=[model_dropdown, text_input, voice_dropdown, ref_audio, ref_text],
        outputs=[audio_output, status_output],
    ).then(
        fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[generate_btn, cancel_btn],
    )

    cancel_btn.click(
        fn=lambda: (None, "Cancelled."),
        outputs=[audio_output, status_output],
        cancels=[gen_event],
    )

    def _unload():
        registry.unload_current()
        return "Model unloaded, GPU memory freed."

    unload_btn.click(fn=_unload, outputs=[status_output])

    # Initialise voice dropdown for default selection
    if choices:
        model_dropdown.change(
            fn=_get_voice_choices,
            inputs=[model_dropdown],
            outputs=[voice_dropdown],
        )

    # Return components for cross-tab wiring in main.py
    return save_profile_btn, model_dropdown, voice_dropdown, ref_audio, ref_text
