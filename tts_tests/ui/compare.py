"""Side-by-side comparison tab."""

from pathlib import Path

import gradio as gr
import soundfile as sf

from tts_tests import registry
from tts_tests.config import OUTPUT_DIR, get_device


def _get_model_choices() -> list[tuple[str, str]]:
    choices = []
    for info, available in registry.list_models():
        if available:
            label = info.name
            icons = ""
            if info.available_voices:
                icons += "\U0001f5e3 "
            if info.supports_voice_cloning:
                icons += "\U0001f3a4 "
            label = icons + label
            if registry.is_remote(info.model_id):
                label += " (remote)"
            else:
                label += " (local)"
            choices.append((label, info.model_id))
    return choices


def _generate_for_model(model_id: str, text: str, ref_audio_path: str | None, ref_text: str):
    """Generate audio for a single model. Returns (audio_tuple, status)."""
    if not text.strip():
        return None, "No text provided."
    if not model_id:
        return None, "No model selected."

    try:
        device = get_device()
        model = registry.load_model(model_id, device=device)
    except Exception as e:
        return None, f"Load failed: {e}"

    ref_path = Path(ref_audio_path) if ref_audio_path else None

    try:
        result = model.generate(
            text=text,
            reference_audio=ref_path,
            reference_text=ref_text if ref_text and ref_text.strip() else None,
        )
    except Exception as e:
        return None, f"Error: {e}"

    out_path = OUTPUT_DIR / f"compare_{model_id}_latest.wav"
    sf.write(str(out_path), result.audio, result.sample_rate)

    status = (
        f"{result.duration:.1f}s in {result.generation_time:.1f}s "
        f"({result.duration / result.generation_time:.1f}x realtime)"
    )
    return (result.sample_rate, result.audio), status


def _generate_both(model_a: str, model_b: str, text: str, ref_audio: str | None, ref_text: str):
    """Generate sequentially for both models, yielding status updates."""
    yield None, "Loading model A (downloading if needed)...", None, "Waiting..."
    audio_a, status_a = _generate_for_model(model_a, text, ref_audio, ref_text)
    yield audio_a, status_a, None, "Loading model B (downloading if needed)..."
    audio_b, status_b = _generate_for_model(model_b, text, ref_audio, ref_text)
    yield audio_a, status_a, audio_b, status_b


def build_compare_tab():
    choices = _get_model_choices()

    text_input = gr.Textbox(
        label="Text to synthesise",
        placeholder="Enter text here...",
        lines=3,
        value="The quick brown fox jumps over the lazy dog.",
    )

    with gr.Row():
        ref_audio = gr.Audio(
            label="Reference audio (optional, for voice cloning models)",
            type="filepath",
        )
        ref_text = gr.Textbox(
            label="Reference transcript",
            placeholder="Transcript of reference audio...",
        )

    with gr.Row():
        generate_btn = gr.Button("Generate Both", variant="primary", size="lg")
        cancel_btn = gr.Button("Cancel", variant="stop", size="lg", visible=False)

    with gr.Row():
        with gr.Column():
            model_a = gr.Dropdown(
                label="Model A",
                choices=choices,
                value=choices[0][1] if choices else None,
            )
            audio_a = gr.Audio(label="Model A Output", type="numpy")
            status_a = gr.Markdown()

        with gr.Column():
            model_b = gr.Dropdown(
                label="Model B",
                choices=choices,
                value=choices[1][1] if len(choices) > 1 else (choices[0][1] if choices else None),
            )
            audio_b = gr.Audio(label="Model B Output", type="numpy")
            status_b = gr.Markdown()

    gen_event = generate_btn.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[generate_btn, cancel_btn],
    ).then(
        fn=_generate_both,
        inputs=[model_a, model_b, text_input, ref_audio, ref_text],
        outputs=[audio_a, status_a, audio_b, status_b],
    ).then(
        fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[generate_btn, cancel_btn],
    )

    cancel_btn.click(
        fn=lambda: (None, "Cancelled.", None, "Cancelled."),
        outputs=[audio_a, status_a, audio_b, status_b],
        cancels=[gen_event],
    )
