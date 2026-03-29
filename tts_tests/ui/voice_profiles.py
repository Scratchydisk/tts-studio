"""Voice profile management tab."""

from pathlib import Path

import gradio as gr
import soundfile as sf

from tts_tests import registry
from tts_tests.config import OUTPUT_DIR, get_device
from tts_tests.profiles import (
    VoiceProfile,
    delete_profile,
    get_profile,
    list_profiles,
    save_profile,
)
from tts_tests.ui.shared import get_model_choices


def _get_voice_choices(model_id: str) -> gr.update:
    """Update voice dropdown based on selected model."""
    for info, _avail in registry.list_models():
        if info.model_id == model_id:
            if info.available_voices:
                return gr.update(
                    choices=info.available_voices,
                    value=info.available_voices[0],
                    visible=True,
                    interactive=True,
                )
            else:
                return gr.update(
                    choices=["(no presets — use reference audio)"],
                    value=None,
                    visible=True,
                    interactive=False,
                )
    return gr.update(
        choices=["(no presets — use reference audio)"],
        value=None,
        visible=True,
        interactive=False,
    )


def _show_ref_audio(model_id: str) -> gr.update:
    """Show/hide reference audio controls based on model capabilities."""
    for info, _avail in registry.list_models():
        if info.model_id == model_id:
            return gr.update(visible=info.supports_voice_cloning)
    return gr.update(visible=False)


def _test_voice(
    model_id: str,
    voice: str | None,
    ref_audio: str | None,
    ref_text: str,
    text: str,
):
    """Generate a test audio clip, yielding status updates."""
    if not text.strip():
        yield None, "Please enter some text."
        return

    if not model_id:
        yield None, "Please select a model."
        return

    yield None, "Loading model (downloading if needed)..."

    try:
        device = get_device(model_id)
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

    out_path = OUTPUT_DIR / f"profile_test_{model_id}_latest.wav"
    sf.write(str(out_path), result.audio, result.sample_rate)

    status = (
        f"Generated {result.duration:.1f}s of audio in "
        f"{result.generation_time:.1f}s "
        f"({result.duration / result.generation_time:.1f}x realtime) "
        f"@ {result.sample_rate} Hz"
    )
    yield (result.sample_rate, result.audio), status


def _save_profile(
    name: str,
    model_id: str,
    voice: str | None,
    ref_audio: str | None,
    ref_text: str,
    notes: str,
) -> tuple:
    """Save a voice profile and return (status, updated_table)."""
    if not name or not name.strip():
        return "Please enter a profile name.", _list_profiles()

    if not model_id:
        return "Please select a model.", _list_profiles()

    profile = VoiceProfile(
        model_id=model_id,
        voice=voice if voice else None,
        reference_audio=ref_audio if ref_audio else None,
        reference_text=ref_text if ref_text and ref_text.strip() else None,
        notes=notes if notes and notes.strip() else None,
    )

    try:
        save_profile(name.strip(), profile)
    except Exception as e:
        return f"Failed to save profile: {e}", _list_profiles()

    return f"Profile **{name.strip()}** saved.", _list_profiles()


def _delete_profile(name: str) -> tuple:
    """Delete a voice profile and return (status, updated_table)."""
    if not name or not name.strip():
        return "Please enter a profile name to delete.", _list_profiles()

    try:
        delete_profile(name.strip())
    except KeyError:
        return f"Profile **{name.strip()}** not found.", _list_profiles()
    except Exception as e:
        return f"Failed to delete profile: {e}", _list_profiles()

    return f"Profile **{name.strip()}** deleted.", _list_profiles()


def _list_profiles() -> list[list]:
    """Return profile data as rows for a Dataframe."""
    profiles = list_profiles()
    rows = []
    for name, p in profiles.items():
        voice_display = p.voice or "(reference audio)"
        rows.append([name, p.model_id, voice_display])
    return rows


def _load_profile(name: str):
    """Load a profile's settings into the form fields."""
    if not name or not name.strip():
        return (
            gr.update(),  # profile_name
            gr.update(),  # model_dropdown
            None,          # pending_voice (gr.State — must be a plain value)
            gr.update(),  # ref_audio
            gr.update(),  # ref_text
            gr.update(),  # notes
            gr.update(),  # ref_audio_group visibility
            "Please select a profile to edit.",
        )

    try:
        profile = get_profile(name.strip())
    except KeyError:
        return (
            gr.update(),
            gr.update(),
            None,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            f"Profile **{name.strip()}** not found.",
        )

    # Determine voice choices and ref audio visibility for this model
    voice_choices = []
    supports_cloning = False
    for info, _avail in registry.list_models():
        if info.model_id == profile.model_id:
            voice_choices = info.available_voices or []
            supports_cloning = info.supports_voice_cloning
            break

    return (
        name.strip(),                                          # profile_name
        profile.model_id,                                      # model_dropdown
        profile.voice,                                         # pending_voice (for model change handler)
        profile.reference_audio,                               # ref_audio
        profile.reference_text or "",                          # ref_text
        profile.notes or "",                                   # notes
        gr.update(visible=supports_cloning),                   # ref_audio_group
        f"Loaded profile **{name.strip()}** — edit and save.", # status
    )


def build_voice_profiles_tab():
    """Build the voice profiles tab. Returns (model_dropdown, voice_dropdown, ref_audio, ref_text)
    so the playground tab can pre-load values."""
    choices = get_model_choices()

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Voice settings")
            model_dropdown = gr.Dropdown(
                label="Model",
                choices=choices,
                value=choices[0][1] if choices else None,
                interactive=True,
                elem_id="vp_model",
                info="Select the TTS model for this profile.",
            )
            voice_dropdown = gr.Dropdown(
                label="Voice preset",
                choices=["(no presets — use reference audio)"],
                visible=True,
                interactive=False,
                elem_id="vp_voice",
            )
            ref_audio_group = gr.Group(visible=False)
            with ref_audio_group:
                ref_audio = gr.Audio(
                    label="Reference audio — upload 3-10s for voice cloning",
                    type="filepath",
                )
                ref_text = gr.Textbox(
                    label="Reference audio transcript",
                    placeholder="Transcript of the reference audio...",
                )

            gr.Markdown("### Test voice")
            test_text = gr.Textbox(
                label="Test text",
                placeholder="Enter text to preview the voice...",
                lines=3,
                value="The north wind and the sun were disputing which was the stronger, when a traveller came along wrapped in a warm cloak.",
            )
            with gr.Row():
                test_btn = gr.Button("Test Voice", variant="secondary")
                cancel_btn = gr.Button("Cancel", variant="stop", visible=False)
                unload_btn = gr.Button("Unload Model", variant="secondary", size="sm")
            test_audio = gr.Audio(label="Preview", type="numpy")
            test_status = gr.Markdown()

        with gr.Column(scale=1):
            gr.Markdown("### Save profile")
            profile_name = gr.Textbox(
                label="Profile name",
                placeholder="e.g. emma, narrator, custom-voice",
                info="A friendly name for this voice profile.",
            )
            notes = gr.Textbox(
                label="Notes",
                placeholder="Optional notes about this profile...",
                lines=2,
            )
            save_btn = gr.Button("Save Profile", variant="primary")
            save_status = gr.Markdown()

            gr.Markdown("### Existing profiles")
            gr.Markdown("*Click a row to select it.*", elem_classes=["text-sm"])
            profile_table = gr.Dataframe(
                headers=["Name", "Model", "Voice"],
                value=_list_profiles(),
                interactive=False,
            )
            selected_profile_name = gr.State(None)
            pending_delete_name = gr.State(None)
            with gr.Row():
                edit_btn = gr.Button("Edit", variant="secondary")
                delete_btn = gr.Button("Delete", variant="stop")
            edit_status = gr.Markdown()

    # Wire events — model selection
    # Note: voice dropdown population is handled by main.py's _apply_pending_voice
    # to support the "Send to Profile" flow from the playground tab.
    model_dropdown.change(
        fn=_show_ref_audio,
        inputs=[model_dropdown],
        outputs=[ref_audio_group],
    )

    # Test voice
    test_event = test_btn.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[test_btn, cancel_btn],
    ).then(
        fn=_test_voice,
        inputs=[model_dropdown, voice_dropdown, ref_audio, ref_text, test_text],
        outputs=[test_audio, test_status],
    ).then(
        fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[test_btn, cancel_btn],
    )

    cancel_btn.click(
        fn=lambda: (None, "Cancelled."),
        outputs=[test_audio, test_status],
        cancels=[test_event],
    )

    def _unload():
        registry.unload_current()
        return "Model unloaded, GPU memory freed."

    unload_btn.click(fn=_unload, outputs=[test_status])

    # Table row selection → store selected profile name
    def _on_table_select(evt: gr.SelectData):
        row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        name = evt.value if isinstance(evt.value, str) else None
        if not name:
            # Fallback: get name from the row data if available
            try:
                name = evt.row_value[0] if hasattr(evt, "row_value") else None
            except (IndexError, TypeError):
                name = None
        if name:
            return name, f"Selected **{name}**.", None
        return None, "", None

    profile_table.select(
        fn=_on_table_select,
        outputs=[selected_profile_name, edit_status, pending_delete_name],
    )

    # Save profile
    save_btn.click(
        fn=_save_profile,
        inputs=[profile_name, model_dropdown, voice_dropdown, ref_audio, ref_text, notes],
        outputs=[save_status, profile_table],
    )

    # Edit profile — load into form fields
    # Note: voice is routed via pending_voice and applied by _on_vp_model_change
    # in main.py when the model dropdown change fires. Do not set voice_dropdown
    # directly here as it would be overridden.
    # Wiring completed in main.py where pending_voice is available.

    # Delete profile (two-click confirmation)
    def _delete_with_confirm(name, pending):
        if not name or not name.strip():
            return "Please select a profile to delete.", _list_profiles(), None, None
        if pending == name:
            # Second click — actually delete
            status, table = _delete_profile(name)
            return status, table, None, None
        else:
            # First click — ask for confirmation
            return (
                f"Click **Delete** again to confirm deletion of **{name.strip()}**.",
                _list_profiles(),
                name,
                name,
            )

    delete_btn.click(
        fn=_delete_with_confirm,
        inputs=[selected_profile_name, pending_delete_name],
        outputs=[edit_status, profile_table, selected_profile_name, pending_delete_name],
    )

    return (model_dropdown, voice_dropdown, ref_audio, ref_text, ref_audio_group,
            edit_btn, selected_profile_name, profile_name, notes, edit_status)
