"""Main Gradio application layout."""

import gradio as gr

from tts_tests.ui.caption_batch import build_caption_batch_tab
from tts_tests.ui.caption_single import build_caption_single_tab
from tts_tests.ui.compare import build_compare_tab
from tts_tests.ui.model_management import build_model_management_tab
from tts_tests.ui.shared import get_icon_legend, get_model_choices
from tts_tests.ui.single import build_playground_tab
from tts_tests.ui.voice_profiles import build_voice_profiles_tab


def build_app(theme: gr.themes.Base | None = None) -> gr.Blocks:
    with gr.Blocks(
        title="TTS Studio",
    ) as app:
        gr.Markdown("# TTS Studio")
        gr.Markdown(get_icon_legend(), elem_classes=["icon-legend"])

        # State: voice to apply after model change (set by "Send to Profile" or Edit)
        pending_voice = gr.State(None)

        with gr.Tabs() as tabs:
            with gr.Tab("Playground", id="playground"):
                playground_refs = build_playground_tab()
            with gr.Tab("Compare", id="compare"):
                compare_refs = build_compare_tab()
            with gr.Tab("Voice Profiles", id="profiles"):
                profile_refs = build_voice_profiles_tab()
            with gr.Tab("Caption Video", id="caption"):
                caption_profile_dd = build_caption_single_tab()
            with gr.Tab("Batch Caption", id="batch"):
                batch_profile_dd = build_caption_batch_tab()
            with gr.Tab("Models", id="models"):
                build_model_management_tab()

        # --- Refresh dropdowns on tab switch ---
        from tts_tests.ui.caption_single import _get_profile_choices

        if playground_refs and profile_refs:
            save_btn, pg_model, pg_voice, pg_ref_audio, pg_ref_text = playground_refs
            (vp_model, vp_voice, vp_ref_audio, vp_ref_text, vp_ref_group,
             vp_edit_btn, vp_selected_profile, vp_profile_name, vp_notes, vp_edit_status) = profile_refs

            cmp_model_a, cmp_model_b = compare_refs

            def _refresh_on_tab_switch():
                profile_choices = _get_profile_choices()
                model_choices = get_model_choices()
                available_model_choices = get_model_choices(include_unavailable=False)
                p = gr.update(choices=profile_choices)
                m = gr.update(choices=model_choices)
                c = gr.update(choices=available_model_choices)
                return p, p, m, m, c, c

            tabs.change(
                fn=_refresh_on_tab_switch,
                outputs=[caption_profile_dd, batch_profile_dd,
                         pg_model, vp_model,
                         cmp_model_a, cmp_model_b],
            )

            # --- Playground → Voice Profiles wiring ---
            from tts_tests import registry
            from tts_tests.ui.voice_profiles import _load_profile, _show_ref_audio

            def _send_to_profile(model_id, voice, ref_audio, ref_text):
                ref_visible = _show_ref_audio(model_id)
                return (
                    voice,          # pending_voice
                    model_id,       # vp_model
                    ref_audio,      # vp_ref_audio
                    ref_text,       # vp_ref_text
                    ref_visible,    # vp_ref_group
                    gr.update(selected="profiles"),  # tabs
                )

            save_btn.click(
                fn=_send_to_profile,
                inputs=[pg_model, pg_voice, pg_ref_audio, pg_ref_text],
                outputs=[pending_voice, vp_model, vp_ref_audio, vp_ref_text,
                         vp_ref_group, tabs],
            )

            # Single handler for all vp_model changes — checks pending_voice
            def _on_vp_model_change(model_id, pending):
                for info, _avail in registry.list_models():
                    if info.model_id == model_id:
                        if info.available_voices:
                            voice = pending if pending and pending in info.available_voices else info.available_voices[0]
                            return gr.update(choices=info.available_voices, value=voice, visible=True, interactive=True), None
                        else:
                            return gr.update(choices=["(no presets — use reference audio)"], value=None, visible=True, interactive=False), None
                return gr.update(choices=["(no presets — use reference audio)"], value=None, visible=True, interactive=False), None

            vp_model.change(
                fn=_on_vp_model_change,
                inputs=[vp_model, pending_voice],
                outputs=[vp_voice, pending_voice],
            )

            # Edit profile — load into form, routing voice via pending_voice.
            # Chain _on_vp_model_change after the click so the voice dropdown
            # updates (Gradio doesn't auto-fire .change from programmatic updates).
            vp_edit_btn.click(
                fn=_load_profile,
                inputs=[vp_selected_profile],
                outputs=[vp_profile_name, vp_model, pending_voice,
                         vp_ref_audio, vp_ref_text, vp_notes,
                         vp_ref_group, vp_edit_status],
            ).then(
                fn=_on_vp_model_change,
                inputs=[vp_model, pending_voice],
                outputs=[vp_voice, pending_voice],
            )

    return app
