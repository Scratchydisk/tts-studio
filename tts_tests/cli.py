"""Unified CLI for TTS Studio.

Commands:
    tts-studio serve          Launch the Gradio web UI (default)
    tts-studio caption        Process a video with caption-to-speech
    tts-studio batch          Batch process a directory of videos
    tts-studio generate       One-shot TTS from command line
    tts-studio profiles       List saved voice profiles
"""

import argparse
import logging
import sys

import gradio as gr

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI usage."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(name)s: %(message)s",
    )


def _suppress_warnings() -> None:
    """Suppress noisy warnings from dependencies."""
    import warnings
    warnings.filterwarnings("ignore", message=".*weight_norm.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*dropout option adds dropout.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*convert audio automatically.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*not in the list of choices.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*Defaulting repo_id.*")
    warnings.filterwarnings("ignore", message=".*Defaulting repo_id.*", category=UserWarning)
    # Kokoro prints this via warnings module or logging
    import logging as _logging
    _logging.getLogger("kokoro").setLevel(_logging.ERROR)


def _cmd_serve(args: argparse.Namespace) -> int:
    """Launch the Gradio web UI."""
    from tts_tests import registry
    from tts_tests.ui.main import build_app

    _suppress_warnings()
    registry.discover()

    models = registry.list_models()
    logger.info("Discovered %d models:", len(models))
    for info, available in models:
        if not available:
            status = "deps missing"
        elif registry.is_remote(info.model_id):
            endpoint = registry.get_endpoint(info.model_id)
            status = f"remote ({endpoint['url']})"
        else:
            status = "local"
        logger.info("  %-20s %s", info.name, status)

    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Source Sans 3"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Source Code Pro"), "monospace"],
    )

    css = """
    /* Row-level highlight on dataframe hover */
    .gradio-dataframe tbody tr:hover { background: var(--color-accent-soft) !important; cursor: pointer; }
    .gradio-dataframe tbody tr td { transition: background 0.15s ease; }

    /* Tighter model-info card */
    .model-info-card { padding: 12px 16px; border-radius: 8px;
        background: var(--block-background-fill); border: 1px solid var(--border-color-primary); }
    .model-info-card p { margin: 2px 0 !important; }

    /* Button group spacing — separate primary actions from secondary */
    .action-buttons { gap: 8px !important; }
    .action-buttons .secondary { margin-left: auto !important; }

    /* Status area */
    .status-msg { min-height: 2em; }

    /* Legend as a quiet subtitle */
    .icon-legend { opacity: 0.65; font-size: 0.85em; }
    """

    app = build_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=theme,
        css=css,
    )
    return 0


def _cmd_caption(args: argparse.Namespace) -> int:
    """Process a single video with caption-to-speech."""
    from tts_tests.api import caption_video

    try:
        result = caption_video(
            video_path=args.input,
            srt_path=args.srt,
            profile=args.profile,
            model_id=args.model,
            voice=args.voice,
            reference_audio=args.ref_audio,
            reference_text=args.ref_text,
            output_path=args.output,
            output_format=args.format,
            silence_before=args.silence_before,
            silence_after=args.silence_after,
            from_stage=args.from_stage,
            keep_intermediates=args.keep_intermediates,
        )
        print(f"Output:   {result.output_path}")
        print(f"Duration: {result.duration:.1f}s")
        print(f"Segments: {result.segments_count}")
        return 0
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1
    except Exception as e:
        logger.error("Caption failed: %s", e)
        return 1


def _cmd_batch(args: argparse.Namespace) -> int:
    """Batch process a directory of videos."""
    from tts_tests.api import caption_batch

    try:
        results = caption_batch(
            input_dir=args.input_dir,
            video_pattern=args.pattern,
            profile=args.profile,
            model_id=args.model,
            voice=args.voice,
            output_format=args.format,
            keep_intermediates=args.keep_intermediates,
        )
        if not results:
            print("No video+SRT pairs found or all failed.")
            return 1
        print(f"Processed {len(results)} video(s):")
        for r in results:
            print(f"  {r.output_path.name}  ({r.duration:.1f}s)")
        return 0
    except Exception as e:
        logger.error("Batch processing failed: %s", e)
        return 1


def _cmd_generate(args: argparse.Namespace) -> int:
    """One-shot TTS generation."""
    from tts_tests.api import generate_tts

    try:
        audio, sr = generate_tts(
            text=args.text,
            profile=args.profile,
            model_id=args.model,
            voice=args.voice,
            output_path=args.output,
        )
        print(f"Generated {len(audio) / sr:.2f}s of audio -> {args.output}")
        return 0
    except Exception as e:
        logger.error("TTS generation failed: %s", e)
        return 1


def _cmd_profiles(_args: argparse.Namespace) -> int:
    """List saved voice profiles."""
    from tts_tests.profiles import list_profiles

    profiles = list_profiles()
    if not profiles:
        print("No voice profiles saved.")
        print("Create profiles in the web UI or edit profiles.json directly.")
        return 0

    print(f"{'Name':<20} {'Model':<20} {'Voice':<15} {'Ref Audio'}")
    print("-" * 75)
    for name, vp in sorted(profiles.items()):
        ref = vp.reference_audio or ""
        voice = vp.voice or ""
        print(f"{name:<20} {vp.model_id:<20} {voice:<15} {ref}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="tts-studio",
        description="TTS Studio — compare and use open-source TTS models",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging",
    )
    subparsers = parser.add_subparsers(dest="command")

    # serve (default when no command given)
    serve_parser = subparsers.add_parser("serve", help="Launch web UI")
    serve_parser.add_argument("--port", type=int, default=7860)
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--share", action="store_true")

    # caption
    caption_parser = subparsers.add_parser("caption", help="Caption a video")
    caption_parser.add_argument("input", help="Video file path")
    caption_parser.add_argument("--srt", help="SRT file (auto-detected if not given)")
    caption_parser.add_argument("--profile", help="Voice profile name")
    caption_parser.add_argument("--model", default=None, help="Model ID")
    caption_parser.add_argument("--voice", help="Voice preset name")
    caption_parser.add_argument(
        "--ref-audio", help="Reference audio for voice cloning",
    )
    caption_parser.add_argument("--ref-text", help="Reference audio transcript")
    caption_parser.add_argument("-o", "--output", help="Output file path")
    caption_parser.add_argument(
        "--format", default="mkv", choices=["mkv", "mp4", "webm"],
    )
    caption_parser.add_argument(
        "--from-stage", choices=["parse", "tts", "timeline", "render"],
    )
    caption_parser.add_argument(
        "--keep-intermediates", action="store_true",
    )
    caption_parser.add_argument(
        "--silence-before", type=float, default=0.3,
    )
    caption_parser.add_argument(
        "--silence-after", type=float, default=0.5,
    )

    # batch
    batch_parser = subparsers.add_parser("batch", help="Batch process directory")
    batch_parser.add_argument("input_dir", help="Directory with video+SRT files")
    batch_parser.add_argument("--pattern", default="*.mp4")
    batch_parser.add_argument("--profile", help="Voice profile name")
    batch_parser.add_argument("--model", default=None)
    batch_parser.add_argument("--voice")
    batch_parser.add_argument("--format", default="mkv")
    batch_parser.add_argument("--keep-intermediates", action="store_true")

    # generate
    gen_parser = subparsers.add_parser("generate", help="One-shot TTS")
    gen_parser.add_argument("text", help="Text to synthesise")
    gen_parser.add_argument("--profile", help="Voice profile name")
    gen_parser.add_argument("--model", default=None)
    gen_parser.add_argument("--voice")
    gen_parser.add_argument("-o", "--output", default="output.wav")

    # profiles
    subparsers.add_parser("profiles", help="List voice profiles")

    args = parser.parse_args()

    _setup_logging(verbose=args.verbose)

    # Default to serve if no command given
    if args.command is None:
        args.command = "serve"
        args.host = "0.0.0.0"
        args.port = 7860
        args.share = False

    # Dispatch to handlers
    handlers = {
        "serve": _cmd_serve,
        "caption": _cmd_caption,
        "batch": _cmd_batch,
        "generate": _cmd_generate,
        "profiles": _cmd_profiles,
    }

    handler = handlers[args.command]
    sys.exit(handler(args))


if __name__ == "__main__":
    main()
