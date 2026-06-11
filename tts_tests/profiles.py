"""Voice profile management.

A voice profile saves a TTS configuration (model, voice, reference audio)
under a friendly name for reuse across the UI and CLI.

Example profiles.json:
{
    "emma": {
        "model_id": "kokoro-82m",
        "voice": "bf_emma"
    },
    "cloned-sarah": {
        "model_id": "f5-tts",
        "reference_audio": "reference_audio/sarah.wav",
        "reference_text": "This is Sarah speaking naturally."
    }
}
"""

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
PROFILES_FILE = _PROJECT_ROOT / "profiles.json"


def _to_relative(audio_path: str | None) -> str | None:
    """Convert an absolute path within the project to a relative path."""
    if not audio_path:
        return None
    try:
        return str(Path(audio_path).resolve().relative_to(_PROJECT_ROOT.resolve()))
    except ValueError:
        return audio_path


def _to_absolute(audio_path: str | None) -> str | None:
    """Resolve a relative path against the project root."""
    if not audio_path:
        return None
    p = Path(audio_path)
    if p.is_absolute():
        return audio_path
    return str(_PROJECT_ROOT / p)


def _persist_reference_audio(audio_path: str | None, profile_name: str) -> str | None:
    """Copy reference audio to persistent storage if it's in a temp directory.

    Returns the new path (or the original if already persistent).
    """
    if not audio_path:
        return None

    src = Path(audio_path).resolve()
    if not src.exists():
        return audio_path

    from tts_tests.config import PROFILE_AUDIO_DIR

    # If already in the reference audio directory, keep as-is
    try:
        src.relative_to(PROFILE_AUDIO_DIR.resolve())
        return audio_path
    except ValueError:
        pass

    # Also check relative path form
    try:
        src.relative_to(_PROJECT_ROOT.resolve())
        # Already within the project — keep it
        return audio_path
    except ValueError:
        pass

    # File is outside the project (e.g. /tmp from Gradio upload) — copy it in
    dest_name = f"{profile_name}_{src.name}"
    dest = PROFILE_AUDIO_DIR / dest_name
    shutil.copy2(src, dest)
    logger.info("Copied reference audio to %s", dest)
    return _to_relative(str(dest))


@dataclass
class VoiceProfile:
    """A named TTS configuration."""

    model_id: str
    voice: str | None = None
    reference_audio: str | None = None
    reference_text: str | None = None
    notes: str | None = None


def _validate_name(name: str) -> None:
    """Ensure the profile name is a non-empty string."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Profile name must be a non-empty string.")


def _read_profiles_file() -> dict:
    """Read the raw JSON data from the profiles file."""
    if not PROFILES_FILE.exists():
        return {}
    try:
        data = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            logger.warning("profiles.json should be a JSON object, ignoring")
            return {}
        return data
    except Exception as e:
        logger.warning("Failed to read profiles.json: %s", e)
        return {}


def _write_profiles_file(data: dict) -> None:
    """Write the raw JSON data to the profiles file atomically."""
    content = json.dumps(data, indent=4, ensure_ascii=False) + "\n"
    tmp = PROFILES_FILE.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    # Path.replace (os.replace) overwrites atomically on Windows; rename does not
    tmp.replace(PROFILES_FILE)


def list_profiles() -> dict[str, VoiceProfile]:
    """Load all profiles from profiles.json."""
    raw = _read_profiles_file()
    profiles: dict[str, VoiceProfile] = {}
    for name, entry in raw.items():
        try:
            profiles[name] = VoiceProfile(
                model_id=entry["model_id"],
                voice=entry.get("voice"),
                reference_audio=_to_absolute(entry.get("reference_audio")),
                reference_text=entry.get("reference_text"),
                notes=entry.get("notes"),
            )
        except (KeyError, TypeError) as e:
            logger.warning("Skipping malformed profile %r: %s", name, e)
    return profiles


def get_profile(name: str) -> VoiceProfile:
    """Get a profile by name. Raises KeyError if not found."""
    _validate_name(name)
    profiles = list_profiles()
    if name not in profiles:
        raise KeyError(f"Profile not found: {name!r}")
    return profiles[name]


def save_profile(name: str, profile: VoiceProfile) -> None:
    """Save or update a profile.

    If the reference audio is in a temp directory, it is copied to
    ~/.cache/tts-studio/reference_audio/ for persistence across reboots.
    """
    _validate_name(name)

    # Persist reference audio if needed
    persisted_audio = _persist_reference_audio(profile.reference_audio, name.strip())
    if persisted_audio != profile.reference_audio:
        profile = VoiceProfile(
            model_id=profile.model_id,
            voice=profile.voice,
            reference_audio=persisted_audio,
            reference_text=profile.reference_text,
            notes=profile.notes,
        )

    # Convert to relative path for portability
    profile = VoiceProfile(
        model_id=profile.model_id,
        voice=profile.voice,
        reference_audio=_to_relative(profile.reference_audio),
        reference_text=profile.reference_text,
        notes=profile.notes,
    )

    raw = _read_profiles_file()
    # Strip None values to keep the JSON tidy
    entry = {k: v for k, v in asdict(profile).items() if v is not None}
    raw[name] = entry
    _write_profiles_file(raw)
    logger.info("Saved profile %r", name)


def delete_profile(name: str) -> None:
    """Delete a profile. Raises KeyError if not found."""
    _validate_name(name)
    raw = _read_profiles_file()
    if name not in raw:
        raise KeyError(f"Profile not found: {name!r}")
    del raw[name]
    _write_profiles_file(raw)
    logger.info("Deleted profile %r", name)
