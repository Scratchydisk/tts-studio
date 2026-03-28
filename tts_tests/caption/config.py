"""Caption TTS configuration — adapted from cap_to_speech for the tts_tests registry."""

from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CaptionTTSConfig:
    model_id: str = "kokoro-82m"
    voice: str | None = None
    reference_audio: str | None = None
    reference_text: str | None = None
    speed: float = 1.0
    silence_before: float = 0.3
    silence_after: float = 0.5


@dataclass(frozen=True)
class OutputConfig:
    format: str = "mkv"


@dataclass(frozen=True)
class Config:
    tts: CaptionTTSConfig = field(default_factory=CaptionTTSConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def load(cls, path: Path) -> "Config":
        if not path.exists():
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        output_data = data.get("output", {})
        # Ignore legacy keys from old configs.
        output_data.pop("formats", None)
        return cls(
            tts=CaptionTTSConfig(
                **{**CaptionTTSConfig().__dict__, **data.get("tts", {})}
            ),
            output=OutputConfig(
                **{**OutputConfig().__dict__, **output_data}
            ),
        )


def load_config(path: Path | None = None) -> Config:
    """Load configuration from a YAML file, falling back to defaults."""
    if path is None:
        return Config()
    return Config.load(path)


def merge_cli_overrides(config: Config, **kwargs) -> Config:
    """Merge CLI overrides into an existing config.

    Recognised keyword arguments (non-None values only):
        voice, model_id, speed, reference_audio, reference_text
    """
    tts_fields = ("voice", "model_id", "speed", "reference_audio", "reference_text")
    tts_overrides = {k: v for k, v in kwargs.items() if k in tts_fields and v is not None}
    if not tts_overrides:
        return config
    new_tts = replace(config.tts, **tts_overrides)
    return replace(config, tts=new_tts)
