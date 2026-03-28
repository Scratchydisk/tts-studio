"""Abstract base class and data types for TTS model wrappers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class TTSResult:
    """Result from a TTS generation."""

    audio: np.ndarray  # float32 numpy array
    sample_rate: int
    duration: float  # seconds
    generation_time: float  # seconds


@dataclass
class ModelInfo:
    """Metadata about a TTS model."""

    name: str
    model_id: str
    supports_voice_cloning: bool = False
    supports_emotions: bool = False
    available_voices: list[str] = field(default_factory=list)
    estimated_vram_gb: float = 0.0
    native_sample_rate: int = 24000
    description: str = ""


class TTSModel(ABC):
    """Base class that all TTS model wrappers must implement."""

    @abstractmethod
    def info(self) -> ModelInfo:
        """Return model metadata."""
        ...

    @abstractmethod
    def load(self, device: str = "cuda") -> None:
        """Load the model onto the specified device."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Unload the model and free GPU memory."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check whether the model is currently loaded."""
        ...

    @abstractmethod
    def generate(
        self,
        text: str,
        voice: str | None = None,
        reference_audio: Path | None = None,
        reference_text: str | None = None,
    ) -> TTSResult:
        """Generate speech from text.

        Args:
            text: The text to synthesise.
            voice: Preset voice name (if the model supports it).
            reference_audio: Path to reference audio for voice cloning.
            reference_text: Transcript of the reference audio.

        Returns:
            TTSResult with the generated audio.
        """
        ...
