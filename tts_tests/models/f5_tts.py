"""F5-TTS wrapper."""

import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from tts_tests.base import ModelInfo, TTSModel, TTSResult


def is_available() -> bool:
    try:
        import f5_tts  # noqa: F401
        return True
    except ImportError:
        return False


def _ensure_wav(audio_path: Path) -> str:
    """Convert any audio file to 24kHz mono WAV using soundfile/pydub.

    This avoids torchcodec entirely.
    """
    suffix = audio_path.suffix.lower()
    if suffix == ".wav":
        return str(audio_path)

    # Use pydub to convert non-wav formats
    from pydub import AudioSegment

    seg = AudioSegment.from_file(str(audio_path))
    seg = seg.set_channels(1).set_frame_rate(24000)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    seg.export(tmp.name, format="wav")
    return tmp.name


class F5TTS(TTSModel):
    def __init__(self):
        self._model = None
        self._device = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="F5-TTS",
            model_id="f5-tts",
            supports_voice_cloning=True,
            available_voices=[],
            estimated_vram_gb=2.5,
            native_sample_rate=24000,
            description="Flow-based TTS with excellent naturalness and voice cloning.",
        )

    def load(self, device: str = "cuda") -> None:
        # torchaudio >= 2.9 requires torchcodec which has CUDA lib issues.
        # Patch torchaudio.load to use soundfile instead.
        import torchaudio
        import soundfile as sf
        import torch

        _original_load = getattr(torchaudio, "_original_load", None)
        if _original_load is None:
            def _sf_load(uri, frame_offset=0, num_frames=-1, normalize=True,
                         channels_first=True, format=None, buffer_size=4096, backend=None):
                data, sr = sf.read(str(uri), start=frame_offset,
                                   stop=frame_offset + num_frames if num_frames > 0 else None,
                                   dtype="float32")
                audio = torch.from_numpy(data)
                if audio.ndim == 1:
                    audio = audio.unsqueeze(0)
                elif channels_first:
                    audio = audio.T
                return audio, sr

            torchaudio._original_load = torchaudio.load
            torchaudio.load = _sf_load

        from f5_tts.api import F5TTS as F5TTSApi
        self._model = F5TTSApi(device=device)
        self._device = device

    def unload(self) -> None:
        del self._model
        self._model = None
        self._device = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def generate(
        self,
        text: str,
        voice: str | None = None,
        reference_audio: Path | None = None,
        reference_text: str | None = None,
    ) -> TTSResult:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        start = time.perf_counter()

        if not reference_audio:
            # Use the bundled English reference audio
            import f5_tts.api
            pkg_dir = Path(f5_tts.api.__file__).parent
            reference_audio = pkg_dir / "infer" / "examples" / "basic" / "basic_ref_en.wav"
            reference_text = "Some call me nature, others call me mother nature."
        else:
            reference_audio = Path(reference_audio)

        ref_file = _ensure_wav(reference_audio)

        wav, sr, _ = self._model.infer(
            ref_file=ref_file,
            ref_text=reference_text or "",
            gen_text=text,
        )

        elapsed = time.perf_counter() - start
        audio = np.array(wav).flatten().astype(np.float32)

        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration=len(audio) / sr,
            generation_time=elapsed,
        )


MODEL_CLASS = F5TTS
