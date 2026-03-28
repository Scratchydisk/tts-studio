"""Voxtral-4B TTS local wrapper.

Runs Voxtral directly via vllm-omni's offline API.
Supports voice cloning via reference audio.
Requires vllm >= 0.18.0, vllm-omni, and ~16GB VRAM.
"""

import logging
import time
from pathlib import Path

import numpy as np

from tts_tests.base import ModelInfo, TTSModel, TTSResult

logger = logging.getLogger(__name__)

MODEL_NAME = "mistralai/Voxtral-4B-TTS-2603"

VOICES = [
    "casual_female", "casual_male", "cheerful_female",
    "neutral_female", "neutral_male",
    "fr_female", "fr_male",
    "es_female", "es_male",
    "de_female", "de_male",
    "it_female", "it_male",
    "pt_female", "pt_male",
    "nl_female", "nl_male",
    "ar_male",
    "hi_female", "hi_male",
]


def is_available() -> bool:
    try:
        import vllm  # noqa: F401
        import vllm_omni  # noqa: F401
        import mistral_common  # noqa: F401
        return True
    except ImportError:
        return False


class VoxtralLocalTTS(TTSModel):
    def __init__(self):
        self._llm = None
        self._tokenizer = None
        self._device = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Voxtral-4B (local)",
            model_id="voxtral-4b-local",
            supports_voice_cloning=True,
            available_voices=VOICES,
            estimated_vram_gb=16.0,
            native_sample_rate=24000,
            description=(
                "Mistral's 4B param TTS model running locally via vllm-omni. "
                "9 languages, 20 preset voices, voice cloning from ~3s of "
                "reference audio. Requires ~16GB VRAM."
            ),
        )

    def load(self, device: str = "cuda") -> None:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from vllm import SamplingParams  # noqa: F401
        from vllm_omni.entrypoints.omni import Omni

        logger.info("Loading Voxtral tokenizer...")
        self._tokenizer = MistralTokenizer.from_hf_hub(MODEL_NAME)

        logger.info("Loading Voxtral model (this may take a while)...")
        self._llm = Omni(model=MODEL_NAME)

        self._device = device
        logger.info("Voxtral loaded.")

    def unload(self) -> None:
        del self._llm
        del self._tokenizer
        self._llm = None
        self._tokenizer = None
        self._device = None

    def is_loaded(self) -> bool:
        return self._llm is not None

    def generate(
        self,
        text: str,
        voice: str | None = None,
        reference_audio: Path | None = None,
        reference_text: str | None = None,
    ) -> TTSResult:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        from mistral_common.protocol.speech.request import SpeechRequest
        from vllm import SamplingParams

        instruct_tokenizer = self._tokenizer.instruct_tokenizer
        voice = voice or "neutral_female"
        start = time.perf_counter()

        if reference_audio and reference_audio.exists():
            ref_audio_bytes = reference_audio.read_bytes()
            tokenized = instruct_tokenizer.encode_speech_request(
                SpeechRequest(input=text, ref_audio=ref_audio_bytes)
            )
            inputs = {
                "prompt_token_ids": tokenized.tokens,
                "multi_modal_data": {
                    "audio": [(
                        tokenized.audios[0].audio_array,
                        tokenized.audios[0].sampling_rate,
                    )]
                },
            }
        else:
            tokenized = instruct_tokenizer.encode_speech_request(
                SpeechRequest(input=text, voice=voice)
            )
            inputs = {
                "prompt_token_ids": tokenized.tokens,
                "additional_information": {"voice": [voice]},
            }

        sampling_params_list = [
            SamplingParams(max_tokens=4096),
            SamplingParams(max_tokens=4096),
        ]

        logger.info("Generating audio...")
        outputs = self._llm.generate(inputs, sampling_params_list=sampling_params_list)

        # Extract audio from output
        if hasattr(outputs, "audio"):
            import io
            import soundfile as sf
            audio, sr = sf.read(io.BytesIO(outputs.audio), dtype="float32")
        elif hasattr(outputs, "outputs"):
            # Some versions return a list of output objects
            audio_data = outputs.outputs[0]
            if hasattr(audio_data, "audio"):
                import io
                import soundfile as sf
                audio, sr = sf.read(io.BytesIO(audio_data.audio), dtype="float32")
            else:
                raise RuntimeError(f"Unexpected output format: {type(audio_data)}")
        else:
            raise RuntimeError(f"Unexpected output format: {type(outputs)}")

        elapsed = time.perf_counter() - start

        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration=len(audio) / sr,
            generation_time=elapsed,
        )


MODEL_CLASS = VoxtralLocalTTS
