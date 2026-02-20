# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Inference-only VibeVoice ASR model compatible with HuggingFace weights.

VibeVoice ASR is a speech-to-text model that combines acoustic/semantic
VAE tokenizers for audio encoding with a Qwen2-based language model for
text generation. It supports:
- Up to 60 minutes of audio input via streaming segmentation
- Customized hotwords
- Rich transcription (Who, When, What)

Reference: https://github.com/microsoft/VibeVoice
Model: https://huggingface.co/microsoft/VibeVoice-ASR
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import BatchFeature
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

# ============================================================================
# Audio Loading Utilities
# ============================================================================
# VibeVoice requires FFmpeg-based audio decoding at 24kHz with specific
# normalization. These are lazily imported from the vibevoice package.


def _load_audio_ffmpeg(filepath: str) -> tuple[np.ndarray, int]:
    """Load audio file using FFmpeg via the vibevoice package.

    Returns:
        Tuple of (audio_waveform, sample_rate). Sample rate is always 24000.
    """
    from vibevoice.processor.audio_utils import (
        AudioNormalizer,
        load_audio_use_ffmpeg,
    )

    audio, sr = load_audio_use_ffmpeg(
        filepath, resample=True, target_sr=24000
    )
    normalizer = AudioNormalizer()
    audio = normalizer(audio)
    return audio, sr


def _load_audio_bytes_ffmpeg(data: bytes) -> tuple[np.ndarray, int]:
    """Load audio bytes using FFmpeg via stdin-pipe decoding.

    Returns:
        Tuple of (audio_waveform, sample_rate). Sample rate is always 24000.
    """
    from vibevoice.processor.audio_utils import (
        AudioNormalizer,
        load_audio_bytes_use_ffmpeg,
    )

    audio, sr = load_audio_bytes_use_ffmpeg(
        data, resample=True, target_sr=24000
    )
    normalizer = AudioNormalizer()
    audio = normalizer(audio)
    return audio, sr


# ============================================================================
# Audio Encoder Components
# ============================================================================


class SpeechConnector(nn.Module):
    """Projects speech features to language model hidden dimension.

    Architecture: fc1 -> RMSNorm -> fc2 (no activation function).
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = _LlamaRMSNorm(output_dim, eps=1e-6)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class _LlamaRMSNorm(nn.Module):
    """RMSNorm layer used in SpeechConnector."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        return self.weight * hidden_states.to(input_dtype)


def _get_cfg(obj: object, key: str, default: object = None) -> object:
    """Get config value from dict or object attribute."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class VibeVoiceAudioEncoder(nn.Module):
    """VibeVoice Audio Encoder module.

    Encapsulates Acoustic/Semantic VAE Tokenizers and projection Connectors.
    Converts raw audio waveforms into embeddings compatible with the
    language model.

    Features:
        - Streaming support for long audio (>60s by default)
        - Configurable dtype for numerical precision
        - Supports both sampling and deterministic (mean) modes
    """

    def __init__(self, config: object):
        super().__init__()
        self.config = config

        from vibevoice.modular.configuration_vibevoice import (
            VibeVoiceAcousticTokenizerConfig,
            VibeVoiceSemanticTokenizerConfig,
        )
        from vibevoice.modular.modular_vibevoice_tokenizer import (
            VibeVoiceAcousticTokenizerModel,
            VibeVoiceSemanticTokenizerModel,
        )

        self.acoustic_vae_dim = int(_get_cfg(config, "acoustic_vae_dim", 64))
        self.semantic_vae_dim = int(_get_cfg(config, "semantic_vae_dim", 128))

        # Determine the LM hidden size to project audio features into
        decoder_config = _get_cfg(config, "decoder_config")
        text_config = _get_cfg(config, "text_config")
        target_hidden_size = None
        if decoder_config is not None:
            target_hidden_size = _get_cfg(decoder_config, "hidden_size")
        if target_hidden_size is None and text_config is not None:
            target_hidden_size = _get_cfg(text_config, "hidden_size")
        if target_hidden_size is None:
            target_hidden_size = _get_cfg(config, "hidden_size")
        if target_hidden_size is None:
            target_hidden_size = 3584  # Default for 7B model
        self.hidden_size = int(target_hidden_size)

        # Build tokenizer configs
        ac_cfg = _get_cfg(config, "acoustic_tokenizer_config")
        sc_cfg = _get_cfg(config, "semantic_tokenizer_config")
        if ac_cfg is None or sc_cfg is None:
            raise ValueError(
                "Missing acoustic/semantic tokenizer config in model config"
            )

        if isinstance(ac_cfg, VibeVoiceAcousticTokenizerConfig):
            acoustic_config = ac_cfg
        elif isinstance(ac_cfg, dict):
            acoustic_config = VibeVoiceAcousticTokenizerConfig(**ac_cfg)
        else:
            raise TypeError(
                f"acoustic_tokenizer_config has unexpected type: {type(ac_cfg)}"
            )

        if isinstance(sc_cfg, VibeVoiceSemanticTokenizerConfig):
            semantic_config = sc_cfg
        elif isinstance(sc_cfg, dict):
            semantic_config = VibeVoiceSemanticTokenizerConfig(**sc_cfg)
        else:
            raise TypeError(
                "semantic_tokenizer_config has unexpected type: "
                f"{type(sc_cfg)}"
            )

        self.acoustic_tokenizer = VibeVoiceAcousticTokenizerModel(
            acoustic_config
        )
        self.semantic_tokenizer = VibeVoiceSemanticTokenizerModel(
            semantic_config
        )

        # Audio encoder dtype from config (defaults to float32 for precision)
        root_torch_dtype = _get_cfg(config, "torch_dtype", None)
        if root_torch_dtype is not None:
            if isinstance(root_torch_dtype, str):
                self._audio_encoder_dtype = getattr(torch, root_torch_dtype)
            else:
                self._audio_encoder_dtype = root_torch_dtype
        else:
            self._audio_encoder_dtype = torch.float32

        self.acoustic_connector = SpeechConnector(
            self.acoustic_vae_dim, self.hidden_size
        )
        self.semantic_connector = SpeechConnector(
            self.semantic_vae_dim, self.hidden_size
        )

        self.compress_ratio = int(
            _get_cfg(config, "speech_tok_compress_ratio", 3200)
        )

        # Streaming controls
        self.sample_rate = int(_get_cfg(config, "target_sample_rate", 24000))
        self.enable_streaming = bool(
            _get_cfg(config, "enable_streaming", True)
        )
        self.streaming_segment_duration = float(
            _get_cfg(config, "streaming_segment_duration", 60.0)
        )

        # Control sampling vs deterministic mode for acoustic tokens
        import os

        use_mean_env = os.getenv("VIBEVOICE_USE_MEAN", "").strip().lower()
        self.use_sample = use_mean_env not in ("1", "true", "yes")

        # LM dtype (set by VibeVoiceForCausalLM.__init__)
        self._lm_dtype: torch.dtype = torch.bfloat16

    def _ensure_audio_encoder_dtype(self) -> None:
        """Ensure audio encoder components use the correct dtype.

        vLLM may convert weights to a different dtype during loading.
        This converts audio encoder components back to the config-specified
        dtype (typically float32) for numerical precision.
        """
        target_dtype = self._audio_encoder_dtype
        for component in (
            self.acoustic_tokenizer,
            self.semantic_tokenizer,
            self.acoustic_connector,
            self.semantic_connector,
        ):
            try:
                current_dtype = next(component.parameters()).dtype
                if current_dtype != target_dtype:
                    component.to(dtype=target_dtype)
            except StopIteration:
                pass

    def forward(
        self,
        audio: torch.Tensor,
        *,
        use_streaming: bool = True,
        segment_duration_s: Optional[float] = None,
    ) -> torch.Tensor:
        """Encode audio with optional streaming for long clips.

        Args:
            audio: Input audio tensor [B, T] or [T]
            use_streaming: Whether to enable segmented encoding
            segment_duration_s: Segment length in seconds (defaults to 60s)

        Returns:
            Audio embeddings tensor compatible with the language model.
        """
        from vibevoice.modular.modular_vibevoice_tokenizer import (
            VibeVoiceTokenizerEncoderOutput,
            VibeVoiceTokenizerStreamingCache,
        )

        self._ensure_audio_encoder_dtype()
        audio = audio.to(dtype=self._audio_encoder_dtype)

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        segment_duration = segment_duration_s or self.streaming_segment_duration
        total_samples = audio.shape[-1]
        segment_samples = int(segment_duration * self.sample_rate)

        use_streaming = (
            use_streaming
            and self.enable_streaming
            and total_samples > segment_samples
        )

        with torch.no_grad():
            if not use_streaming:
                acoustic_input = audio.unsqueeze(1)
                acoustic_out = self.acoustic_tokenizer.encode(acoustic_input)
                if self.use_sample:
                    acoustic_tokens = acoustic_out.sample(
                        dist_type=self.acoustic_tokenizer.std_dist_type
                    )[0]
                else:
                    acoustic_tokens = acoustic_out.mean
                acoustic_embeds = self.acoustic_connector(acoustic_tokens)

                semantic_out = self.semantic_tokenizer.encode(acoustic_input)
                semantic_tokens = semantic_out.mean
                semantic_embeds = self.semantic_connector(semantic_tokens)
            else:
                # Streaming path for long audio
                acoustic_cache = VibeVoiceTokenizerStreamingCache()
                semantic_cache = VibeVoiceTokenizerStreamingCache()
                acoustic_mean_segments: list[torch.Tensor] = []
                semantic_mean_segments: list[torch.Tensor] = []
                batch_size = audio.shape[0]
                sample_indices = torch.arange(
                    batch_size, device=audio.device
                )

                segments = [
                    (start, min(start + segment_samples, total_samples))
                    for start in range(0, total_samples, segment_samples)
                    if start < total_samples
                ]
                num_segments = len(segments)

                for seg_idx, (start, end) in enumerate(segments):
                    chunk = audio[:, start:end].contiguous()
                    if chunk.numel() == 0:
                        continue
                    is_final = seg_idx == num_segments - 1

                    acoustic_enc_out = self.acoustic_tokenizer.encode(
                        chunk.unsqueeze(1),
                        cache=acoustic_cache,
                        sample_indices=sample_indices,
                        use_cache=True,
                        is_final_chunk=is_final,
                    )
                    acoustic_mean_segments.append(acoustic_enc_out.mean)

                    semantic_enc_out = self.semantic_tokenizer.encode(
                        chunk.unsqueeze(1),
                        cache=semantic_cache,
                        sample_indices=sample_indices,
                        use_cache=True,
                        is_final_chunk=is_final,
                    )
                    semantic_mean_segments.append(semantic_enc_out.mean)

                if len(acoustic_mean_segments) == 0:
                    acoustic_mean_full = torch.zeros(
                        (batch_size, 0, self.acoustic_vae_dim),
                        device=audio.device,
                        dtype=self._audio_encoder_dtype,
                    )
                else:
                    acoustic_mean_full = torch.cat(
                        acoustic_mean_segments, dim=1
                    ).contiguous()

                acoustic_enc_full = VibeVoiceTokenizerEncoderOutput(
                    mean=acoustic_mean_full,
                    std=self.acoustic_tokenizer.fix_std,
                )
                if self.use_sample:
                    acoustic_tokens = acoustic_enc_full.sample(
                        dist_type=self.acoustic_tokenizer.std_dist_type
                    )[0]
                else:
                    acoustic_tokens = acoustic_enc_full.mean
                acoustic_embeds = self.acoustic_connector(acoustic_tokens)

                if len(semantic_mean_segments) == 0:
                    semantic_tokens = torch.zeros(
                        (batch_size, 0, self.semantic_vae_dim),
                        device=audio.device,
                        dtype=self._audio_encoder_dtype,
                    )
                else:
                    semantic_tokens = torch.cat(
                        semantic_mean_segments, dim=1
                    ).contiguous()
                semantic_embeds = self.semantic_connector(semantic_tokens)

        # Combine acoustic and semantic embeddings
        combined_embeds = acoustic_embeds + semantic_embeds
        # Convert to language model dtype for compatibility
        return combined_embeds.to(dtype=self._lm_dtype)


# ============================================================================
# vLLM Multimodal Processing Infrastructure
# ============================================================================


class VibeVoiceProcessingInfo(BaseProcessingInfo):
    """Processing info for VibeVoice multimodal model."""

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        """Get a WhisperFeatureExtractor for vLLM profiling.

        IMPORTANT: This is NOT used in actual inference!
        VibeVoice uses its own acoustic/semantic VAE tokenizers operating
        on raw 24kHz waveforms, NOT Whisper mel spectrograms.

        This exists to satisfy vLLM's multimodal profiling infrastructure
        which may query audio parameters for memory estimation.
        """
        import json
        import os

        model_path = self.ctx.model_config.model
        preprocessor_path = os.path.join(
            model_path, "preprocessor_config.json"
        )

        config = {
            "sampling_rate": 24000,
            "feature_size": 128,
            "hop_length": 240,
            "chunk_length": 30,
            "n_fft": 400,
            "padding_value": 0.0,
        }

        if os.path.exists(preprocessor_path):
            try:
                with open(preprocessor_path) as f:
                    file_config = json.load(f)
                    config.update(
                        {
                            k: file_config[k]
                            for k in config
                            if k in file_config
                        }
                    )
            except Exception:
                pass

        return WhisperFeatureExtractor(
            feature_size=config["feature_size"],
            sampling_rate=config["sampling_rate"],
            hop_length=config["hop_length"],
            chunk_length=config["chunk_length"],
            n_fft=config["n_fft"],
            padding_value=config["padding_value"],
        )

    def get_audio_token_info(self) -> dict:
        """Get audio special tokens and their IDs."""
        tokenizer = self.get_tokenizer()
        vocab = tokenizer.get_vocab()
        tokens = {
            "audio_token": "<|AUDIO|>",
            "audio_bos_token": "<|audio_bos|>",
            "audio_eos_token": "<|audio_eos|>",
        }
        tokens["audio_token_id"] = vocab.get(tokens["audio_token"])
        tokens["audio_bos_id"] = vocab.get(tokens["audio_bos_token"])
        tokens["audio_eos_id"] = vocab.get(tokens["audio_eos_token"])
        return tokens

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        """Return the maximum number of audio tokens per item.

        Formula: audio_tokens = ceil(audio_samples / compress_ratio) + 3
        where +3 accounts for speech_start, speech_end, and newline tokens.
        """
        hf_config = self.get_hf_config()
        compress_ratio = int(_get_cfg(hf_config, "speech_tok_compress_ratio", 3200))
        sample_rate = int(_get_cfg(hf_config, "target_sample_rate", 24000))

        # Upper bound: 61-minute audio at 24 kHz
        max_audio_samples = 61 * 60 * sample_rate
        max_audio_tokens = int(np.ceil(max_audio_samples / compress_ratio)) + 3
        max_audio_tokens = min(max_audio_tokens, seq_len)
        return {"audio": max_audio_tokens}


class VibeVoiceDummyInputsBuilder(
    BaseDummyInputsBuilder[VibeVoiceProcessingInfo]
):
    """Build dummy inputs for multimodal profiling."""

    def _get_max_audio_samples(self, seq_len: int) -> int:
        """Compute maximum audio samples consistent with max tokens."""
        hf_config = self.info.get_hf_config()
        compress_ratio = int(
            _get_cfg(hf_config, "speech_tok_compress_ratio", 3200)
        )
        sample_rate = int(_get_cfg(hf_config, "target_sample_rate", 24000))

        max_hour_samples = 61 * 60 * sample_rate
        max_tokens_from_audio = (
            int(np.ceil(max_hour_samples / compress_ratio)) + 3
        )
        max_tokens = min(max_tokens_from_audio, seq_len)
        return max_tokens * compress_ratio

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        if num_audios <= 0:
            return ""
        token_info = self.info.get_audio_token_info()
        return token_info["audio_token"] * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
        mm_processor_kwargs: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        """Generate dummy audio data for profiling."""
        num_audios = mm_counts.get("audio", 0)
        max_audio_len = self._get_max_audio_samples(seq_len)

        audio_overrides = mm_options.get("audio") if mm_options else None
        return {
            "audio": self._get_dummy_audios(
                length=max_audio_len,
                num_audios=num_audios,
                overrides=audio_overrides,
            )
        }

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> ProcessorInputs:
        """Build ProcessorInputs for dummy profiling."""
        return ProcessorInputs(
            prompt=self.get_dummy_text(mm_counts),
            mm_data=self.get_dummy_mm_data(seq_len, mm_counts, mm_options),
        )


def _vibevoice_field_config(
    hf_inputs: Mapping[str, torch.Tensor],
) -> dict[str, MultiModalFieldConfig]:
    """Map HF processor output keys to audio modality."""
    config: dict[str, MultiModalFieldConfig] = {
        "raw_audio": MultiModalFieldConfig.batched("audio"),
        "raw_audio_lengths": MultiModalFieldConfig.batched("audio"),
        "salt": MultiModalFieldConfig.batched("audio"),
    }
    if "input_features" in hf_inputs:
        config["input_features"] = MultiModalFieldConfig.batched("audio")
    if "feature_attention_mask" in hf_inputs:
        config["feature_attention_mask"] = MultiModalFieldConfig.batched(
            "audio"
        )
    return config


class VibeVoiceMultiModalProcessor(
    BaseMultiModalProcessor[VibeVoiceProcessingInfo]
):
    """Multimodal processor for VibeVoice.

    Handles the conversion of raw audio inputs to model-ready features,
    and manages the prompt token replacement for audio placeholders.
    """

    def _get_data_parser(self) -> MultiModalDataParser:
        """Create a data parser with the correct target sample rate."""
        return MultiModalDataParser(target_sr=24000)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Process prompt and audio for vLLM multimodal pipeline.

        We intentionally do NOT run a HF processor that would pre-expand
        ``<|AUDIO|>`` inside this method. Instead we:
        1) Tokenize the prompt as-is (``<|AUDIO|>`` stays a single token)
        2) Store raw audio tensors for ``embed_multimodal`` to encode later
        3) Let vLLM call ``_get_prompt_updates`` to expand the single
           ``<|AUDIO|>`` into the full ASR format.
        """
        import uuid

        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", None)
        if audios is not None and "audio" not in mm_data:
            mm_data["audio"] = audios

        if not mm_data.get("audio"):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(
                dict(input_ids=[prompt_ids]), tensor_type="pt"
            )

        raw_audio_list = mm_data.get("audio")
        if isinstance(raw_audio_list, np.ndarray):
            raw_audio_list = [raw_audio_list]
        elif not isinstance(raw_audio_list, list):
            raw_audio_list = list(raw_audio_list)

        tokenizer = self.info.get_tokenizer()
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)

        result = BatchFeature(
            dict(input_ids=[prompt_ids]), tensor_type="pt"
        )

        max_len = max(len(a) for a in raw_audio_list)
        raw_audio_tensors = []
        audio_lengths = []
        for audio in raw_audio_list:
            audio_len = len(audio)
            audio_lengths.append(audio_len)
            if audio_len < max_len:
                audio = np.pad(
                    audio, (0, max_len - audio_len), mode="constant"
                )
            raw_audio_tensors.append(torch.from_numpy(audio).float())

        stacked_audio = torch.stack(raw_audio_tensors, dim=0)
        result["raw_audio"] = stacked_audio
        result["raw_audio_lengths"] = torch.tensor(
            audio_lengths, dtype=torch.long
        )

        salt_val = hash(str(uuid.uuid4())) % 100000
        result["salt"] = torch.tensor([salt_val], dtype=torch.long).expand(
            len(raw_audio_list)
        )

        return result

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: object,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        """We handle token expansion via _get_prompt_updates."""
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _vibevoice_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: object,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Define how to replace the audio placeholder in the prompt.

        Expands the single ``<|AUDIO|>`` token into N repeated
        ``<|speech_pad|>`` tokens, wrapped in speech start/end markers.
        """
        token_info = self.info.get_audio_token_info()
        audio_token = token_info["audio_token"]
        audio_token_id = token_info["audio_token_id"]
        audio_bos_id = token_info.get("audio_bos_id")
        audio_eos_id = token_info.get("audio_eos_id")

        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        def _tok_id(name: str) -> int | None:
            return vocab.get(name)

        # Look up speech token IDs from vocabulary
        speech_start_id = (
            _tok_id("<|object_ref_start|>")
            or getattr(tokenizer, "speech_start_id", None)
            or _tok_id("<|speech_start|>")
        )
        speech_end_id = (
            _tok_id("<|object_ref_end|>")
            or getattr(tokenizer, "speech_end_id", None)
            or _tok_id("<|speech_end|>")
        )
        speech_pad_id = (
            _tok_id("<|box_start|>")
            or getattr(tokenizer, "speech_pad_id", None)
            or _tok_id("<|speech_pad|>")
        )

        if audio_token_id is None:
            return []

        out_mm_data = out_mm_kwargs.get_data()
        raw_audio_lengths = out_mm_data.get("raw_audio_lengths", [])

        hf_config = self.info.get_hf_config()
        compress_ratio = int(
            _get_cfg(hf_config, "speech_tok_compress_ratio", 3200)
        )

        def _to_int_len(x: object) -> int:
            if x is None:
                return 0
            if isinstance(x, torch.Tensor):
                if x.numel() == 1:
                    return int(x.item())
                return int(x.shape[0])
            return int(x)

        def get_replacement(item_idx: int) -> PromptUpdateDetails:
            if raw_audio_lengths and item_idx < len(raw_audio_lengths):
                audio_len = _to_int_len(raw_audio_lengths[item_idx])
                num_features = max(
                    1, int(np.ceil(audio_len / compress_ratio))
                )
            else:
                num_features = int(np.ceil(30 * 24000 / compress_ratio))

            if num_features == 0:
                raise ValueError(
                    f"Audio at index {item_idx} is too short to be "
                    "represented"
                )

            newline_id = 198  # '\n' token
            if (
                speech_start_id is not None
                and speech_pad_id is not None
                and speech_end_id is not None
            ):
                embed_id = int(speech_pad_id)
                replacement_ids = (
                    [int(speech_start_id)]
                    + [embed_id] * num_features
                    + [int(speech_end_id), newline_id]
                )
            elif audio_bos_id is not None and audio_eos_id is not None:
                embed_id = int(audio_token_id)
                replacement_ids = (
                    [int(audio_bos_id)]
                    + [embed_id] * num_features
                    + [int(audio_eos_id)]
                )
            else:
                embed_id = int(audio_token_id)
                replacement_ids = [embed_id] * num_features

            return PromptUpdateDetails.select_token_id(
                replacement_ids,
                embed_token_id=embed_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement,
            )
        ]


# ============================================================================
# Main Model Class
# ============================================================================


@MULTIMODAL_REGISTRY.register_processor(
    VibeVoiceMultiModalProcessor,
    info=VibeVoiceProcessingInfo,
    dummy_inputs=VibeVoiceDummyInputsBuilder,
)
class VibeVoiceForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
    """VibeVoice ASR model with native vLLM multimodal integration.

    Combines VibeVoice acoustic/semantic tokenizers for audio encoding
    with a Qwen2-based causal language model for text generation.
    """

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        """Return the placeholder string for a given modality.

        Returns ``<|AUDIO|>`` which vLLM inserts into the conversation
        prompt. This is later expanded by ``_get_prompt_updates``.
        """
        if modality.startswith("audio"):
            return "<|AUDIO|>"
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_encoder = VibeVoiceAudioEncoder(config)

        decoder_config = getattr(config, "decoder_config", config)

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=decoder_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Qwen2ForCausalLM"],
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # Set LM dtype for audio encoder output conversion
        lm_dtype = vllm_config.model_config.dtype
        if lm_dtype is not None:
            self.audio_encoder._lm_dtype = lm_dtype

    def compute_logits(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Extract audio embeddings using VibeVoice's tokenizers.

        Called by vLLM to get audio embeddings that replace audio
        placeholder tokens.
        """
        raw_audio = kwargs.get("raw_audio")
        raw_audio_lengths = kwargs.get("raw_audio_lengths")

        if raw_audio is None:
            return []

        if isinstance(raw_audio, (list, tuple)) and len(raw_audio) == 0:
            return []

        # Flatten raw_audio_lengths
        flat_lengths = _flatten_lengths(raw_audio_lengths)

        use_streaming_flag = bool(
            kwargs.get(
                "use_streaming",
                getattr(self.audio_encoder, "enable_streaming", True),
            )
        )
        streaming_segment_duration = kwargs.get(
            "streaming_segment_duration",
            getattr(self.audio_encoder, "streaming_segment_duration", 60.0),
        )

        embeddings: list[torch.Tensor] = []

        try:
            device = next(self.audio_encoder.parameters()).device
        except StopIteration:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        # Handle both stacked tensor and list of tensors
        if isinstance(raw_audio, torch.Tensor):
            if raw_audio.dim() == 3:
                audio_list = [
                    raw_audio[i].squeeze(0)
                    for i in range(raw_audio.shape[0])
                ]
            elif raw_audio.dim() == 2:
                audio_list = [
                    raw_audio[i] for i in range(raw_audio.shape[0])
                ]
            else:
                audio_list = [raw_audio]
        elif isinstance(raw_audio, (list, tuple)):
            audio_list = list(raw_audio)
        else:
            audio_list = [raw_audio]

        for i, audio_tensor in enumerate(audio_list):
            if isinstance(audio_tensor, list):
                audio_tensor = torch.stack(audio_tensor)
            if not isinstance(audio_tensor, torch.Tensor):
                audio_tensor = torch.tensor(audio_tensor)

            audio_tensor = audio_tensor.to(device=device)

            # Trim to actual length if available
            if flat_lengths and i < len(flat_lengths):
                actual_len = int(flat_lengths[i])
                if 0 < actual_len <= audio_tensor.shape[-1]:
                    audio_tensor = audio_tensor[..., :actual_len]

            # Skip audio too short for even 1 frame
            if audio_tensor.numel() < 160:
                continue

            audio_embeds = self.audio_encoder(
                audio_tensor,
                use_streaming=use_streaming_flag,
                segment_duration_s=streaming_segment_duration,
            )
            embeddings.append(audio_embeds.squeeze(0))

        return tuple(embeddings)

    def get_input_embeddings(self) -> nn.Module:
        """Return the text embedding layer (embed_tokens)."""
        if hasattr(self.language_model, "model") and hasattr(
            self.language_model.model, "embed_tokens"
        ):
            return self.language_model.model.embed_tokens
        if hasattr(self.language_model, "embed_tokens"):
            return self.language_model.embed_tokens
        inner = self.language_model
        if hasattr(inner, "language_model"):
            inner = inner.language_model
        if hasattr(inner, "model") and hasattr(inner.model, "embed_tokens"):
            return inner.model.embed_tokens
        raise AttributeError("Cannot find embed_tokens layer")

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Forward pass for VibeVoice ASR model."""
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if intermediate_tensors is not None:
            inputs_embeds = None

        language_model = self.language_model
        if hasattr(language_model, "language_model"):
            language_model = language_model.language_model

        hidden_states = language_model.model(
            input_ids=None,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """Load model weights from checkpoint.

        The checkpoint uses prefixes like:
        - model.acoustic_tokenizer.* -> audio_encoder.acoustic_tokenizer.*
        - model.semantic_tokenizer.* -> audio_encoder.semantic_tokenizer.*
        - model.acoustic_connector.* -> audio_encoder.acoustic_connector.*
        - model.semantic_connector.* -> audio_encoder.semantic_connector.*
        - model.language_model.*     -> language_model.model.*
        - lm_head.*                  -> language_model.lm_head.*
        """
        mapper = WeightsMapper(
            orig_to_new_prefix={
                "model.acoustic_tokenizer.": "audio_encoder.acoustic_tokenizer.",
                "model.semantic_tokenizer.": "audio_encoder.semantic_tokenizer.",
                "model.acoustic_connector.": "audio_encoder.acoustic_connector.",
                "model.semantic_connector.": "audio_encoder.semantic_connector.",
                "model.language_model.": "language_model.model.",
                "lm_head.": "language_model.lm_head.",
            }
        )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=mapper)


# Alias for training checkpoint compatibility
VibeVoiceForASRTraining = VibeVoiceForCausalLM


# ============================================================================
# Helper functions
# ============================================================================


def _flatten_lengths(lengths: object) -> list[int]:
    """Flatten nested lists/tensors of lengths to a single list."""
    if lengths is None:
        return []

    result: list[int] = []
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.tolist()

    if isinstance(lengths, (list, tuple)):
        for item in lengths:
            if isinstance(item, (list, tuple)):
                result.extend(_flatten_lengths(item))
            elif isinstance(item, torch.Tensor):
                if item.dim() == 0:
                    result.append(int(item.item()))
                else:
                    result.extend(int(v) for v in item.tolist())
            else:
                result.append(int(item))
    else:
        result.append(int(lengths))
    return result
