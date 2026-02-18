# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import List, Optional, Tuple, Union, Dict, Any, Iterable, Mapping, Sequence, ClassVar, Literal
from pathlib import Path
import torch
import torch.nn as nn
from transformers import PretrainedConfig, Qwen2Config
from vllm.transformers_utils.configs.vibevoice import (
    VibeVoiceConfig, VibeVoiceAcousticTokenizerConfig,
    VibeVoiceSemanticTokenizerConfig)

import math
import typing as tp
from functools import partial
from dataclasses import dataclass, field
import copy
import numpy as np
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from vllm.model_executor.layers.layernorm import RMSNorm as LlamaRMSNorm

# --- Normalization modules ---
class ConvLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape: tp.Union[int, tp.List[int], torch.Size], **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = nn.functional.layer_norm(x.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps).type_as(x)
        x = x.transpose(1, 2)
        return x

class ConvRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine=True, weight_shape=None):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            weight_shape = (dim,) if weight_shape is None else weight_shape
            self.weight = nn.Parameter(torch.ones(weight_shape))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x = x.transpose(1, 2)
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        output = output.transpose(1, 2)
        return output

# --- Convolutional utilities ---
def pad1d(x: torch.Tensor, paddings: tp.Tuple[int, int], mode: str = 'zero', value: float = 0.):
    if mode == 'zero':
        return F.pad(x, paddings, 'constant', value)
    elif mode == 'reflect':
        return F.pad(x, paddings, 'reflect')
    elif mode == 'replicate':
        return F.pad(x, paddings, 'replicate')
    elif mode == 'constant':
        return F.pad(x, paddings, 'constant', value)
    else:
        raise ValueError(f"Unknown padding mode: {mode}")

def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int, padding_total: int) -> int:
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_frames = math.ceil(n_frames)
    target_length = (ideal_frames - 1) * stride + kernel_size - padding_total
    return max(0, target_length - length)

class NormConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, causal=False, norm='none', norm_kwargs={}, pad_mode='reflect'):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation, groups=groups, bias=bias)
        self.causal = causal

    def forward(self, x):
        return self.conv(x)

class NormConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, causal=False, norm='none', norm_kwargs={}, bias=True):
        super().__init__()
        self.convtr = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, bias=bias)

    def forward(self, x):
        return self.convtr(x)

class SConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, causal=False, norm='none', norm_kwargs={}, pad_mode='reflect'):
        super().__init__()
        self.conv = NormConv1d(in_channels, out_channels, kernel_size, stride, dilation, groups, bias, causal, norm, norm_kwargs, pad_mode)
        self.causal = causal
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad_mode = pad_mode
        self.padding_total = (kernel_size - 1) * dilation - (stride - 1)
        self.context_size = (kernel_size - 1) * dilation - (stride - 1)
        self._layer_id = None

    @property
    def layer_id(self):
        if self._layer_id is None:
            self._layer_id = f"sconv1d_{id(self)}"
        return self._layer_id

    def forward(self, x, cache=None, sample_indices=None, use_cache=False, debug=False, is_final_chunk=False):
        if not use_cache or cache is None:
            # Non-streaming
            padding_total = self.padding_total
            extra_padding = get_extra_padding_for_conv1d(x, self.kernel_size, self.stride, padding_total)
            if self.causal:
                x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
            else:
                padding_right = padding_total // 2
                padding_left = padding_total - padding_right
                x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
            return self.conv(x)

        # Streaming
        B, C, T = x.shape
        cached_states = cache.get(self.layer_id, sample_indices)
        if cached_states is None:
            cached_states = torch.zeros(B, C, max(0, self.context_size), device=x.device, dtype=x.dtype)

        input_with_context = torch.cat([cached_states, x], dim=2) if cached_states.shape[2] > 0 else x
        if is_final_chunk:
            extra_padding = get_extra_padding_for_conv1d(input_with_context, self.kernel_size, self.stride, self.padding_total)
            if extra_padding > 0:
                input_with_context = pad1d(input_with_context, (0, extra_padding), mode=self.pad_mode)

        output = self.conv(input_with_context)
        if self.context_size > 0:
            total_input_length = input_with_context.shape[2]
            new_cache = input_with_context[:, :, max(0, total_input_length - self.context_size):]
            cache.set(self.layer_id, sample_indices, new_cache)
        return output

class SConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, causal=False, norm='none', trim_right_ratio=1., norm_kwargs={}, bias=True):
        super().__init__()
        self.convtr = NormConvTranspose1d(in_channels, out_channels, kernel_size, stride, causal, norm, norm_kwargs, bias)
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_total = kernel_size - stride
        self.context_size = kernel_size - 1
        self._layer_id = None

    @property
    def layer_id(self):
        if self._layer_id is None:
            self._layer_id = f"sconvtr1d_{id(self)}"
        return self._layer_id

    def forward(self, x, cache=None, sample_indices=None, use_cache=False, debug=False):
        if not use_cache or cache is None:
            output = self.convtr(x)
            padding_total = self.padding_total
            if self.causal:
                padding_right = math.ceil(padding_total * self.trim_right_ratio)
                padding_left = padding_total - padding_right
                if padding_right > 0:
                    output = output[..., padding_left:-padding_right]
                else:
                    output = output[..., padding_left:]
            else:
                padding_right = padding_total // 2
                padding_left = padding_total - padding_right
                if padding_right > 0:
                    output = output[..., padding_left:-padding_right]
                else:
                    output = output[..., padding_left:]
            return output

        # Streaming
        B, C, T = x.shape
        cached_input = cache.get(self.layer_id, sample_indices)
        if cached_input is None:
            cached_input = torch.zeros(B, C, self.context_size, device=x.device, dtype=x.dtype)

        input_with_context = torch.cat([cached_input, x], dim=2)
        output = self.convtr(input_with_context)
        output = output[..., -T * self.stride:]
        new_cache = input_with_context[:, :, -self.context_size:]
        cache.set(self.layer_id, sample_indices, new_cache)
        return output

class Mixer(nn.Module):
    def __init__(self, dim, mixer_layer='conv', kernel_size=7, causal=True, pad_mode='reflect', norm='none', bias=True):
        super().__init__()
        if mixer_layer == 'depthwise_conv':
            self.conv = SConv1d(dim, dim, kernel_size, groups=dim, causal=causal, pad_mode=pad_mode, norm=norm, bias=bias)
        else:
            self.conv = nn.Identity()

    def forward(self, x, **kwargs):
        if isinstance(self.conv, SConv1d):
            return self.conv(x, **kwargs)
        return self.conv(x)

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim=None, act_fn='silu', bias=True):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Block1D(nn.Module):
    def __init__(self, dim, mixer_layer='conv', kernel_size=7, causal=True, pad_mode='reflect', norm='none', bias=True, layernorm='LN', eps=1e-6, layer_scale_init_value=0, drop_path=0.):
        super().__init__()
        if layernorm == 'LN':
            self.norm = ConvLayerNorm(dim, eps=eps)
            self.ffn_norm = nn.LayerNorm(dim, eps=eps)
        else:
            self.norm = ConvRMSNorm(dim, eps=eps)
            self.ffn_norm = LlamaRMSNorm(dim, eps=eps)

        self.mixer = Mixer(dim, mixer_layer, kernel_size, causal, pad_mode, norm, bias)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None

        self.ffn = FFN(dim)
        self.ffn_gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None

    def forward(self, x, **kwargs):
        residual = x
        x = self.norm(x)
        x = self.mixer(x, **kwargs)
        if self.gamma is not None:
            x = x * self.gamma.unsqueeze(-1)
        x = residual + x

        residual = x
        # x is [B, C, T], ffn_norm expects [B, T, C]
        x = self.ffn_norm(x.transpose(1, 2))
        x = self.ffn(x)
        x = x.transpose(1, 2) # Back to [B, C, T]
        if self.ffn_gamma is not None:
            x = x * self.ffn_gamma.unsqueeze(-1)
        x = residual + x
        return x

class VibeVoiceTokenizerStreamingCache:
    def __init__(self):
        self.states = {}
    def get(self, layer_id, indices):
        return self.states.get(layer_id)
    def set(self, layer_id, indices, state):
        self.states[layer_id] = state

@dataclass
class VibeVoiceTokenizerEncoderOutput:
    mean: torch.Tensor
    std: Optional[Union[float, torch.Tensor]] = None
    def sample(self, dist_type='fix'):
        if dist_type == 'fix':
            return self.mean + self.std * torch.randn_like(self.mean), self.std
        elif dist_type == 'gaussian':
            batch_size = self.mean.size(0)
            std = torch.randn(batch_size, 1, 1, device=self.mean.device, dtype=self.mean.dtype) * (self.std / 0.8)
            return self.mean + std * torch.randn_like(self.mean), std
        return self.mean, self.std

class TokenizerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dimension = config.dimension
        self.n_filters = config.n_filters
        self.ratios = config.ratios
        self.depths = config.depths
        self.causal = config.causal
        self.pad_mode = config.pad_mode

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.ModuleList([
            SConv1d(config.channels, self.n_filters, kernel_size=7, causal=self.causal, pad_mode=self.pad_mode)
        ]))

        for i, ratio in enumerate(self.ratios):
            in_ch = self.n_filters * (2**i)
            out_ch = self.n_filters * (2**(i+1))
            self.downsample_layers.append(nn.ModuleList([
                SConv1d(in_ch, out_ch, kernel_size=ratio*2, stride=ratio, causal=self.causal, pad_mode=self.pad_mode)
            ]))

        self.stages = nn.ModuleList()
        for i in range(len(self.depths)):
            ch = self.n_filters * (2**i)
            self.stages.append(nn.ModuleList([
                Block1D(ch, mixer_layer=config.mixer_layer, causal=self.causal, pad_mode=self.pad_mode, layernorm=config.layernorm)
                for _ in range(self.depths[i])
            ]))

        self.norm = ConvRMSNorm(out_ch) if config.layernorm == 'RMSNorm' else ConvLayerNorm(out_ch)
        self.head = SConv1d(out_ch, self.dimension, kernel_size=3, causal=self.causal, pad_mode=self.pad_mode)

    def forward(self, x, **kwargs):
        for i in range(len(self.depths)):
            for layer in self.downsample_layers[i]:
                x = layer(x, **kwargs)
            for block in self.stages[i]:
                x = block(x, **kwargs)
        x = self.norm(x)
        x = self.head(x, **kwargs)
        return x

class VibeVoiceAcousticTokenizerModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.fix_std = config.fix_std
        self.std_dist_type = getattr(config, "std_dist_type", "fix")
        encoder_depths = [int(d) for d in config.encoder_depths.split('-')] if isinstance(config.encoder_depths, str) else config.encoder_depths

        enc_cfg = copy.deepcopy(config)
        enc_cfg.dimension = config.vae_dim
        enc_cfg.depths = encoder_depths
        self.encoder = TokenizerEncoder(enc_cfg)

    def encode(self, audio, **kwargs):
        latents = self.encoder(audio, **kwargs)
        return VibeVoiceTokenizerEncoderOutput(mean=latents.permute(0, 2, 1), std=self.fix_std)

class VibeVoiceSemanticTokenizerModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        encoder_depths = [int(d) for d in config.encoder_depths.split('-')] if isinstance(config.encoder_depths, str) else config.encoder_depths

        enc_cfg = copy.deepcopy(config)
        enc_cfg.dimension = config.vae_dim
        enc_cfg.depths = encoder_depths
        self.encoder = TokenizerEncoder(enc_cfg)

    def encode(self, audio, **kwargs):
        latents = self.encoder(audio, **kwargs)
        return VibeVoiceTokenizerEncoderOutput(mean=latents.permute(0, 2, 1))

class SpeechConnector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = LlamaRMSNorm(output_dim, eps=1e-6)
        self.fc2 = nn.Linear(output_dim, output_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x

class VibeVoiceAudioEncoder(nn.Module):
    def __init__(self, config: VibeVoiceConfig):
        super().__init__()
        self.config = config
        self.acoustic_vae_dim = config.acoustic_vae_dim
        self.semantic_vae_dim = config.semantic_vae_dim
        self.hidden_size = config.hidden_size
        self.acoustic_tokenizer = VibeVoiceAcousticTokenizerModel(config.acoustic_tokenizer_config)
        self.semantic_tokenizer = VibeVoiceSemanticTokenizerModel(config.semantic_tokenizer_config)
        self.acoustic_connector = SpeechConnector(self.acoustic_vae_dim, self.hidden_size)
        self.semantic_connector = SpeechConnector(self.semantic_vae_dim, self.hidden_size)
        self.sample_rate = getattr(config, "target_sample_rate", 24000)
        self.enable_streaming = getattr(config, "enable_streaming", True)
        self.streaming_segment_duration = getattr(config, "streaming_segment_duration", 60.0)

    def forward(self, audio: torch.Tensor, use_streaming: bool = True) -> torch.Tensor:
        total_samples = audio.shape[-1]
        segment_samples = int(self.streaming_segment_duration * self.sample_rate)
        use_streaming = use_streaming and self.enable_streaming and total_samples > segment_samples
        if not use_streaming:
            acoustic_input = audio.unsqueeze(1)
            acoustic_out = self.acoustic_tokenizer.encode(acoustic_input)
            acoustic_embeds = self.acoustic_connector(acoustic_out.mean)
            semantic_out = self.semantic_tokenizer.encode(acoustic_input)
            semantic_embeds = self.semantic_connector(semantic_out.mean)
        else:
            acoustic_cache = VibeVoiceTokenizerStreamingCache()
            semantic_cache = VibeVoiceTokenizerStreamingCache()
            acoustic_mean_segments = []
            semantic_mean_segments = []
            batch_size = audio.shape[0]
            sample_indices = torch.arange(batch_size, device=audio.device)
            for start in range(0, total_samples, segment_samples):
                end = min(start + segment_samples, total_samples)
                chunk = audio[:, start:end].contiguous()
                is_final = (end == total_samples)
                ac_enc_out = self.acoustic_tokenizer.encode(chunk.unsqueeze(1), cache=acoustic_cache, sample_indices=sample_indices, use_cache=True, is_final_chunk=is_final)
                acoustic_mean_segments.append(ac_enc_out.mean)
                se_enc_out = self.semantic_tokenizer.encode(chunk.unsqueeze(1), cache=semantic_cache, sample_indices=sample_indices, use_cache=True, is_final_chunk=is_final)
                semantic_mean_segments.append(se_enc_out.mean)
            acoustic_embeds = self.acoustic_connector(torch.cat(acoustic_mean_segments, dim=1))
            semantic_embeds = self.semantic_connector(torch.cat(semantic_mean_segments, dim=1))
        return acoustic_embeds + semantic_embeds

from .interfaces import SupportsMultiModal, SupportsPP, MultiModalEmbeddings
from .utils import init_vllm_registered_model, maybe_prefix, AutoWeightsLoader, WeightsMapper
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors

class VibeVoiceForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_encoder = VibeVoiceAudioEncoder(self.config)
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(vllm_config=vllm_config, hf_config=self.config.decoder_config, prefix=maybe_prefix(prefix, "language_model"), architectures=["Qwen2ForCausalLM"])
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio = kwargs.pop("audio", None)
        if audio is None: return []
        if isinstance(audio, torch.Tensor):
            if audio.ndim == 1: audio = audio.unsqueeze(0)
            return self.audio_encoder(audio).unbind(dim=0)
        elif isinstance(audio, list):
            return tuple(self.audio_encoder(a.unsqueeze(0) if a.ndim == 1 else a).squeeze(0) for a in audio)
        return []
    def embed_input_ids(self, input_ids: torch.Tensor, multimodal_embeddings: Optional[MultiModalEmbeddings] = None, *, is_multimodal: Optional[torch.Tensor] = None, **kwargs: object) -> torch.Tensor:
        from vllm.model_executor.models.utils import _merge_multimodal_embeddings
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        if multimodal_embeddings is not None and is_multimodal is not None:
            inputs_embeds = _merge_multimodal_embeddings(inputs_embeds, multimodal_embeddings, is_multimodal)
        return inputs_embeds
    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None and input_ids is not None: inputs_embeds = self.embed_input_ids(input_ids)
        if intermediate_tensors is not None: inputs_embeds = None
        return self.language_model.model(input_ids=None if inputs_embeds is not None else input_ids, positions=positions, intermediate_tensors=intermediate_tensors, inputs_embeds=inputs_embeds)
    def compute_logits(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states)
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        mapper = WeightsMapper(orig_to_new_prefix={"model.acoustic_tokenizer.": "audio_encoder.acoustic_tokenizer.", "model.semantic_tokenizer.": "audio_encoder.semantic_tokenizer.", "model.acoustic_connector.": "audio_encoder.acoustic_connector.", "model.semantic_connector.": "audio_encoder.semantic_connector.", "model.language_model.": "language_model.model.", "lm_head.": "language_model.lm_head."})
        return AutoWeightsLoader(self).load_weights(weights, mapper=mapper)

VibeVoiceForASRTraining = VibeVoiceForCausalLM

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
    BaseDummyInputsBuilder,
    ProcessorInputs
)
from transformers import WhisperFeatureExtractor, BatchFeature

class VibeVoiceProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self): return self.ctx.get_hf_config()
    def get_feature_extractor(self, **kwargs): return WhisperFeatureExtractor(feature_size=128, sampling_rate=24000, hop_length=240, chunk_length=30, n_fft=400)
    def get_audio_token_info(self):
        tokenizer = self.get_tokenizer()
        vocab = tokenizer.get_vocab()
        tokens = {"audio_token": "<|AUDIO|>", "audio_bos_token": "<|audio_bos|>", "audio_eos_token": "<|audio_eos|>"}
        tokens.update({f"{k}_id": vocab.get(v) for k, v in tokens.items()})
        return tokens
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: return {"audio": None}

class VibeVoiceDummyInputsBuilder(BaseDummyInputsBuilder[VibeVoiceProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num = mm_counts.get("audio", 0)
        return self.info.get_audio_token_info()["audio_token"] * num if num > 0 else ""
    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options=None):
        num = mm_counts.get("audio", 0)
        return {"audio": [np.zeros(30 * 24000, dtype=np.float32) for _ in range(num)]}
    def get_dummy_processor_inputs(self, seq_len, mm_counts, mm_options=None):
        return ProcessorInputs(prompt=self.get_dummy_text(mm_counts), mm_data=self.get_dummy_mm_data(seq_len, mm_counts, mm_options))

class VibeVoiceMultiModalProcessor(BaseMultiModalProcessor[VibeVoiceProcessingInfo]):
    def _get_data_parser(self): return MultiModalDataParser(target_sr=24000)
    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        from .vibevoice_utils import load_audio_use_ffmpeg, load_audio_bytes_use_ffmpeg, AudioNormalizer
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", mm_data.get("audio"))
        if not audios:
            prompt_ids = self._apply_hf_processor_tokens_only(self.info.get_tokenizer().encode(prompt))
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")
        if not isinstance(audios, list): audios = [audios]

        # Load audio using FFmpeg if not already loaded
        loaded_audios = []
        normalizer = AudioNormalizer()
        for audio_item in audios:
            if isinstance(audio_item, (str, Path)):
                audio, _ = load_audio_use_ffmpeg(str(audio_item), resample=True, target_sr=24000)
                loaded_audios.append(normalizer(audio))
            elif isinstance(audio_item, bytes):
                audio, _ = load_audio_bytes_use_ffmpeg(audio_item, resample=True, target_sr=24000)
                loaded_audios.append(normalizer(audio))
            elif isinstance(audio_item, np.ndarray):
                loaded_audios.append(audio_item)
            elif isinstance(audio_item, torch.Tensor):
                loaded_audios.append(audio_item.cpu().numpy())
            else:
                loaded_audios.append(audio_item)

        prompt_ids = self._apply_hf_processor_tokens_only(self.info.get_tokenizer().encode(prompt, add_special_tokens=False))
        result = BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")
        max_l = max(len(a) for a in loaded_audios)
        tensors = [torch.from_numpy(np.pad(a, (0, max_l - len(a)), mode='constant')).float() for a in loaded_audios]
        result["raw_audio"] = torch.stack(tensors, dim=0)
        result["raw_audio_lengths"] = torch.tensor([len(a) for a in loaded_audios], dtype=torch.long)
        return result
    def _hf_processor_applies_updates(self, *args, **kwargs): return False
    def _get_mm_fields_config(self, hf_inputs, hf_proc_mm_kwargs): return {"raw_audio": MultiModalFieldConfig.batched("audio"), "raw_audio_lengths": MultiModalFieldConfig.batched("audio")}
    def _get_prompt_updates(self, mm_items, hf_proc_mm_kwargs, out_mm_kwargs):
        ti = self.info.get_audio_token_info()
        vocab = self.info.get_tokenizer().get_vocab()
        def _id(n): return vocab.get(n)
        s_start, s_end, s_pad = _id("<|object_ref_start|>") or _id("<|speech_start|>"), _id("<|object_ref_end|>") or _id("<|speech_end|>"), _id("<|box_start|>") or _id("<|speech_pad|>")
        if ti["audio_token_id"] is None: return []
        lengths = out_mm_kwargs.get_data().get("raw_audio_lengths", [])
        ratio = int(getattr(self.info.get_hf_config(), "speech_tok_compress_ratio", 3200))
        def get_repl(idx):
            num = max(1, int(np.ceil(int(lengths[idx]) / ratio))) if idx < len(lengths) else int(np.ceil(30 * 24000 / ratio))
            if s_start and s_pad and s_end: return PromptUpdateDetails.select_token_id([int(s_start)] + [int(s_pad)] * num + [int(s_end), 198], embed_token_id=int(s_pad))
            if ti["audio_bos_id"] and ti["audio_eos_id"]: return PromptUpdateDetails.select_token_id([int(ti["audio_bos_id"])] + [int(ti["audio_token_id"])] * num + [int(ti["audio_eos_id"])], embed_token_id=int(ti["audio_token_id"]))
            return PromptUpdateDetails.select_token_id([int(ti["audio_token_id"])] * num, embed_token_id=int(ti["audio_token_id"]))
        return [PromptReplacement(modality="audio", target=ti["audio_token"], replacement=get_repl)]

MULTIMODAL_REGISTRY.register_processor(VibeVoiceMultiModalProcessor, info=VibeVoiceProcessingInfo, dummy_inputs=VibeVoiceDummyInputsBuilder)

from transformers import Qwen2TokenizerFast
class VibeVoiceASRTextTokenizerFast(Qwen2TokenizerFast):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    @property
    def speech_start_id(self): return self.convert_tokens_to_ids("<|object_ref_start|>")
    @property
    def speech_end_id(self): return self.convert_tokens_to_ids("<|object_ref_end|>")
    @property
    def speech_pad_id(self): return self.convert_tokens_to_ids("<|box_start|>")

from transformers import AutoTokenizer, AutoConfig, AutoProcessor, Qwen2AudioProcessor
try:
    AutoConfig.register("vibevoice", VibeVoiceConfig)
    AutoTokenizer.register(VibeVoiceConfig, slow_tokenizer_class=None, fast_tokenizer_class=VibeVoiceASRTextTokenizerFast)
    AutoProcessor.register(VibeVoiceConfig, processor_class=Qwen2AudioProcessor)
except Exception: pass
