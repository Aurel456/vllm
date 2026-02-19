# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
VibeVoice ASR model implementation for vLLM.

This module implements the VibeVoice ASR model with full vLLM multimodal registry
integration for speech-to-text inference. It includes the model architecture,
configuration, tokenizer, and processor components.
"""
import copy
import math
import os
import threading
from dataclasses import dataclass
from functools import partial
from subprocess import run
from typing import (Any, Dict, Iterable, List, Literal, Mapping, Optional,
                    Sequence, Tuple, TypeAlias, Union)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoConfig, AutoModel, BatchFeature, PretrainedConfig,
                          Qwen2Config)
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.models.whisper import WhisperFeatureExtractor
from transformers.utils import logging

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (MultiModalEmbeddings,
                                                   SupportsMultiModal,
                                                   SupportsPP)
from vllm.model_executor.models.utils import (AutoWeightsLoader, WeightsMapper,
                                              init_vllm_registered_model,
                                              maybe_prefix)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalFieldConfig,
                                    MultiModalKwargsItems)
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing import (BaseDummyInputsBuilder,
                                        BaseMultiModalProcessor,
                                        BaseProcessingInfo, ProcessorInputs,
                                        PromptReplacement, PromptUpdate,
                                        PromptUpdateDetails)
from vllm.sequence import IntermediateTensors

logger = logging.get_logger(__name__)

# Try to import APEX FusedRMSNorm
try:
    from apex.normalization.fused_layer_norm import fused_rms_norm_affine
    APEX_AVAILABLE = True
    if int(os.getenv("OPTIMIZE_FOR_SPEED", "0")) == 0:
        APEX_AVAILABLE = False
except ImportError:
    APEX_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

def _convert_dtype_to_string(config_dict: dict) -> dict:
    if "torch_dtype" in config_dict and config_dict["torch_dtype"] is not None:
        dtype = config_dict["torch_dtype"]
        if isinstance(dtype, torch.dtype):
            config_dict["torch_dtype"] = str(dtype).replace("torch.", "")
    return config_dict


class VibeVoiceAcousticTokenizerConfig(PretrainedConfig):
    model_type = "vibevoice_acoustic_tokenizer"

    def __init__(
        self,
        channels: int = 1,
        corpus_normalize: float = 0.0,
        causal: bool = True,
        vae_dim: int = 64,
        fix_std: float = 0.5,
        std_dist_type: str = 'gaussian',
        mixer_layer: str = 'depthwise_conv',
        conv_norm: str = 'none',
        pad_mode: str = 'constant',
        disable_last_norm: bool = True,
        layernorm: str = 'RMSNorm',
        layernorm_eps: float = 1e-5,
        layernorm_elementwise_affine: bool = True,
        conv_bias: bool = True,
        layer_scale_init_value: float = 1e-6,
        weight_init_value: float = 1e-2,
        encoder_n_filters: int = 32,
        encoder_ratios: Optional[List[int]] = None,
        encoder_depths: str = "3-3-3-3-3-3-8",
        decoder_n_filters: int = 32,
        decoder_ratios: Optional[List[int]] = None,
        decoder_depths: Optional[str] = None,
        **kwargs
    ):
        if encoder_ratios is None:
            encoder_ratios = [8, 5, 5, 4, 2, 2]
        super().__init__(**kwargs)
        self.channels = channels
        self.corpus_normalize = corpus_normalize
        self.causal = causal
        self.vae_dim = vae_dim
        self.fix_std = fix_std
        self.std_dist_type = std_dist_type

        self.conv_norm = conv_norm
        self.pad_mode = pad_mode
        self.layernorm_eps = layernorm_eps
        self.disable_last_norm = disable_last_norm
        self.layernorm = layernorm
        self.layernorm_elementwise_affine = layernorm_elementwise_affine
        self.conv_bias = conv_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.weight_init_value = weight_init_value
        self.mixer_layer = mixer_layer

        self.encoder_n_filters = encoder_n_filters
        self.encoder_ratios = encoder_ratios
        self.encoder_depths = encoder_depths

        self.decoder_ratios = decoder_ratios if decoder_ratios is not None else encoder_ratios
        self.decoder_n_filters = decoder_n_filters
        self.decoder_depths = decoder_depths


class VibeVoiceSemanticTokenizerConfig(PretrainedConfig):
    model_type = "vibevoice_semantic_tokenizer"

    def __init__(
        self,
        channels: int = 1,
        corpus_normalize: float = 0.0,
        causal: bool = True,
        vae_dim: int = 64,
        fix_std: float = 0,
        std_dist_type: str = 'none',
        mixer_layer: str = 'depthwise_conv',
        conv_norm: str = 'none',
        pad_mode: str = 'constant',
        disable_last_norm: bool = True,
        layernorm: str = 'RMSNorm',
        layernorm_eps: float = 1e-5,
        layernorm_elementwise_affine: bool = True,
        conv_bias: bool = True,
        layer_scale_init_value: float = 1e-6,
        weight_init_value: float = 1e-2,
        encoder_n_filters: int = 32,
        encoder_ratios: Optional[List[int]] = None,
        encoder_depths: str = "3-3-3-3-3-3-8",
        **kwargs
    ):
        if encoder_ratios is None:
            encoder_ratios = [8, 5, 5, 4, 2, 2]
        super().__init__(**kwargs)
        self.channels = channels
        self.corpus_normalize = corpus_normalize
        self.causal = causal
        self.vae_dim = vae_dim
        self.fix_std = fix_std
        self.std_dist_type = std_dist_type

        self.conv_norm = conv_norm
        self.pad_mode = pad_mode
        self.layernorm_eps = layernorm_eps
        self.disable_last_norm = disable_last_norm
        self.layernorm = layernorm
        self.layernorm_elementwise_affine = layernorm_elementwise_affine
        self.conv_bias = conv_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.weight_init_value = weight_init_value
        self.mixer_layer = mixer_layer

        self.encoder_n_filters = encoder_n_filters
        self.encoder_ratios = encoder_ratios
        self.encoder_depths = encoder_depths


class VibeVoiceConfig(PretrainedConfig):
    model_type = "vibevoice"
    is_composition = True
    sub_configs = {
        "acoustic_tokenizer_config": VibeVoiceAcousticTokenizerConfig,
        "semantic_tokenizer_config": VibeVoiceSemanticTokenizerConfig,
        "decoder_config": Qwen2Config,
    }

    def __init__(
        self,
        acoustic_tokenizer_config=None,
        semantic_tokenizer_config=None,
        decoder_config=None,
        **kwargs
    ):
        kwargs["_attn_implementation_autoset"] = False

        if acoustic_tokenizer_config is None:
            self.acoustic_tokenizer_config = self.sub_configs["acoustic_tokenizer_config"]()
        elif isinstance(acoustic_tokenizer_config, dict):
            acoustic_tokenizer_config["model_type"] = "vibevoice_acoustic_tokenizer"
            self.acoustic_tokenizer_config = self.sub_configs["acoustic_tokenizer_config"](**acoustic_tokenizer_config)
        elif isinstance(acoustic_tokenizer_config, VibeVoiceAcousticTokenizerConfig):
            self.acoustic_tokenizer_config = acoustic_tokenizer_config

        if semantic_tokenizer_config is None:
            self.semantic_tokenizer_config = self.sub_configs["semantic_tokenizer_config"]()
        elif isinstance(semantic_tokenizer_config, dict):
            semantic_tokenizer_config["model_type"] = "vibevoice_semantic_tokenizer"
            self.semantic_tokenizer_config = self.sub_configs["semantic_tokenizer_config"](**semantic_tokenizer_config)
        elif isinstance(semantic_tokenizer_config, VibeVoiceSemanticTokenizerConfig):
            self.semantic_tokenizer_config = semantic_tokenizer_config

        if decoder_config is None:
            self.decoder_config = self.sub_configs["decoder_config"]()
        elif isinstance(decoder_config, dict):
            if decoder_config.get("model_type", '') == "qwen2":
                self.decoder_config = Qwen2Config(**decoder_config)
            else:
                raise ValueError(f"Unsupported decoder model type: {decoder_config.get('model_type', '')}")
        elif isinstance(decoder_config, Qwen2Config):
            self.decoder_config = decoder_config

        self.acoustic_vae_dim = getattr(self.acoustic_tokenizer_config, 'vae_dim', 64)
        self.semantic_vae_dim = getattr(self.semantic_tokenizer_config, 'vae_dim', 128)

        super().__init__(**kwargs)

    def get_text_config(self, decoder=False):
        return self.decoder_config

    def to_dict(self):
        output = super().to_dict()
        return _convert_dtype_to_string(output)

    @property
    def vocab_size(self):
        return self.decoder_config.vocab_size

    @vocab_size.setter
    def vocab_size(self, value):
        self.decoder_config.vocab_size = value

    @property
    def num_attention_heads(self):
        return self.decoder_config.num_attention_heads

    @num_attention_heads.setter
    def num_attention_heads(self, value):
        self.decoder_config.num_attention_heads = value

    @property
    def num_key_value_heads(self):
        return self.decoder_config.num_key_value_heads

    @num_key_value_heads.setter
    def num_key_value_heads(self, value):
        self.decoder_config.num_key_value_heads = value

    @property
    def hidden_size(self):
        return self.decoder_config.hidden_size

    @hidden_size.setter
    def hidden_size(self, value):
        self.decoder_config.hidden_size = value

    @property
    def num_hidden_layers(self):
        return self.decoder_config.num_hidden_layers

    @num_hidden_layers.setter
    def num_hidden_layers(self, value):
        self.decoder_config.num_hidden_layers = value

    @property
    def head_dim(self):
        return getattr(self.decoder_config, 'head_dim', self.hidden_size // self.num_attention_heads)


# Register Configs
AutoConfig.register("vibevoice", VibeVoiceConfig)
AutoConfig.register("vibevoice_acoustic_tokenizer", VibeVoiceAcousticTokenizerConfig)
AutoConfig.register("vibevoice_semantic_tokenizer", VibeVoiceSemanticTokenizerConfig)


# ============================================================================
# Tokenizer Models (Acoustic & Semantic)
# ============================================================================

class ConvLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape: Union[int, List[int], torch.Size], **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = nn.functional.layer_norm(x.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps).type_as(x)
        x = x.transpose(1, 2)
        return x

class RMSNorm(nn.Module):
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
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

class ConvRMSNorm(RMSNorm):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine=True, weight_shape=None):
        super().__init__(dim, eps, elementwise_affine, weight_shape)

    def forward(self, x):
        x = x.transpose(1, 2)
        if (not APEX_AVAILABLE) or (not self.elementwise_affine):
            output = self._norm(x.float()).type_as(x)
            if self.weight is not None:
                output = output * self.weight
        else:
            output = fused_rms_norm_affine(x, self.weight, self.weight.shape, self.eps)
        output = output.transpose(1, 2)
        return output

CONV_NORMALIZATIONS = frozenset(['none', 'weight_norm', 'spectral_norm',
                                'time_layer_norm', 'layer_norm', 'time_group_norm'])

def apply_parametrization_norm(module: nn.Module, norm: str = 'none') -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        return nn.utils.weight_norm(module)
    elif norm == 'spectral_norm':
        return nn.utils.spectral_norm(module)
    else:
        return module

def get_norm_module(module: nn.Module, causal: bool = False, norm: str = 'none', **norm_kwargs) -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == 'layer_norm':
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()

def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0) -> int:
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length

def pad1d(x: torch.Tensor, paddings: Tuple[int, int], mode: str = 'zero', value: float = 0.):
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)

class NormConv1d(nn.Module):
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                norm_kwargs: Dict[str, Any] = {}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x

class VibeVoiceTokenizerStreamingCache:
    def __init__(self):
        self.cache = {}

    def get(self, layer_id: str, sample_indices: torch.Tensor) -> Optional[torch.Tensor]:
        states = []
        max_length = 0
        for idx in sample_indices.tolist():
            key = (layer_id, idx)
            if key not in self.cache:
                return None
            state = self.cache[key]
            states.append(state)
            max_length = max(max_length, state.shape[-1])

        if len(states) > 0 and states[0].dim() >= 2:
            padded_states = []
            for state in states:
                if state.shape[-1] < max_length:
                    pad_size = max_length - state.shape[-1]
                    padded_state = F.pad(state, (pad_size, 0), mode='constant', value=0)
                    padded_states.append(padded_state)
                else:
                    padded_states.append(state)
            return torch.stack(padded_states, dim=0)
        else:
            return torch.stack(states, dim=0)

    def set(self, layer_id: str, sample_indices: torch.Tensor, states: torch.Tensor):
        for i, idx in enumerate(sample_indices.tolist()):
            key = (layer_id, idx)
            self.cache[key] = states[i].detach()

class SConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                kernel_size: int, stride: int = 1, dilation: int = 1,
                groups: int = 1, bias: bool = True, causal: bool = False,
                norm: str = 'none', norm_kwargs: Dict[str, Any] = {},
                pad_mode: str = 'reflect'):
        super().__init__()
        self.conv = NormConv1d(in_channels, out_channels, kernel_size, stride,
                            dilation=dilation, groups=groups, bias=bias, causal=causal,
                            norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.pad_mode = pad_mode
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.context_size = (kernel_size - 1) * dilation - (stride - 1)
        self.padding_total = (kernel_size - 1) * dilation - (stride - 1)
        self._layer_id = None

    @property
    def layer_id(self):
        if self._layer_id is None:
            self._layer_id = f"sconv1d_{id(self)}"
        return self._layer_id

    def forward(self, x: torch.Tensor,
                cache: Optional[VibeVoiceTokenizerStreamingCache] = None,
                sample_indices: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                debug: bool = False,
                is_final_chunk: bool = False) -> torch.Tensor:
        B, C, T = x.shape
        if not use_cache or cache is None:
            extra_padding = get_extra_padding_for_conv1d(x, self.kernel_size, self.stride, self.padding_total)
            if self.causal:
                if self.pad_mode == 'constant':
                    x = pad1d(x, (self.padding_total, extra_padding), mode=self.pad_mode, value=0)
                else:
                    x = pad1d(x, (self.padding_total, extra_padding), mode=self.pad_mode)
            else:
                padding_right = self.padding_total // 2
                padding_left = self.padding_total - padding_right
                x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
            return self.conv(x)

        cached_states = cache.get(self.layer_id, sample_indices)
        if cached_states is None:
            if self.context_size > 0:
                cached_states = torch.zeros(B, C, self.context_size, device=x.device, dtype=x.dtype)
            else:
                cached_states = torch.zeros(B, C, 0, device=x.device, dtype=x.dtype)

        if cached_states.shape[2] > 0:
            input_with_context = torch.cat([cached_states, x], dim=2)
        else:
            input_with_context = x

        if is_final_chunk:
            extra_padding = get_extra_padding_for_conv1d(
                input_with_context, self.kernel_size, self.stride, self.padding_total
            )
            if extra_padding > 0:
                input_with_context = pad1d(input_with_context, (0, extra_padding), mode=self.pad_mode)

        output = self.conv(input_with_context)

        if self.context_size > 0:
            total_input_length = input_with_context.shape[2]
            if total_input_length >= self.context_size:
                new_cache_start = total_input_length - self.context_size
                new_cache = input_with_context[:, :, new_cache_start:]
            else:
                new_cache = input_with_context
            cache.set(self.layer_id, sample_indices, new_cache)

        return output

class FFN(nn.Module):
    def __init__(self, embed_dim, ffn_dim, bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear1 = nn.Linear(self.embed_dim, ffn_dim, bias=bias)
        self.gelu = ACT2FN["gelu"]
        self.linear2 = nn.Linear(ffn_dim, self.embed_dim, bias=bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x

class Convlayer(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, dilation=1,
            groups=1, bias=True, pad_mode='zeros', norm='weight_norm', causal=True,
        ):
        super().__init__()
        self.conv = SConv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                           groups=groups, bias=bias, pad_mode=pad_mode, norm=norm, causal=causal)

    def forward(self, x):
        return self.conv(x)

class Block1D(nn.Module):
    def __init__(self, dim, kernel_size=7, drop_path=0., mixer_layer='conv',
                layer_scale_init_value=1e-6, **kwargs):
        super().__init__()
        if kwargs.get('layernorm', 'LN') == 'LN':
            self.norm = ConvLayerNorm(dim, eps=kwargs.get('eps', 1e-6))
            self.ffn_norm = ConvLayerNorm(dim, eps=kwargs.get('eps', 1e-6))
        elif kwargs.get('layernorm', 'RMSNorm') == 'RMSNorm':
            self.norm = ConvRMSNorm(dim, eps=kwargs.get('eps', 1e-6))
            self.ffn_norm = ConvRMSNorm(dim, eps=kwargs.get('eps', 1e-6))

        if mixer_layer == 'conv':
            self.mixer = Convlayer(dim, dim, groups=kwargs.get('groups', 1),
                                kernel_size=kernel_size,
                                pad_mode=kwargs.get('pad_mode', 'reflect'),
                                norm=kwargs.get('norm', 'none'),
                                causal=kwargs.get('causal', True),
                                bias=kwargs.get('bias', True))
        elif mixer_layer == 'depthwise_conv':
            self.mixer = Convlayer(dim, dim, groups=dim,
                                kernel_size=kernel_size,
                                pad_mode=kwargs.get('pad_mode', 'reflect'),
                                norm=kwargs.get('norm', 'none'),
                                causal=kwargs.get('causal', True),
                                bias=kwargs.get('bias', True))
        else:
            raise ValueError(f"Unsupported mixer layer: {mixer_layer}")

        self.ffn = FFN(dim, kwargs.get('ffn_expansion', 4) * dim, bias=kwargs.get('bias', False))
        self.drop_path = nn.Identity() if drop_path <= 0. else nn.modules.DropPath(drop_path)

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.ffn_gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma = None
            self.ffn_gamma = None

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mixer(x)
        if self.gamma is not None:
            x = x * self.gamma.unsqueeze(-1)
        x = residual + self.drop_path(x)

        residual = x
        x = self.ffn_norm(x)
        x = x.permute(0, 2, 1)
        x = self.ffn(x)
        x = x.permute(0, 2, 1)
        if self.ffn_gamma is not None:
            x = x * self.ffn_gamma.unsqueeze(-1)
        x = residual + self.drop_path(x)
        return x

class TokenizerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.channels = config.channels
        self.dimension = config.dimension
        self.n_filters = config.n_filters
        self.ratios = list(reversed(config.ratios))
        self.depths = config.depths
        self.causal = config.causal

        kernel_size = getattr(config, "kernel_size", 7)
        last_kernel_size = getattr(config, "last_kernel_size", 7)
        norm = getattr(config, "norm", "none")
        norm_params = getattr(config, "norm_params", {})
        pad_mode = getattr(config, "pad_mode", "reflect")
        bias = getattr(config, "bias", True)
        layernorm = getattr(config, "layernorm", "LN")
        layernorm_eps = getattr(config, "layernorm_eps", 1e-6)
        layernorm_elementwise_affine = getattr(config, "layernorm_elementwise_affine", True)
        drop_path_rate = getattr(config, "drop_path_rate", 0.0)
        mixer_layer = getattr(config, "mixer_layer", "conv")
        layer_scale_init_value = getattr(config, "layer_scale_init_value", 0)
        disable_last_norm = getattr(config, "disable_last_norm", False)

        if layernorm == 'LN':
            norm_type = ConvLayerNorm
        elif layernorm == 'RMSNorm':
            norm_type = partial(ConvRMSNorm, elementwise_affine=layernorm_elementwise_affine)
        else:
            raise ValueError(f"Unsupported norm type: {layernorm}")

        stem = nn.Sequential(
                SConv1d(self.channels, self.n_filters, kernel_size, norm=norm, norm_kwargs=norm_params, causal=self.causal, pad_mode=pad_mode, bias=bias),
            )

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(stem)
        for i in range(len(self.ratios)):
            in_ch = self.n_filters * (2 ** i)
            out_ch = self.n_filters * (2 ** (i + 1))
            downsample_layer = nn.Sequential(
                SConv1d(in_ch, out_ch, kernel_size=self.ratios[i] * 2, stride=self.ratios[i], causal=self.causal, pad_mode=pad_mode, norm=norm, bias=bias)
            )
            self.downsample_layers.append(downsample_layer)

        layer_type = partial(
            Block1D, mixer_layer=mixer_layer, layernorm=layernorm, eps=layernorm_eps,
            causal=self.causal, pad_mode=pad_mode, norm=norm, bias=bias,
            layer_scale_init_value=layer_scale_init_value,
        )

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0

        for i in range(len(self.depths)):
            in_ch = self.n_filters * (2 ** i)
            stage = nn.Sequential(
                *[layer_type(dim=in_ch, drop_path=dp_rates[cur + j]) for j in range(self.depths[i])]
            )
            self.stages.append(stage)
            cur += self.depths[i]

        if not disable_last_norm:
            self.norm = norm_type(in_ch, eps=layernorm_eps)
        else:
            self.norm = nn.Identity()
        self.head = SConv1d(in_ch, self.dimension, kernel_size=last_kernel_size, causal=self.causal, pad_mode=pad_mode, norm=norm, bias=bias)

    def forward_features(self, x, cache=None, sample_indices=None, use_cache=False, debug=False, is_final_chunk=False):
        for i in range(len(self.depths)):
            for layer in self.downsample_layers[i]:
                if isinstance(layer, SConv1d):
                    x = layer(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug, is_final_chunk=is_final_chunk)
                else:
                    x = layer(x)

            for block in self.stages[i]:
                if hasattr(block, 'mixer') and hasattr(block.mixer, 'conv') and isinstance(block.mixer.conv, SConv1d):
                    residual = x
                    x = block.norm(x)
                    x = block.mixer.conv(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug, is_final_chunk=is_final_chunk)
                    if block.gamma is not None:
                        x = x * block.gamma.unsqueeze(-1)
                    x = residual + x

                    residual = x
                    x = block.ffn_norm(x)
                    x = x.permute(0, 2, 1)
                    x = block.ffn(x)
                    x = x.permute(0, 2, 1)
                    if block.ffn_gamma is not None:
                        x = x * block.ffn_gamma.unsqueeze(-1)
                    x = residual + x
                else:
                    x = block(x)
        return self.norm(x)

    def forward(self, x, cache=None, sample_indices=None, use_cache=False, debug=False, is_final_chunk=False):
        x = self.forward_features(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug, is_final_chunk=is_final_chunk)
        x = self.head(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug, is_final_chunk=is_final_chunk)
        return x

@dataclass
class VibeVoiceTokenizerEncoderOutput:
    mean: torch.Tensor
    std: Optional[Union[float, torch.Tensor]] = None

    def sample(self, dist_type='fix'):
        if dist_type == 'fix':
            x = self.mean + self.std * torch.randn_like(self.mean)
            return x, self.std
        elif dist_type == 'gaussian':
            batch_size = self.mean.size(0)
            value = self.std / 0.8
            std = torch.randn(batch_size, device=self.mean.device, dtype=self.mean.dtype) * value
            while std.dim() < self.mean.dim():
                std = std.unsqueeze(-1)
            x = self.mean + std * torch.randn_like(self.mean)
            return x, std
        else:
            return self.mean, self.std

class VibeVoiceAcousticTokenizerModel(PreTrainedModel):
    config_class = VibeVoiceAcousticTokenizerConfig
    base_model_prefix = "vibevoice_acoustic_tokenizer"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config):
        super().__init__(config)
        self.register_buffer('fix_std', torch.tensor(config.fix_std), persistent=False)
        self.std_dist_type = getattr(config, "std_dist_type", "fix")

        if isinstance(config.encoder_depths, str):
            encoder_depths = [int(d) for d in config.encoder_depths.split('-')]
        else:
            encoder_depths = config.encoder_depths

        encoder_config = copy.deepcopy(config)
        encoder_config.dimension = config.vae_dim
        encoder_config.n_filters = config.encoder_n_filters
        encoder_config.ratios = config.encoder_ratios
        encoder_config.depths = encoder_depths
        encoder_config.norm = config.conv_norm
        encoder_config.pad_mode = config.pad_mode
        encoder_config.bias = config.conv_bias
        encoder_config.layernorm_eps = config.layernorm_eps
        encoder_config.layernorm_elementwise_affine = config.layernorm_elementwise_affine
        encoder_config.mixer_layer = config.mixer_layer
        encoder_config.layer_scale_init_value = config.layer_scale_init_value
        encoder_config.disable_last_norm = config.disable_last_norm

        self.encoder = TokenizerEncoder(encoder_config)

    @torch.no_grad()
    def encode(self, audio, cache=None, sample_indices=None, use_cache=False, debug=False, is_final_chunk=False):
        latents = self.encoder(audio, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug, is_final_chunk=is_final_chunk)
        return VibeVoiceTokenizerEncoderOutput(mean=latents.permute(0, 2, 1), std=self.fix_std)

class VibeVoiceSemanticTokenizerModel(PreTrainedModel):
    config_class = VibeVoiceSemanticTokenizerConfig
    base_model_prefix = "vibevoice_semantic_tokenizer"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config):
        super().__init__(config)

        if isinstance(config.encoder_depths, str):
            encoder_depths = [int(d) for d in config.encoder_depths.split('-')]
        else:
            encoder_depths = config.encoder_depths

        encoder_config = copy.deepcopy(config)
        encoder_config.dimension = config.vae_dim
        encoder_config.n_filters = config.encoder_n_filters
        encoder_config.ratios = config.encoder_ratios
        encoder_config.depths = encoder_depths
        encoder_config.norm = config.conv_norm
        encoder_config.pad_mode = config.pad_mode
        encoder_config.bias = config.conv_bias
        encoder_config.layernorm_eps = config.layernorm_eps
        encoder_config.layernorm_elementwise_affine = config.layernorm_elementwise_affine
        encoder_config.mixer_layer = config.mixer_layer
        encoder_config.layer_scale_init_value = config.layer_scale_init_value
        encoder_config.disable_last_norm = config.disable_last_norm

        self.encoder = TokenizerEncoder(encoder_config)

    @torch.no_grad()
    def encode(self, audio, cache=None, sample_indices=None, use_cache=False, debug=False, is_final_chunk=False):
        latents = self.encoder(audio, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug, is_final_chunk=is_final_chunk)
        return VibeVoiceTokenizerEncoderOutput(mean=latents.permute(0, 2, 1))

AutoModel.register(VibeVoiceAcousticTokenizerConfig, VibeVoiceAcousticTokenizerModel)
AutoModel.register(VibeVoiceSemanticTokenizerConfig, VibeVoiceSemanticTokenizerModel)


# ============================================================================
# Main Model Architecture
# ============================================================================

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

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
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.acoustic_vae_dim = getattr(config, "acoustic_vae_dim", 64)
        self.semantic_vae_dim = getattr(config, "semantic_vae_dim", 128)
        self.hidden_size = getattr(config, "hidden_size", 3584)

        ac_cfg = getattr(config, "acoustic_tokenizer_config", None)
        sc_cfg = getattr(config, "semantic_tokenizer_config", None)

        if ac_cfg is None or sc_cfg is None:
            raise ValueError("Missing acoustic/semantic tokenizer config")

        if isinstance(ac_cfg, dict):
            acoustic_config = VibeVoiceAcousticTokenizerConfig(**ac_cfg)
        else:
            acoustic_config = ac_cfg

        if isinstance(sc_cfg, dict):
            semantic_config = VibeVoiceSemanticTokenizerConfig(**sc_cfg)
        else:
            semantic_config = sc_cfg

        self.acoustic_tokenizer = VibeVoiceAcousticTokenizerModel(acoustic_config)
        self.semantic_tokenizer = VibeVoiceSemanticTokenizerModel(semantic_config)

        root_torch_dtype = getattr(config, "torch_dtype", None)
        if root_torch_dtype is not None:
            if isinstance(root_torch_dtype, str):
                self._audio_encoder_dtype = getattr(torch, root_torch_dtype)
            else:
                self._audio_encoder_dtype = root_torch_dtype
        else:
            self._audio_encoder_dtype = torch.float32

        self.acoustic_connector = SpeechConnector(self.acoustic_vae_dim, self.hidden_size)
        self.semantic_connector = SpeechConnector(self.semantic_vae_dim, self.hidden_size)

        self.compress_ratio = getattr(config, "speech_tok_compress_ratio", 3200)
        self.sample_rate = getattr(config, "target_sample_rate", 24000)
        self.enable_streaming = getattr(config, "enable_streaming", True)
        self.streaming_segment_duration = getattr(config, "streaming_segment_duration", 60.0)

        use_mean_env = os.getenv("VIBEVOICE_USE_MEAN", "").strip().lower()
        self.use_sample = use_mean_env not in ("1", "true", "yes")
        self._lm_dtype: torch.dtype = torch.bfloat16

    def _ensure_audio_encoder_dtype(self):
        target_dtype = self._audio_encoder_dtype
        for module in [self.acoustic_tokenizer, self.semantic_tokenizer, self.acoustic_connector, self.semantic_connector]:
            try:
                mod_dtype = next(module.parameters()).dtype
                if mod_dtype != target_dtype:
                    module.to(dtype=target_dtype)
            except StopIteration:
                pass

    def forward(
        self,
        audio: torch.Tensor,
        *,
        use_streaming: bool = True,
        segment_duration_s: Optional[float] = None,
        use_sample: Optional[bool] = None,
    ) -> torch.Tensor:
        self._ensure_audio_encoder_dtype()
        audio = audio.to(dtype=self._audio_encoder_dtype)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        segment_duration = segment_duration_s or self.streaming_segment_duration
        sample_rate = self.sample_rate
        total_samples = audio.shape[-1]
        segment_samples = int(segment_duration * sample_rate)
        use_streaming = use_streaming and self.enable_streaming and total_samples > segment_samples
        if use_sample is None:
            use_sample = self.use_sample

        with torch.no_grad():
            if not use_streaming:
                acoustic_input = audio.unsqueeze(1)
                acoustic_out = self.acoustic_tokenizer.encode(acoustic_input)
                if use_sample:
                    acoustic_tokens = acoustic_out.sample(dist_type=self.acoustic_tokenizer.std_dist_type)[0]
                else:
                    acoustic_tokens = acoustic_out.mean
                acoustic_embeds = self.acoustic_connector(acoustic_tokens)

                semantic_out = self.semantic_tokenizer.encode(acoustic_input)
                semantic_tokens = semantic_out.mean
                semantic_embeds = self.semantic_connector(semantic_tokens)
            else:
                # Streaming path
                acoustic_cache = VibeVoiceTokenizerStreamingCache()
                semantic_cache = VibeVoiceTokenizerStreamingCache()
                acoustic_mean_segments = []
                semantic_mean_segments = []
                batch_size = audio.shape[0]
                sample_indices = torch.arange(batch_size, device=audio.device)

                def _iter_segments(total_length: int, segment_length: int):
                    for start in range(0, total_length, segment_length):
                        end = min(start + segment_length, total_length)
                        if end > start:
                            yield start, end

                segments = list(_iter_segments(total_samples, segment_samples))
                num_segments = len(segments)
                for seg_idx, (start, end) in enumerate(segments):
                    chunk = audio[:, start:end].contiguous()
                    if chunk.numel() == 0:
                        continue
                    is_final = (seg_idx == num_segments - 1)
                    acoustic_enc_out = self.acoustic_tokenizer.encode(
                        chunk.unsqueeze(1), cache=acoustic_cache, sample_indices=sample_indices,
                        use_cache=True, is_final_chunk=is_final,
                    )
                    acoustic_mean_segments.append(acoustic_enc_out.mean)
                    semantic_enc_out = self.semantic_tokenizer.encode(
                        chunk.unsqueeze(1), cache=semantic_cache, sample_indices=sample_indices,
                        use_cache=True, is_final_chunk=is_final,
                    )
                    semantic_mean_segments.append(semantic_enc_out.mean)

                if not acoustic_mean_segments:
                    acoustic_mean_full = torch.zeros((batch_size, 0, self.acoustic_vae_dim), device=audio.device, dtype=self._audio_encoder_dtype)
                else:
                    acoustic_mean_full = torch.cat(acoustic_mean_segments, dim=1).contiguous()

                acoustic_enc_full = VibeVoiceTokenizerEncoderOutput(mean=acoustic_mean_full, std=self.acoustic_tokenizer.fix_std)
                if use_sample:
                    acoustic_tokens = acoustic_enc_full.sample(dist_type=self.acoustic_tokenizer.std_dist_type)[0]
                else:
                    acoustic_tokens = acoustic_enc_full.mean
                acoustic_embeds = self.acoustic_connector(acoustic_tokens)

                if not semantic_mean_segments:
                    semantic_tokens = torch.zeros((batch_size, 0, self.semantic_vae_dim), device=audio.device, dtype=self._audio_encoder_dtype)
                else:
                    semantic_tokens = torch.cat(semantic_mean_segments, dim=1).contiguous()
                semantic_embeds = self.semantic_connector(semantic_tokens)

        combined_embeds = acoustic_embeds + semantic_embeds
        combined_embeds = combined_embeds.to(dtype=self._lm_dtype)
        return combined_embeds


# ============================================================================
# Audio Utils
# ============================================================================

_FFMPEG_MAX_CONCURRENCY = int(os.getenv("VIBEVOICE_FFMPEG_MAX_CONCURRENCY", "0"))
_FFMPEG_SEM = threading.Semaphore(_FFMPEG_MAX_CONCURRENCY) if _FFMPEG_MAX_CONCURRENCY > 0 else None

def _run_ffmpeg(cmd: list, *, stdin_bytes: bytes = None):
    if _FFMPEG_SEM is None:
        return run(cmd, capture_output=True, check=True, input=stdin_bytes)
    with _FFMPEG_SEM:
        return run(cmd, capture_output=True, check=True, input=stdin_bytes)

# NOTE: This function is preserved for reference or direct usage if needed,
# but VibeVoiceMultiModalProcessor currently uses the standard DataParser
# followed by explicit AudioNormalizer application.
def load_audio(file: Union[str, bytes], target_sr: int = 24000) -> np.ndarray:
    """Load audio from file path or bytes using FFmpeg, resample to 24kHz and normalize."""
    if isinstance(file, bytes):
        cmd = [
            "ffmpeg", "-loglevel", "error", "-threads", "0", "-i", "pipe:0",
            "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(target_sr), "-"
        ]
        out = _run_ffmpeg(cmd, stdin_bytes=file).stdout
    else:
        cmd = [
            "ffmpeg", "-loglevel", "error", "-nostdin", "-threads", "0", "-i", file,
            "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(target_sr), "-"
        ]
        out = _run_ffmpeg(cmd).stdout

    audio_data = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    # Normalize
    normalizer = AudioNormalizer()
    audio_data = normalizer(audio_data)
    return audio_data

class AudioNormalizer:
    def __init__(self, target_dB_FS: float = -25, eps: float = 1e-6):
        self.target_dB_FS = target_dB_FS
        self.eps = eps

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(audio**2))
        scalar = 10 ** (self.target_dB_FS / 20) / (rms + self.eps)
        audio = audio * scalar
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / (max_val + self.eps)
        return audio


# ============================================================================
# Processor
# ============================================================================

class VibeVoiceProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_feature_extractor(self, **kwargs) -> WhisperFeatureExtractor:
        # Dummy feature extractor for profiling consistency
        return WhisperFeatureExtractor(
            feature_size=128, sampling_rate=24000, hop_length=240,
            chunk_length=30, n_fft=400, padding_value=0.0
        )

    def get_audio_token_info(self) -> dict:
        tokenizer = self.get_tokenizer()
        vocab = tokenizer.get_vocab()
        return {
            "audio_token": "<|AUDIO|>",
            "audio_bos_token": "<|audio_bos|>",
            "audio_eos_token": "<|audio_eos|>",
            "audio_token_id": vocab.get("<|AUDIO|>"),
            "audio_bos_id": vocab.get("<|audio_bos|>"),
            "audio_eos_id": vocab.get("<|audio_eos|>"),
        }

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: Mapping[str, int]) -> Mapping[str, int]:
        hf_config = self.get_hf_config()
        compress_ratio = int(getattr(hf_config, "speech_tok_compress_ratio", 3200))
        sample_rate = int(getattr(hf_config, "target_sample_rate", 24000))
        max_audio_samples = 61 * 60 * sample_rate
        max_audio_tokens = int(np.ceil(max_audio_samples / compress_ratio)) + 3
        max_audio_tokens = min(max_audio_tokens, seq_len)
        return {"audio": max_audio_tokens}


class VibeVoiceDummyInputsBuilder(BaseDummyInputsBuilder[VibeVoiceProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return "<|AUDIO|>" * num_audios if num_audios > 0 else ""

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        num_audios = mm_counts.get("audio", 0)
        hf_config = self.info.get_hf_config()
        compress_ratio = int(getattr(hf_config, "speech_tok_compress_ratio", 3200))
        max_tokens_from_audio = self.info.get_mm_max_tokens_per_item(seq_len, mm_counts)["audio"]
        max_audio_len = (max_tokens_from_audio - 3) * compress_ratio
        audio_overrides = mm_options.get("audio") if mm_options else None
        return {
            "audio": self._get_dummy_audios(length=max_audio_len, num_audios=num_audios, overrides=audio_overrides)
        }


def _vibevoice_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    config = {
        "raw_audio": MultiModalFieldConfig.batched("audio"),
        "raw_audio_lengths": MultiModalFieldConfig.batched("audio"),
        "salt": MultiModalFieldConfig.batched("audio"),
    }
    return config


class VibeVoiceMultiModalProcessor(BaseMultiModalProcessor[VibeVoiceProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=24000)

    def _call_hf_processor(
        self, prompt: str, mm_data: Mapping[str, object], mm_kwargs: Mapping[str, object], tok_kwargs: Mapping[str, object]
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", None)
        if audios is not None and "audio" not in mm_data:
            mm_data["audio"] = audios

        if not mm_data.get("audio"):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        raw_audio_list = mm_data.get("audio")
        if isinstance(raw_audio_list, np.ndarray):
            raw_audio_list = [raw_audio_list]
        elif not isinstance(raw_audio_list, list):
            raw_audio_list = list(raw_audio_list)

        # Apply normalization
        # VibeVoice requires specific audio normalization (-25 dBFS)
        normalizer = AudioNormalizer()
        normalized_audio_list = []
        for audio in raw_audio_list:
            if isinstance(audio, np.ndarray):
                # Ensure float32
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                normalized_audio_list.append(normalizer(audio))
            else:
                normalized_audio_list.append(audio)
        raw_audio_list = normalized_audio_list

        tokenizer = self.info.get_tokenizer()
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
        result = BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        max_len = max(len(a) for a in raw_audio_list)
        raw_audio_tensors = []
        audio_lengths = []
        for audio in raw_audio_list:
            audio_len = len(audio)
            audio_lengths.append(audio_len)
            if audio_len < max_len:
                audio = np.pad(audio, (0, max_len - audio_len), mode='constant')
            raw_audio_tensors.append(torch.from_numpy(audio).float())

        stacked_audio = torch.stack(raw_audio_tensors, dim=0)
        result["raw_audio"] = stacked_audio
        result["raw_audio_lengths"] = torch.tensor(audio_lengths, dtype=torch.long)

        # Add random salt to bypass caching if needed
        import uuid
        salt_val = hash(str(uuid.uuid4())) % 100000
        result["salt"] = torch.tensor([salt_val], dtype=torch.long).expand(len(raw_audio_list))

        return result

    def _get_mm_fields_config(self, hf_inputs: BatchFeature, hf_processor_mm_kwargs: Mapping[str, object]) -> Mapping[str, MultiModalFieldConfig]:
        return _vibevoice_field_config(hf_inputs)

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs: Mapping[str, object], out_mm_kwargs: MultiModalKwargsItems) -> Sequence[PromptUpdate]:
        token_info = self.info.get_audio_token_info()
        audio_token = token_info["audio_token"]
        audio_token_id = token_info["audio_token_id"]

        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        def _tok_id(name: str) -> int | None:
            return vocab.get(name)

        speech_start_id = _tok_id("<|object_ref_start|>") or getattr(tokenizer, "speech_start_id", None) or _tok_id("<|speech_start|>")
        speech_end_id = _tok_id("<|object_ref_end|>") or getattr(tokenizer, "speech_end_id", None) or _tok_id("<|speech_end|>")
        speech_pad_id = _tok_id("<|box_start|>") or getattr(tokenizer, "speech_pad_id", None) or _tok_id("<|speech_pad|>")

        if audio_token_id is None:
            return []

        out_mm_data = out_mm_kwargs.get_data()
        raw_audio_lengths = out_mm_data.get("raw_audio_lengths", [])

        hf_config = self.info.get_hf_config()
        compress_ratio = int(getattr(hf_config, "speech_tok_compress_ratio", 3200))

        def get_replacement(item_idx: int):
            if raw_audio_lengths is not None and item_idx < len(raw_audio_lengths):
                audio_len = int(raw_audio_lengths[item_idx].item())
                num_features = max(1, int(np.ceil(audio_len / compress_ratio)))
            else:
                num_features = int(np.ceil(30 * 24000 / compress_ratio))

            newline_id = 198
            if speech_start_id and speech_pad_id and speech_end_id:
                replacement_ids = [speech_start_id] + [speech_pad_id] * num_features + [speech_end_id, newline_id]
            else:
                replacement_ids = [audio_token_id] * num_features

            return PromptUpdateDetails.select_token_id(replacement_ids, embed_token_id=int(speech_pad_id if speech_pad_id else audio_token_id))

        return [PromptReplacement(modality="audio", target=audio_token, replacement=get_replacement)]


@MULTIMODAL_REGISTRY.register_processor(
    VibeVoiceMultiModalProcessor,
    info=VibeVoiceProcessingInfo,
    dummy_inputs=VibeVoiceDummyInputsBuilder,
)
class VibeVoiceForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|AUDIO|>"
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        self.audio_encoder = VibeVoiceAudioEncoder(config)

        decoder_config = getattr(config, "decoder_config", config)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=decoder_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

        lm_dtype = vllm_config.model_config.dtype
        if lm_dtype is not None:
            self.audio_encoder._lm_dtype = lm_dtype

        try:
            self.audio_encoder._ensure_audio_encoder_dtype()
        except Exception:
            pass

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        raw_audio = kwargs.get("raw_audio")
        raw_audio_lengths = kwargs.get("raw_audio_lengths")

        if raw_audio is None:
            return []

        # Flatten raw_audio_lengths if needed
        def flatten_lengths(lengths):
            if lengths is None: return []
            result = []
            if isinstance(lengths, torch.Tensor):
                lengths = lengths.tolist()
            if isinstance(lengths, (list, tuple)):
                for item in lengths:
                    if isinstance(item, (list, tuple)):
                        result.extend(flatten_lengths(item))
                    elif isinstance(item, torch.Tensor):
                        result.append(item.item() if item.dim() == 0 else item.tolist())
                    else:
                        result.append(item)
            else:
                result.append(lengths)
            return result

        raw_audio_lengths = flatten_lengths(raw_audio_lengths)

        use_streaming_flag = bool(kwargs.get("use_streaming", getattr(self.audio_encoder, "enable_streaming", True)))
        streaming_segment_duration = kwargs.get("streaming_segment_duration", getattr(self.audio_encoder, "streaming_segment_duration", 60.0))

        embeddings = []
        try:
            device = next(self.audio_encoder.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(raw_audio, torch.Tensor):
            if raw_audio.dim() == 3:
                num_audios = raw_audio.shape[0]
                audio_list = [raw_audio[i].squeeze(0) for i in range(num_audios)]
            elif raw_audio.dim() == 2:
                num_audios = raw_audio.shape[0]
                audio_list = [raw_audio[i] for i in range(num_audios)]
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

            if raw_audio_lengths and i < len(raw_audio_lengths):
                actual_len = int(raw_audio_lengths[i])
                if actual_len > 0 and actual_len <= audio_tensor.shape[-1]:
                    audio_tensor = audio_tensor[..., :actual_len]

            if audio_tensor.numel() < 160:
                # Add a dummy embedding or skip? If we skip, indices might mismatch.
                # vLLM expects 1-to-1 mapping if placeholders are present.
                # If audio is too short, we should probably raise error or padding.
                # Raising error is safer.
                raise ValueError(f"Audio at index {i} is too short ({audio_tensor.numel()} samples).")

            audio_embeds = self.audio_encoder(
                audio_tensor,
                use_streaming=use_streaming_flag,
                segment_duration_s=streaming_segment_duration,
            )
            final_embed = audio_embeds.squeeze(0)
            embeddings.append(final_embed)

        return tuple(embeddings)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
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

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.language_model.model.embed_tokens(input_ids)

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
            **kwargs
        )
        return hidden_states
