# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import List, Optional
from transformers import PretrainedConfig, Qwen2Config

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
        encoder_ratios: Optional[List[int]] = [8,5,5,4,2,2],
        encoder_depths: str = "3-3-3-3-3-3-8",
        decoder_n_filters: int = 32,
        decoder_ratios: Optional[List[int]] = None,
        decoder_depths: Optional[str] = None,
        **kwargs
    ):
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
        encoder_ratios: Optional[List[int]] = [8,5,5,4,2,2],
        encoder_depths: str = "3-3-3-3-3-3-8",
        **kwargs
    ):
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

    def __init__(
        self,
        acoustic_tokenizer_config=None,
        semantic_tokenizer_config=None,
        decoder_config=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if isinstance(acoustic_tokenizer_config, dict):
            self.acoustic_tokenizer_config = VibeVoiceAcousticTokenizerConfig(**acoustic_tokenizer_config)
        else:
            self.acoustic_tokenizer_config = acoustic_tokenizer_config

        if isinstance(semantic_tokenizer_config, dict):
            self.semantic_tokenizer_config = VibeVoiceSemanticTokenizerConfig(**semantic_tokenizer_config)
        else:
            self.semantic_tokenizer_config = semantic_tokenizer_config

        if isinstance(decoder_config, dict):
            self.decoder_config = Qwen2Config(**decoder_config)
        else:
            self.decoder_config = decoder_config

        self.acoustic_vae_dim = getattr(self.acoustic_tokenizer_config, 'vae_dim', 64)
        self.semantic_vae_dim = getattr(self.semantic_tokenizer_config, 'vae_dim', 128)

    def get_text_config(self, decoder=False):
        return self.decoder_config

    @property
    def vocab_size(self):
        return self.decoder_config.vocab_size

    @property
    def hidden_size(self):
        return self.decoder_config.hidden_size
