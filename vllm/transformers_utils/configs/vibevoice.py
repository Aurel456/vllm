# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VibeVoice ASR model configuration for vLLM.

This module provides the VibeVoice config class that wraps around
the vibevoice package's config, making it compatible with vLLM's
model loading infrastructure.

Since `vibevoice` model_type is not recognized by the transformers
library, we register this config so that AutoConfig.from_pretrained()
can load VibeVoice checkpoints.
"""

from transformers import PretrainedConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


class VibeVoiceAcousticTokenizerConfig(PretrainedConfig):
    model_type = "vibevoice_acoustic_tokenizer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)


class VibeVoiceSemanticTokenizerConfig(PretrainedConfig):
    model_type = "vibevoice_semantic_tokenizer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)


class VibeVoiceConfig(PretrainedConfig):
    """Configuration class for VibeVoice ASR models.

    This is a composition config that contains sub-configs for:
    - acoustic_tokenizer_config: Acoustic VAE tokenizer parameters
    - semantic_tokenizer_config: Semantic VAE tokenizer parameters
    - decoder_config: Qwen2 language model parameters
    """

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
        **kwargs,
    ):
        kwargs["_attn_implementation_autoset"] = False

        if acoustic_tokenizer_config is None:
            self.acoustic_tokenizer_config = VibeVoiceAcousticTokenizerConfig()
        elif isinstance(acoustic_tokenizer_config, dict):
            acoustic_tokenizer_config["model_type"] = (
                "vibevoice_acoustic_tokenizer"
            )
            self.acoustic_tokenizer_config = VibeVoiceAcousticTokenizerConfig(
                **acoustic_tokenizer_config
            )
        elif isinstance(
            acoustic_tokenizer_config, VibeVoiceAcousticTokenizerConfig
        ):
            self.acoustic_tokenizer_config = acoustic_tokenizer_config

        if semantic_tokenizer_config is None:
            self.semantic_tokenizer_config = VibeVoiceSemanticTokenizerConfig()
        elif isinstance(semantic_tokenizer_config, dict):
            semantic_tokenizer_config["model_type"] = (
                "vibevoice_semantic_tokenizer"
            )
            self.semantic_tokenizer_config = VibeVoiceSemanticTokenizerConfig(
                **semantic_tokenizer_config
            )
        elif isinstance(
            semantic_tokenizer_config, VibeVoiceSemanticTokenizerConfig
        ):
            self.semantic_tokenizer_config = semantic_tokenizer_config

        if decoder_config is None:
            self.decoder_config = Qwen2Config()
        elif isinstance(decoder_config, dict):
            if decoder_config.get("model_type", "") == "qwen2":
                self.decoder_config = Qwen2Config(**decoder_config)
            else:
                self.decoder_config = Qwen2Config(**decoder_config)
        elif isinstance(decoder_config, Qwen2Config):
            self.decoder_config = decoder_config

        # Expose key properties from sub-configs
        self.acoustic_vae_dim = getattr(
            self.acoustic_tokenizer_config, "vae_dim", 64
        )
        self.semantic_vae_dim = getattr(
            self.semantic_tokenizer_config, "vae_dim", 128
        )

        # Store all remaining kwargs as attributes  
        for key, value in kwargs.items():
            if key != "_attn_implementation_autoset" and not hasattr(
                self, key
            ):
                setattr(self, key, value)

        super().__init__(**kwargs)

    def get_text_config(self, decoder=False):
        """Return the text (decoder) config for generation."""
        return self.decoder_config

    @property
    def vocab_size(self):
        return self.decoder_config.vocab_size

    @property
    def num_attention_heads(self):
        return self.decoder_config.num_attention_heads

    @property
    def num_key_value_heads(self):
        return self.decoder_config.num_key_value_heads

    @property
    def hidden_size(self):
        return self.decoder_config.hidden_size

    @property
    def num_hidden_layers(self):
        return self.decoder_config.num_hidden_layers

    @property
    def head_dim(self):
        return getattr(
            self.decoder_config,
            "head_dim",
            self.hidden_size // self.num_attention_heads,
        )

    def to_dict(self):
        output = super().to_dict()
        # Handle torch.dtype serialization
        if "torch_dtype" in output and output["torch_dtype"] is not None:
            import torch

            dtype = output["torch_dtype"]
            if isinstance(dtype, torch.dtype):
                output["torch_dtype"] = str(dtype).replace("torch.", "")
        return output
