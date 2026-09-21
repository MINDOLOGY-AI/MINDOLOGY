"""
Config for allenai/OLMo-2-0425-1B-Instruct.

This is the concrete 1B checkpoint config, pulled from the cached HF weights.
Keep model-specific hyperparameters here so the generic modules in
synapse/algorithms stay reusable.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Olmo2Config:
    """Configuration for the OLMo2 1B Instruct model."""

    # tokenization
    vocab_size: int = 100352
    pad_token_id: int = 100277
    bos_token_id: Optional[int] = None
    eos_token_id: int = 100257

    # model size
    hidden_size: int = 2048
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    intermediate_size: int = 8192

    # activation / norm
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    cast_norm_to_float32: bool = True

    # RoPE
    max_position_embeddings: int = 4096
    rope_theta: float = 500000.0
    rope_type: str = "default"

    # attention
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # init
    initializer_range: float = 0.02

    # inference
    use_cache: bool = True

    # architecture toggles: lets us reuse the same primitives across models
    # attention extras
    qk_norm: bool = True
    use_flash_attention: bool = False

    # mlp extras
    mlp_bias: bool = False
    separate_gate_up_proj: bool = True

    # lm head
    tie_word_embeddings: bool = False

    # checkpoint dtype stored on disk
    torch_dtype: str = "bfloat16"

    # convenience derived props
    head_dim: int = field(init=False)
    num_key_value_groups: int = field(init=False)

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads


# shortcut for the 1B instruct checkpoint
OLMO2_1B_INSTRUCT_CONFIG = Olmo2Config()
