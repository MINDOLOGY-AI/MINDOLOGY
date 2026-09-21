'''
OlMo2 decoder-only causal LM.

Built from the generic primitives in synapse/algorithms:
- basic.RMSnorm.RMSNorm
- encoding.RoPE.RoPE
- encoding.rope_utils.apply_rotary_pos_emb
- attention.GQA.GQA
- basic.gatedMLP.GatedMLP
- encoding.mask.create_olmo_causal_mask

Architecture toggles are driven by the config object:
- qk_norm / use_flash_attention
- mlp_bias / separate_gate_up_proj

OlMo2 itself is always post-norm: norms are applied to sub-layer outputs
inside the residual branch, no config flag.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..basic.RMSnorm import RMSNorm
from ..basic.gatedMLP import GatedMLP
from ..encoding.RoPE import RoPE
from ..encoding.mask import create_olmo_causal_mask
from ..attention.GQA import GQA


class Olmo2DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = GQA(
            embed_dim=config.hidden_size,
            num_q_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            qk_norm=config.qk_norm,
            rms_norm_eps=config.rms_norm_eps,
            use_flash=config.use_flash_attention,
        )

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias,
            separate_gate_up_proj=config.separate_gate_up_proj,
        )

        # OlMo2 is post-norm: the norm sits inside the residual branch,
        # applied to the sub-layer output before the residual add
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, cast_to_float32=config.cast_norm_to_float32
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, cast_to_float32=config.cast_norm_to_float32
        )

    def forward(self, hidden_states, attention_mask, position_embeddings, past_key_value=None):
        # attention branch
        residual = hidden_states
        attn_output, present_kv = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
        )
        attn_output = self.post_attention_layernorm(attn_output)
        hidden_states = residual + attn_output

        # mlp branch
        residual = hidden_states
        mlp_output = self.mlp(hidden_states)
        mlp_output = self.post_feedforward_layernorm(mlp_output)
        hidden_states = residual + mlp_output

        return hidden_states, present_kv


class Olmo2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.padding_idx = config.pad_token_id

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Olmo2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, cast_to_float32=config.cast_norm_to_float32
        )
        self.rotary_emb = RoPE(
            config.head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(2) if past_key_values[0] is not None else 0

        if position_ids is None:
            position_ids = torch.arange(
                past_length, past_length + seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        causal_mask = create_olmo_causal_mask(
            inputs_embeds,
            attention_mask=attention_mask,
            past_key_values_length=past_length,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        presents = []

        for layer, past_kv in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)
        return hidden_states, presents


class Olmo2ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Olmo2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
    ):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )

        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # shift: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss, past_key_values

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        """Helper used by local generate loops."""
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        position_ids = kwargs.get("position_ids")
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
