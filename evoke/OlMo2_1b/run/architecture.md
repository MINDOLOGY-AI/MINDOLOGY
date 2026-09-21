# OlMo2-1B-Instruct Architecture

This document describes the local implementation in `MINDOLOGY/synapse/algorithms/transformer/olmo2.py` for the checkpoint `allenai/OLMo-2-0425-1B-Instruct`.

---

## 1. High-level stack

```text
input_ids
    │
    ▼
┌─────────────────┐
│  embed_tokens   │  nn.Embedding(vocab=100352, dim=2048)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   decoder_0     │
│   decoder_1     │
│      ...        │  16 identical layers
│  decoder_15     │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   final RMSNorm │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│     lm_head     │  nn.Linear(2048 → 100352, bias=False)
└─────────────────┘
    │
    ▼
  logits
```

Word embeddings and the LM head are **not tied** for this checkpoint (`tie_word_embeddings=False`).

---

## 2. 1B configuration snapshot

| hyperparameter | value |
|---|---|
| vocab_size | 100352 |
| hidden_size | 2048 |
| intermediate_size | 8192 |
| num_hidden_layers | 16 |
| num_attention_heads | 16 |
| num_key_value_heads | 16 (standard MHA for this checkpoint) |
| head_dim | 128 |
| max_position_embeddings | 4096 |
| RoPE theta | 500000 |
| norm epsilon | 1e-6 |
| activation | SiLU |
| attention dropout | 0.0 |
| attention bias | False |
| MLP bias | False |

---

## 3. Decoder layer (one block)

OlMo2 uses a **post-normalization** residual branch. Each layer is:

```text
input
  │
  ├──► self_attn ──► post_attention_layernorm ──┐
  │                                              ├──► add ──► output_of_attention_branch
  └─────────────────────────────────────────────┘
  │
  ├──► mlp ──► post_feedforward_layernorm ──────┐
  │                                              ├──► add ──► layer_output
  └─────────────────────────────────────────────┘
```

Component by component:

### 3.1 Self-attention

Type: **GQA** (Grouped Query Attention) — for 1B it happens to be full MHA because `num_kv_heads == num_q_heads`, but the code supports `num_kv_heads < num_q_heads`.

Inside the attention:

```text
hidden_states
    │
    ├──► q_proj ──► q_norm? ──► reshape + RoPE ──┐
    ├──► k_proj ──► k_norm? ──► reshape + RoPE ──┼──► scaled dot-product attention
    └──► v_proj ──────────────► reshape ──────────┘
    │
    ▼
  o_proj
```

- `qk_norm=True` for this checkpoint: RMSNorm is applied to the projected Q and K *before* RoPE.
- RoPE is applied per-layer using cos/sin computed once per forward from `position_ids`.
- The attention core defaults to a manual eager implementation; set `use_flash_attention=True` to route through `F.scaled_dot_product_attention`.

### 3.2 MLP

Type: **SwiGLU** (gated MLP).

```text
hidden_states
    │
    ├──► gate_proj ──► SiLU ──┐
    │                          ├──► elementwise_mul ──► down_proj ──► output
    └──► up_proj ──────────────┘
```

- `gate_proj` and `up_proj` are separate linear layers by default (`separate_gate_up_proj=True`).
- All projections have `bias=False` for this checkpoint.

### 3.3 Normalization

- Norm type: **RMSNorm**.
- Computation is done in float32 for stability, then cast back to the model dtype.
- In this checkpoint the norm is applied **after** the sub-layer inside the residual branch (`post_norm=True`, `pre_norm=False`).

---

## 4. Position encoding

Type: **RoPE (Rotary Position Embedding)**.

- Not a learned embedding; frequencies are fixed.
- `rope_theta = 500000` for this checkpoint.
- A single `RoPE` module lives at the model level and produces `(cos, sin)` from `position_ids`.
- The same cos/sin pair is passed into every layer's attention and applied to Q and K.
- Supports KV-cache decoding by offsetting `position_ids` with the cached sequence length.

---

## 5. KV-cache

During generation the model returns the full keys and values for each layer. The next forward pass reuses them so attention is only computed over the new token.

```text
generation step 0:
    prompt_tokens ──► model ──► logits + past_kv[0..15]

generation step 1..N:
    new_token ──► model(past_key_values=past_kv) ──► logits + updated past_kv
```

---

## 6. Loss

`Olmo2ForCausalLM.forward` computes cross-entropy when `labels` are provided:

```text
logits[:, :-1, :]  vs  labels[:, 1:]
```

The loss is computed inside the causal-LM module, matching the behavior of your existing `ModernCausalLM`.

---

## 7. File map

| concept | implementation file |
|---|---|
| Config | `MINDOLOGY/evoke/OlMo2_1b/config.py` |
| Model / Causal LM | `MINDOLOGY/synapse/algorithms/transformer/olmo2.py` |
| RMSNorm | `MINDOLOGY/synapse/algorithms/basic/RMSnorm.py` |
| GQA attention | `MINDOLOGY/synapse/algorithms/attention/GQA.py` |
| RoPE + helpers | `MINDOLOGY/synapse/algorithms/encoding/RoPE.py`, `rope_utils.py` |
| Causal mask | `MINDOLOGY/synapse/algorithms/encoding/mask.py` |
| SwiGLU MLP | `MINDOLOGY/synapse/algorithms/basic/gatedMLP.py` |
| Weight loader | `MINDOLOGY/evoke/OlMo2_1b/loader.py` |
| Chat loop | `MINDOLOGY/evoke/OlMo2_1b/chat_local.py` |
