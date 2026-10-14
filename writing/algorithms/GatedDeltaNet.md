# Gated Delta Net

code: `synapse/algorithms/attention/GatedDeltaNet.py`
papers: DeltaNet (Yang et al. 2024, "Parallelizing Linear Transformers with the Delta Rule"), Gated DeltaNet (Yang, Kautz, Hatamizadeh, ICLR 2025)

## where it sits in Qwen3.5-4B

- 32 layers = 8 × [GDN, GDN, GDN, full attention]
- so 24 of 32 layers mix tokens with Gated DeltaNet, only 8 use normal softmax attention
- it replaces attention inside the block; the residual stream, the norms and the MLP are unchanged

## 1. the problem with softmax attention

$$o_t = \sum_{s \le t} \mathrm{softmax}_s(q_t \cdot k_s)\, v_s$$
- every token looks at every past token: O(T²) compute, and the KV cache grows with T
- at 100k tokens the cache is huge, and each new token reads all of it

## 2. linear attention = a memory matrix

drop the softmax and the sum can be reordered:
$$o_t = \sum_s (q_t \cdot k_s)\, v_s = \Big(\sum_s k_s v_s^\top\Big)^{\!\top} q_t = S^\top q_t, \qquad S = \sum_s k_s v_s^\top$$
- S: (d_k, d_v) = (128, 128) per head, a fixed size no matter how long the text is
- write: $S \leftarrow S + k v^\top$ ("store value v under key k")
- read: $o = S^\top q$ ("what value is stored under q?")
- decode is O(1) per token: update S, read S

it's a key → value associative memory. if the keys are orthonormal, reading with $q = k_i$ returns exactly $v_i$.

**the flaw:** it only ever adds. with 128-dim keys you can store ≤128 orthogonal things; after that keys overlap and every read returns a blur of everything. you can't overwrite and you can't forget.

## 3. the delta rule: overwrite instead of add

before writing, read what the memory currently says for this key, and write only the correction:
$$v_{\text{old}} = S^\top k \qquad \text{(what S currently returns for key k)}$$

$$S \leftarrow S + \beta\, k\, (v - v_{\text{old}})^\top \qquad \text{(move it toward v by a fraction } \beta\text{)}$$
- β ∈ (0, 1) = write strength. β = 1 replaces the old value completely; β = 0.1 nudges it
- this is exactly one gradient step on $\tfrac12 \lVert S^\top k - v \rVert^2$ with learning rate β: the layer **learns a tiny regression at test time**, token by token
- tiny example (d=2, β=1):
  - store k=(1,0) → v=5: S returns 5 for k
  - store k=(1,0) → v=7: plain linear attn now returns 5+7=12 (wrong); the delta rule computes v_old=5, writes +2, returns 7 (overwritten)

## 4. the gate: forgetting

the delta rule can overwrite but can't forget things nobody revisits. add a decay α_t ∈ (0, 1) on the whole memory:
$$S_t' = \alpha_t\, S_{t-1} \qquad \text{(fade everything a bit)}$$

$$S_t = S_t' + \beta_t\, k_t\, (v_t - S_t'^{\top} k_t)^\top \qquad \text{(then delta-rule write)}$$

$$o_t = S_t^\top q_t \qquad \text{(then read)}$$
(this is the literal loop in `torch_recurrent_gated_delta_rule`; the code stores $g = \log \alpha$)

- α near 1 = long memory; α near 0 = wipe the memory (e.g. at a document boundary)
- α and β are computed **from the current token**, so the model decides per token when to forget and how hard to write

## 5. how α and β are computed

$$\beta = \sigma(\texttt{in\_proj\_b}(x)) \qquad \text{shape (batch, seq, 32 heads)}$$

$$g = -e^{A_{\log}} \cdot \mathrm{softplus}(\texttt{in\_proj\_a}(x) + \texttt{dt\_bias}) < 0$$

$$\alpha = e^{g} \in (0, 1)$$
- borrowed from Mamba2: `exp(A_log)` = a learned per-head forgetting speed, `softplus(a + dt_bias)` = a per-token "time step" Δ
- $\alpha = e^{-A \Delta}$: a big Δ means "a lot of time passed", so more decay
- some heads learn slow decay (long memory), others fast (local context)

## 6. the full layer, step by step

numbers for Qwen3.5-4B: hidden 2560, 16 key heads, 32 value heads, head dim 128, conv kernel 4

| step | code | shape |
|---|---|---|
| project to q, k, v | `in_proj_qkv` | (B, T, 2560) → (B, T, 2048 + 2048 + 4096) |
| short causal conv + silu | `conv1d` (depthwise, kernel 4) | same; each channel mixes the current + last 3 tokens |
| split into heads | reshape | q, k: (B, T, 16, 128); v: (B, T, 32, 128) |
| gates | `in_proj_b`, `in_proj_a` | β, g: (B, T, 32) |
| share key heads | `repeat_interleave(2)` | q, k: (B, T, 32, 128) (each key head serves 2 value heads, like GQA) |
| l2-normalize q, k; q /= √128 | `l2norm` | unit-length keys |
| gated delta rule | chunked or recurrent kernel | out: (B, T, 32, 128); state S: (B, 32, 128, 128) |
| output gate | `norm(out) * silu(z)`, `z = in_proj_z(x)` | (B, T, 32, 128) |
| back to residual | `out_proj` | (B, T, 4096) → (B, T, 2560) |

why each piece:
- **short conv**: cheap local mixing (previous-token-head style patterns) before the memory sees anything; the kernel is only 4 tokens wide
- **l2 norm on k**: the delta update multiplies S by $(I - \beta k k^\top)$ along k. with ‖k‖ = 1 and β ≤ 1 that shrinks S along k and never blows it up. unnormalized keys would make the recurrence unstable over long sequences
- **per-head RMSNorm × silu(z)**: S's output scale varies a lot with context length, the norm fixes that; silu(z) is a learned per-channel valve on what leaves the layer

memory per layer: 32 heads × 128 × 128 = 524k numbers, constant. full attention at 100k tokens would cache 100k × (k + v) per head instead.

## 7. two kernels, same math

**recurrent** (`torch_recurrent_gated_delta_rule`): the literal loop over tokens from §4. used for decode (1 new token at a time).

**chunked** (`torch_chunk_gated_delta_rule`): used for prefill and for training/SAE capture (whole sequences). a loop over 4096 tokens is slow on a GPU, so:
- cut the sequence into chunks of 64
- **inside a chunk**, everything is computed with matmuls. the catch: token t's correction $v_t - S^\top k_t$ depends on what tokens 1..t−1 of the same chunk just wrote. that's a triangular linear system
  - $A = -(\beta K K^\top \odot \text{decay})$, strictly lower-triangular (`attn` in the code)
  - the `for i in range(1, chunk_size)` loop is forward substitution: it builds $(I + L)^{-1}$ (the "UT transform")
  - `value = attn @ (β v)` and `k_cumdecay = attn @ (β k · decay)` then give every token's actual write `v_new = value − k_cumdecay @ S_prev` in one shot
- **across chunks**, carry S with one recurrent step per chunk (`last_recurrent_state = ...`)
- output per token = (read of the state from before the chunk, decayed) + (causal intra-chunk attention over the chunk's writes):
  `core_attn_out = attn_inter + attn @ v_new`

so it's a loop of T/64 steps instead of T, with matmuls inside. same result as the recurrent kernel up to float error.

**fast path** (`GatedDeltaNet(fast=True)`, set by `config.use_fla` / `load_qwen3_5_model(fast=True)`): the same two kernels from flash-linear-attention (`fla-core`), `chunk_gated_delta_rule` and `fused_recurrent_gated_delta_rule`, as fused triton kernels. the torch versions above launch hundreds of tiny gpu ops per layer (the 63-step forward substitution, fp32 casts), so they are what the gpu waits on in training. the fast path is gpu only and torch.export can't trace it, so interpviz and cpu runs keep `fast=False`. `evoke/Qwen3_5_4b/run/verify_fast_parity.py` compares the two.

## 8. the cache (generation only)

per GDN layer, two tensors (see `Qwen3_5Cache` in `transformer/qwen3_5.py`):
- `conv_states`: the last 3 pre-conv q/k/v inputs (the conv kernel is 4 wide), shape (B, 8192, 3)
- `recurrent_states`: S, shape (B, 32, 128, 128)

full-attention layers still keep a normal growing KV cache. that's why only every 4th layer is full attention: they do exact long-range lookups, the 24 GDN layers keep the cost flat.

## 9. what it means for interp

- GDN layers write into the residual stream like attention (`out_proj` output added to x), so residual SAEs / WCCs don't care which kind of layer produced it
- "which past token did this come from" is not directly visible: there's no attention pattern, everything passes through S. the closest analogue is the per-token write `v_new` and the read $S^\top q$
- α per head per token says when a head flushes its memory, β says when it writes
