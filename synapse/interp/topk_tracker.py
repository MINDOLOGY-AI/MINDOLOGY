# streaming per-unit example picker for labeling (see synapse/interp/DOC.md, dynamic analysis).
# per hook, per unit, over the whole run, vectorized over units on device:
#   top-k by activation, greedy-deduped at run time so no two picks collide (same chunk within the window)
#   importance-weighted draw ~ activation^2, streamed as weighted reservoir keys log(u) / a^2: the run keeps
#     the IW_RESERVOIR highest keys with no dedup; at save the reservoir is walked from the highest key down,
#     skipping entries that collide with a final top-k pick or an already-accepted iw pick, until IW_K are
#     accepted. by exponential memorylessness that walk is exactly sequential ~a^2 draws with suppression
#     over the non-top-k tokens. a unit that exhausts the reservoir gets empty slots.
#   random windows: fixed (chunk, pos) targets drawn once per run, captured for every unit as they pass;
#     at save, per unit, the ones colliding with any of its picks are dropped
#   quantile grid: each batch's exact quantiles at integer percentiles 0..100, token-weighted running average
# every kept pick carries the unit's activation over its label window (+-10 tokens), so no second pass is needed.

import json
from pathlib import Path

import numpy as np
import torch

TOP_K = 20
IW_K = 20
# reservoir must cover the skipped draws (top-k zone tokens and neighbors of accepted picks are the heaviest)
IW_RESERVOIR = 200
N_RANDOM = 20
N_RANDOM_CANDIDATES = 40
WINDOW_BEFORE = 10
WINDOW_AFTER = 10
WINDOW = WINDOW_BEFORE + 1 + WINDOW_AFTER
N_QUANTILES = 101  # percentiles 0, 1, ..., 100
RANDOM_SEED = 0


class TopKTracker:
    def __init__(self, hook_dims, ctx_len, n_chunks, device):
        # hook_dims: {name: D}; n_chunks: chunks the run will cover (chunk ids 0..n_chunks-1)
        assert ctx_len > WINDOW, f"ctx_len {ctx_len} too short for a {WINDOW} window"
        self.ctx_len = ctx_len
        self.device = device
        # {name: {field: tensor}} running picks per hook, first dim = slot, second = unit
        self.top = {name: self._empty(TOP_K, d) for name, d in hook_dims.items()}
        self.iw = {name: self._empty(IW_RESERVOIR, d) for name, d in hook_dims.items()}
        # (WINDOW,) offsets from the firing token to each window token
        self.win_offsets = torch.arange(-WINDOW_BEFORE, WINDOW_AFTER + 1, device=device)

        # random targets, same for every unit: (N_RANDOM_CANDIDATES,) chunk ids and valid center positions
        g = torch.Generator().manual_seed(RANDOM_SEED)
        self.rand_chunk = torch.randint(n_chunks, (N_RANDOM_CANDIDATES,), generator=g).int().to(device)
        self.rand_pos = torch.randint(WINDOW_BEFORE, ctx_len - WINDOW_AFTER, (N_RANDOM_CANDIDATES,), generator=g).to(torch.int8).to(device)
        # {name: (N_RANDOM_CANDIDATES, D, WINDOW)} every unit's window on each random target, filled as chunks pass
        self.rand_win = {name: torch.zeros((N_RANDOM_CANDIDATES, d, WINDOW), dtype=torch.float16, device=device) for name, d in hook_dims.items()}
        self.rand_seen = torch.zeros(N_RANDOM_CANDIDATES, dtype=torch.bool, device=device)

        # {name: (N_QUANTILES, D)} running batch-averaged quantile grid, {name: int} tokens averaged in so far
        self.quant = {name: torch.zeros((N_QUANTILES, d), device=device) for name, d in hook_dims.items()}
        self.n_seen = {name: 0 for name in hook_dims}
        # (N_QUANTILES,) percentiles as fractions
        self.grid = torch.arange(N_QUANTILES, device=device) / (N_QUANTILES - 1)

    def _empty(self, k, d):
        return {
            "score": torch.full((k, d), float("-inf"), device=self.device),  # activation (top) or reservoir key (iw)
            "chunk": torch.full((k, d), -1, dtype=torch.int32, device=self.device),
            "pos": torch.zeros((k, d), dtype=torch.int8, device=self.device),
            "win": torch.zeros((k, d, WINDOW), dtype=torch.float16, device=self.device),
        }

    def _greedy_pick(self, score, k):
        # score: (b*L, D), -inf where not allowed. picks the best k rows per unit, blocking the collision
        # zone (same chunk, within WINDOW-1) after each pick. returns rows (k, D) and their scores (k, D).
        L = self.ctx_len
        rows = torch.arange(score.shape[0], device=self.device).unsqueeze(1)  # (b*L, 1)
        picked, picked_score = [], []  # [(D,)] x k
        for _ in range(k):
            best = score.argmax(dim=0)  # (D,)
            picked.append(best)
            picked_score.append(score.gather(0, best.unsqueeze(0)).squeeze(0))
            same_chunk = (rows // L) == (best // L).unsqueeze(0)  # (b*L, D)
            near = (rows - best.unsqueeze(0)).abs() <= WINDOW - 1
            score = score.masked_fill(same_chunk & near, float("-inf"))
        return torch.stack(picked), torch.stack(picked_score)

    def update(self, name, acts, chunk_ids):
        # acts: (b, ctx_len, D) this batch's activations for one hook; chunk_ids: (b,) int source chunk of each row
        b, L, d = acts.shape
        assert L == self.ctx_len
        # (b*L, D) fp32 flat over the batch, row r = chunk r // L, position r % L
        flat = acts.reshape(b * L, d).float()
        # (b*L, 1) True where a window around this position stays inside its chunk
        pos = torch.arange(b * L, device=self.device) % L
        valid = ((pos >= WINDOW_BEFORE) & (pos <= L - 1 - WINDOW_AFTER)).unsqueeze(1)

        # --- top-k candidates: greedy over activations ---
        cand, cand_score = self._greedy_pick(flat.masked_fill(~valid, float("-inf")), TOP_K)
        self._merge(self.top[name], flat, chunk_ids, cand, cand_score, TOP_K)

        # --- iw candidates: highest reservoir keys, weight = relu(a)^2 (w == 0 -> -inf, never picked) ---
        keys = torch.log(torch.rand_like(flat)) / flat.clamp(min=0).pow(2)
        # a short final batch may have fewer rows than the reservoir
        cand_score, cand = torch.topk(keys.masked_fill(~valid, float("-inf")), min(IW_RESERVOIR, b * L), dim=0)
        self._merge(self.iw[name], flat, chunk_ids, cand, cand_score, IW_RESERVOIR)

        # --- random targets whose chunk is in this batch: capture every unit's window ---
        # (N_RANDOM_CANDIDATES, b) which target sits in which batch row
        match = self.rand_chunk.unsqueeze(1) == chunk_ids.to(self.device).int().unsqueeze(0)
        j_idx, i_idx = match.nonzero(as_tuple=True)
        if j_idx.numel():
            # (m, WINDOW) flat rows of each matched target's window
            win_rows = (i_idx * L + self.rand_pos[j_idx].long()).unsqueeze(1) + self.win_offsets.unsqueeze(0)
            # (m, WINDOW, D) -> (m, D, WINDOW)
            self.rand_win[name][j_idx] = flat[win_rows].transpose(1, 2).half()
            self.rand_seen[j_idx] = True

        # --- quantile grid: this batch's exact quantiles over all tokens, averaged in by token count ---
        # (b*L, D) ascending per unit
        sorted_flat = torch.sort(flat, dim=0).values
        # value at fractional index p*(N-1) with linear interpolation (torch.quantile's rule)
        pos_q = self.grid * (b * L - 1)
        lo, hi = pos_q.floor().long(), pos_q.ceil().long()
        w_hi = (pos_q - lo.float()).unsqueeze(1)  # (N_QUANTILES, 1)
        q_batch = sorted_flat[lo] * (1 - w_hi) + sorted_flat[hi] * w_hi  # (N_QUANTILES, D)
        n_seen = self.n_seen[name]
        self.quant[name] = (self.quant[name] * n_seen + q_batch * (b * L)) / (n_seen + b * L)
        self.n_seen[name] = n_seen + b * L

    def _merge(self, state, flat, chunk_ids, cand, cand_score, k):
        # cand: (k, D) flat row indices of this batch's candidates, cand_score: (k, D) their scores.
        # candidates come from chunks never seen before, so dedup against existing entries is unnecessary.
        d = flat.shape[1]
        L = self.ctx_len
        units = torch.arange(d, device=self.device).view(1, d, 1)
        # dead candidates (score -inf: unit had nothing valid in this batch) carry arbitrary rows;
        # clamp so their garbage window gather stays in bounds - they are dropped by score anyway
        win_rows = (cand.unsqueeze(2) + self.win_offsets.view(1, 1, WINDOW)).clamp_(0, flat.shape[0] - 1)
        # (k, D, WINDOW) the unit's activation on every window token of each candidate
        cand_win = flat[win_rows, units].half()
        cand_chunk = chunk_ids.to(self.device)[cand // L].int()
        cand_pos = (cand % L).to(torch.int8)

        # keep the best k of old + new, per unit
        score = torch.cat([state["score"], cand_score])  # (2k, D)
        keep = torch.topk(score, k, dim=0).indices  # (k, D)
        state["score"] = score.gather(0, keep)
        state["chunk"] = torch.cat([state["chunk"], cand_chunk]).gather(0, keep)
        state["pos"] = torch.cat([state["pos"], cand_pos]).gather(0, keep)
        state["win"] = torch.cat([state["win"], cand_win]).gather(0, keep.unsqueeze(2).expand(-1, -1, WINDOW))

    def save(self, out_dir, meta):
        # writes per-hook arrays (D, slot, ...): picks = top-k slots then iw slots, randoms separately
        assert bool(self.rand_seen.all()), "some random target chunks were never passed to update()"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in self.top:
            top, iw = self.top[name], self.iw[name]
            d = top["score"].shape[1]

            # iw: walk the reservoir from the highest key down (topk left it sorted), accept an entry unless it
            # collides with a top-k pick or an already-accepted iw pick, until IW_K accepted per unit
            acc_chunk = torch.full((IW_K, d), -1, dtype=torch.int32, device=self.device)
            acc_pos = torch.zeros((IW_K, d), dtype=torch.int8, device=self.device)
            acc_src = torch.zeros((IW_K, d), dtype=torch.long, device=self.device)  # reservoir slot of each accepted
            n_acc = torch.zeros(d, dtype=torch.long, device=self.device)
            units = torch.arange(d, device=self.device)
            for j in range(IW_RESERVOIR):
                cj, pj = iw["chunk"][j], iw["pos"][j].int()  # (D,)
                hit_top = ((cj == top["chunk"]) & ((pj - top["pos"].int()).abs() <= WINDOW - 1)).any(dim=0)
                hit_acc = ((cj == acc_chunk) & ((pj - acc_pos.int()).abs() <= WINDOW - 1)).any(dim=0)
                ok = ~hit_top & ~hit_acc & ~torch.isinf(iw["score"][j]) & (n_acc < IW_K)
                slot, u = n_acc[ok], units[ok]
                acc_chunk[slot, u] = cj[ok]
                acc_pos[slot, u] = iw["pos"][j][ok]
                acc_src[slot, u] = j
                n_acc += ok
            picks = {
                "chunk": torch.cat([top["chunk"].masked_fill(torch.isinf(top["score"]), -1), acc_chunk]),
                "pos": torch.cat([top["pos"], acc_pos]),
                "win": torch.cat([top["win"], iw["win"].gather(0, acc_src.unsqueeze(2).expand(-1, -1, WINDOW))]),
            }

            # random: per unit, drop candidates colliding with any of its picks, keep the first N_RANDOM in order
            rc = self.rand_chunk.unsqueeze(1).expand(-1, d)  # (N_RANDOM_CANDIDATES, D)
            rp = self.rand_pos.unsqueeze(1).expand(-1, d)
            # (N_RANDOM_CANDIDATES, D) candidate collides with any of the unit's picks (same chunk, within the window)
            same_chunk = rc.unsqueeze(1) == picks["chunk"].unsqueeze(0)  # (N_RANDOM_CANDIDATES, 2*K, D)
            near = (rp.unsqueeze(1).int() - picks["pos"].unsqueeze(0).int()).abs() <= WINDOW - 1
            ok = ~(same_chunk & near).any(dim=1)
            # earliest ok candidates first: score -j for ok, -inf otherwise, top N_RANDOM
            order = torch.where(ok, -torch.arange(N_RANDOM_CANDIDATES, device=self.device, dtype=torch.float32).unsqueeze(1), torch.tensor(float("-inf"), device=self.device))
            r_score, r_keep = torch.topk(order, N_RANDOM, dim=0)  # (N_RANDOM, D)
            rand = {
                "chunk": rc.gather(0, r_keep).masked_fill(torch.isinf(r_score), -1),
                "pos": rp.gather(0, r_keep),
                "win": self.rand_win[name].gather(0, r_keep.unsqueeze(2).expand(-1, -1, WINDOW)),
            }

            for prefix, arrays in [("pick", picks), ("random", rand)]:
                for field, dtype in [("chunk", np.int32), ("pos", np.int8), ("win", np.float16)]:
                    fname = f"{prefix}_windows" if field == "win" else f"{prefix}_{field}"
                    if prefix == "pick" and field == "win":
                        fname = "windows"
                    arrays[field].transpose(0, 1).contiguous().cpu().numpy().astype(dtype).tofile(out_dir / f"{name}.{fname}.bin")
            self.quant[name].T.contiguous().cpu().numpy().astype(np.float16).tofile(out_dir / f"{name}.quantiles.bin")
        meta = {
            **meta,
            "window_before": WINDOW_BEFORE,
            "window_after": WINDOW_AFTER,
            "top_k": TOP_K,
            "iw_k": IW_K,
            "n_random": N_RANDOM,
            "n_quantiles": N_QUANTILES,
            "n_tokens": next(iter(self.n_seen.values())),
            "hooks": {name: int(s["score"].shape[1]) for name, s in self.top.items()},
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
