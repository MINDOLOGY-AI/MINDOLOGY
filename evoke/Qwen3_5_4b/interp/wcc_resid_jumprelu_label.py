# picks + labels for the weakly causal crosscoder's features (wcc_resid_jumprelu.py): the same autointerp as the SAEs
# (synapse/interp/DOC.md: picks, label). a feature of encoder layer i is picked and labeled from its activations on
# layer i's residual, exactly like an SAE feature of layer i.
# run from repo root, two phases:
#   torchrun --nproc_per_node 8 -m evoke.Qwen3_5_4b.interp.wcc_resid_jumprelu_label picks
#       the gpu box, after training: every rank gathers the picks of its own encoder layers into picks/rank<r>/, rank 0
#       then moves them into picks/ under one meta.json
#   python -m evoke.Qwen3_5_4b.interp.wcc_resid_jumprelu_label label
#       any machine with picks/, the eval.bin prefix the picks index into, and the openrouter key. resumes from the
#       jsonl files; the summary at the end needs core.json from the trainer's eval
# inputs:  weights/evoke/Qwen3_5_4b/wcc_resid_jumprelu/{L<i>.pt, shared.pt}, results/.../core.json (label summary)
# outputs: results/Qwen3_5_4b/wcc_resid_jumprelu/{picks/, labels/, autointerp.json, summary.md}

import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from evoke.Qwen3_5_4b.run.loader import load_qwen3_5_model, load_qwen3_5_tokenizer
from synapse.interp.hooked_model import HookedModel
from synapse.interp.label import label_units
from synapse.interp.picks import gather_picks
from synapse.probes.sae.WCCEncoder import WCCEncoder

BIN_DIR = Path.cwd() / "data" / "datasteps" / "tokenized" / "qwen3_5_4b_interp_dataset"
WEIGHTS_DIR = Path.cwd() / "weights" / "evoke" / "Qwen3_5_4b" / "wcc_resid_jumprelu"
RESULTS_DIR = Path.cwd() / "results" / "Qwen3_5_4b" / "wcc_resid_jumprelu"
N_LAYERS = 32
PICK_CHUNKS = 16000  # x 128 = 2.05M tokens, same as sae_resid_topk
PICK_BATCH_CHUNKS = 16
MODEL = "deepseek/deepseek-v4-flash-0731"
WORKERS = 500  # units in flight (sae_resid_topk: 200); 655k features
SEED = 21  # the picks' importance-weighted draws


class HookedQwenWCC(HookedModel):
    # exposes encoder layers' feature activations as hook names, so the picks tracker treats features like neurons
    def __init__(self, model, encoders):
        # encoders: {name: (layer, WCCEncoder on the model's device)}; the encoder reads resid_post.<layer>
        self.encoders = encoders
        super().__init__(model)

    def hook_points(self):
        return {f"resid.{layer}": (self.model.model.language_model.layers[layer], False) for layer, _ in self.encoders.values()}

    def run(self, tokens, names):
        # names: encoder names -> self.acts[name] = (b, T, F) feature activations
        super().run(tokens, sorted({f"resid.{self.encoders[n][0]}" for n in names}))
        with torch.no_grad():
            for name in names:
                layer, enc = self.encoders[name]
                resid = self.acts[f"resid.{layer}"]  # (b, T, d_in)
                # (b*T, d_in) -> (b*T, F) -> (b, T, F)
                self.acts[name] = enc.acts(resid.flatten(0, 1)).view(*resid.shape[:2], -1)


def main(phase, pick_chunks=PICK_CHUNKS, label_limit=None, weights_dir=WEIGHTS_DIR, results_dir=RESULTS_DIR):
    # label_limit: label only the first n units per layer (None = every unit); for smoke tests
    assert phase in ("picks", "label"), f"phase must be picks or label, got {phase!r}"
    picks_dir = results_dir / "picks"
    labels_dir = results_dir / "labels"
    names = [f"L{i}" for i in range(N_LAYERS)]
    if phase == "picks":
        assert not (picks_dir / "meta.json").exists(), f"{picks_dir} already gathered"
        cuda = torch.cuda.is_available()
        dist.init_process_group("nccl" if cuda else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        dev = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if cuda else torch.device("cpu")
        if cuda:
            torch.cuda.set_device(dev)
        torch.manual_seed(SEED)
        meta = json.loads((BIN_DIR / "meta.json").read_text())
        lm = load_qwen3_5_model(device=dev)[0]
        lm.config.use_cache = False
        # every rank costs the same per layer here (one encoder + one tracker), so layers go round robin
        encoders = {names[i]: (i, WCCEncoder(weights_dir / f"L{i}.pt", weights_dir / "shared.pt", i).to(dev))
                    for i in range(rank, N_LAYERS, world)}
        # the picks only read the first pick_chunks chunks of eval.bin, so the source is that prefix
        gather_picks(HookedQwenWCC(lm, encoders), BIN_DIR / "eval.bin", (pick_chunks, meta["chunk_size"]), list(encoders),
                     pick_chunks, picks_dir / f"rank{rank}", PICK_BATCH_CHUNKS)
        dist.barrier()
        if rank == 0:
            hooks = {}  # {name: D} over every rank
            for r in range(world):
                d = picks_dir / f"rank{r}"
                pm = json.loads((d / "meta.json").read_text())
                hooks.update(pm["hooks"])
                (d / "meta.json").unlink()
                for f in d.iterdir():
                    f.rename(picks_dir / f.name)
                d.rmdir()
            assert set(hooks) == set(names), f"missing picks for {set(names) - set(hooks)}"
            pm["hooks"] = {n: hooks[n] for n in names}
            (picks_dir / "meta.json").write_text(json.dumps(pm, indent=2))
            print(f"picks for {len(names)} layers in {picks_dir}", flush=True)
        dist.destroy_process_group()
        return

    # phase == "label": label + score every feature (resumes from the jsonl files)
    tokenizer = load_qwen3_5_tokenizer()
    units = {n: list(range(label_limit)) for n in names} if label_limit else None
    n_failed = asyncio.run(label_units(picks_dir, labels_dir, tokenizer, MODEL, names, workers=WORKERS, units=units,
                                       unit_word="feature"))
    assert n_failed == 0, f"{n_failed} units failed labeling, see {labels_dir / 'errors.json'}; rerun to retry them"

    # --- summary, next to the trainer's core eval ---
    core = json.loads((results_dir / "core.json").read_text())
    pm = json.loads((picks_dir / "meta.json").read_text())
    K_picks = pm["top_k"] + pm["iw_k"]
    summary = {}  # {name: core stats + autointerp stats}
    rows = []  # [str] markdown table rows
    for n in names:
        W = pm["window_before"] + 1 + pm["window_after"]
        # (D, K_picks) each pick's activation at its center token; the tracker also keeps zero-activation picks, so
        # a unit fired at a pick iff this is > 0 (label.py's "no positive activation" test)
        pick_act = np.fromfile(picks_dir / f"{n}.windows.bin", dtype=np.float16).reshape(pm["hooks"][n], K_picks, W)[:, :, pm["window_before"]]
        rs = [json.loads(l) for l in open(labels_dir / f"{n}.jsonl")]
        sc = np.array([r["score"] for r in rs if r["score"] is not None])
        summary[n] = {**core["saes"][n], "never_fired_frac": float((pick_act[:, 0] <= 0).mean()),
                      "full_picks_frac": float((pick_act > 0).all(1).mean()), "n_scored": int(len(sc)),
                      "autointerp_mean": float(sc.mean()), "autointerp_median": float(np.median(sc)),
                      "autointerp_frac_ge_0.7": float((sc >= 0.7).mean())}
        s = summary[n]
        rows.append(f"| {n} | {s['l0']:.1f} | {s['recon_err_pct']:.1f}% | {s['ce_increase_pct']:.1f}% | {s['never_fired_frac']:.1%} | "
                    f"{s['full_picks_frac']:.0%} | {s['autointerp_mean']:.3f} | {s['autointerp_median']:.2f} | {s['autointerp_frac_ge_0.7']:.0%} |")
    (results_dir / "autointerp.json").write_text(json.dumps(summary, indent=2))
    table = "\n".join(["| layer | L0 (its features) | recon err (its residual) | CE increase | never fired | >=40 firings | autointerp mean | median | >=0.7 |",
                       "|---|---|---|---|---|---|---|---|---|", *rows])
    cfg = core["config"]
    (results_dir / "summary.md").write_text(
        f"# weakly causal crosscoder (JumpReLU, {cfg['features_per_layer']} features per layer) on Qwen3.5-4B resid_post\n\n"
        f"train {cfg['train_tokens']:,} tokens, final lambda {cfg['lam']}, total L0 {core['l0_total']:.0f}. picks over "
        f"{pm['n_chunks'] * pm['context_chunk_size']:,} tokens. labeler {MODEL}. clean CE {core['ce_clean']:.3f}.\n\n{table}\n")
    print(table, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
