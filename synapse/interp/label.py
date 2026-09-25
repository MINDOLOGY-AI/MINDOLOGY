# labels every unit of a picks dir (written by picks.gather_picks) with an llm, then scores each label by
# detection on held-out examples. see synapse/interp/DOC.md, "label pipeline". model-agnostic: needs only the
# picks dir, the tokenizer that produced the source dataset, and an openrouter model name.

import asyncio
import json
from collections import deque
from pathlib import Path

import numpy as np
from transformers.convert_slow_tokenizer import bytes_to_unicode

from synapse.interp.openrouter import chat, OpenRouterGaveUp
from synapse.interp.interp_prompt import (render_window, generate_messages, score_messages, parse_score_answer,
                                          GENERATE_MAX_TOKENS)

ACT_THRESHOLD_FRAC = 0.01  # a token is marked <<active>> in label windows if its activation > this * the unit's max
FIRE_PERCENTILE = 99  # a random window "fires" if any token exceeds the unit's value at this percentile
SEED = 0
# circuit breaker: abort if more than this fraction of the last BREAKER_WINDOW finished units failed (network down / api broken)
BREAKER_WINDOW = 1000
BREAKER_MAX_FAIL_FRAC = 0.02


async def label_units(picks_dir, labels_dir, tokenizer, model, hook_names, workers=200, units=None, unit_word="neuron"):
    # writes <labels_dir>/<hook>.jsonl (one line per labeled unit, appended) and <labels_dir>/errors.json
    # (units whose api calls exhausted every retry this run; they are not in the jsonl, so a rerun retries them).
    # returns the number of failed units.
    # unit_word: what the prompts call a unit ("neuron" for MLP neurons, "feature" for SAE features)
    # units: optional {hook_name: [unit ids]} to label a subset instead of every unit
    # workers: max units in flight at once (a unit holds its slot across both of its calls, otherwise the
    # fifo semaphore would queue every second call behind all 131k first calls and nothing completes for ages)
    picks_dir = Path(picks_dir)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((picks_dir / "meta.json").read_text())
    n_chunks, L = meta["n_chunks"], meta["context_chunk_size"]
    wb, wa = meta["window_before"], meta["window_after"]
    W = wb + 1 + wa
    top_k, iw_k, n_random = meta["top_k"], meta["iw_k"], meta["n_random"]
    # (n_chunks, L) token ids of the chunks the picks index into (a prefix of the source .bin)
    tokens = np.memmap(meta["source_dataset"], dtype=np.int32, mode="r", shape=(n_chunks, L))
    # {byte-level BPE char: byte}, inverse of the tokenizer's byte -> printable-char map
    byte_of = {c: b for b, c in bytes_to_unicode().items()}
    # [bytes] raw utf-8 bytes per token id (a token can hold part of a multi-byte character)
    token_bytes = [bytes(byte_of[c] for c in tokenizer.convert_ids_to_tokens(i)) for i in range(len(tokenizer))]
    # slots used for the label prompt vs held out for the test (first half of each of top-k and iw)
    label_slots = list(range(top_k // 2)) + list(range(top_k, top_k + iw_k // 2))
    test_slots = list(range(top_k // 2, top_k)) + list(range(top_k + iw_k // 2, top_k + iw_k))

    # load every hook up front so all (hook, unit) jobs share one thread pool
    data = {}  # {name: {field: array}}
    todo = []  # [(name, unit)]
    for name in hook_names:
        d = meta["hooks"][name]
        K = top_k + iw_k
        data[name] = {
            "pick_chunk": np.fromfile(picks_dir / f"{name}.pick_chunk.bin", dtype=np.int32).reshape(d, K),
            "pick_pos": np.fromfile(picks_dir / f"{name}.pick_pos.bin", dtype=np.int8).reshape(d, K).astype(int),
            "windows": np.fromfile(picks_dir / f"{name}.windows.bin", dtype=np.float16).reshape(d, K, W).astype(np.float32),
            "rand_chunk": np.fromfile(picks_dir / f"{name}.random_chunk.bin", dtype=np.int32).reshape(d, n_random),
            "rand_pos": np.fromfile(picks_dir / f"{name}.random_pos.bin", dtype=np.int8).reshape(d, n_random).astype(int),
            "rand_win": np.fromfile(picks_dir / f"{name}.random_windows.bin", dtype=np.float16).reshape(d, n_random, W).astype(np.float32),
            "quant": np.fromfile(picks_dir / f"{name}.quantiles.bin", dtype=np.float16).reshape(d, meta["n_quantiles"]).astype(np.float32),
        }
        out_path = labels_dir / f"{name}.jsonl"
        done = {json.loads(line)["unit"] for line in out_path.read_text().splitlines()} if out_path.exists() else set()  # {int}
        todo += [(name, u) for u in (units[name] if units else range(d)) if u not in done]
        print(f"{name}: {len([1 for n, _ in todo if n == name])} units to label ({len(done)} already done)", flush=True)

    sem = asyncio.Semaphore(workers)

    async def label_one(name, u):
        h = data[name]
        pick_chunk, pick_pos, windows = h["pick_chunk"], h["pick_pos"], h["windows"]
        unit_max = windows[u, 0, wb]
        if pick_chunk[u, 0] < 0 or unit_max <= 0:
            return name, {"unit": u, "label": None, "score": None, "reason": "no positive activation"}
        rng = np.random.default_rng(SEED + u)
        async with sem:
            try:
                return name, await _label_unit(name, u, unit_max, rng)
            except OpenRouterGaveUp as e:
                return name, {"unit": u, "error": str(e)}

    async def _label_unit(name, u, unit_max, rng):
        h = data[name]
        pick_chunk, pick_pos, windows = h["pick_chunk"], h["pick_pos"], h["windows"]

        def text(c, p, win, marked):
            window_bytes = [token_bytes[i] for i in tokens[c, p - wb:p + wa + 1]]
            return render_window(window_bytes, (win > ACT_THRESHOLD_FRAC * unit_max).tolist() if marked else None)

        # label prompt: valid label slots, strongest first (slots are already ordered within top-k / iw)
        lab = [s for s in label_slots if pick_chunk[u, s] >= 0]
        lab.sort(key=lambda s: -windows[u, s, wb])
        raw_label = await chat(generate_messages([text(pick_chunk[u, s], pick_pos[u, s], windows[u, s], True) for s in lab], unit_word),
                               model, GENERATE_MAX_TOKENS)
        label = raw_label.split("activates on")[-1].rstrip(".").strip()

        # test set: held-out picks (fire by construction) + random windows (fire iff any token > p99)
        fire_at = h["quant"][u, FIRE_PERCENTILE]
        items = [(("pick", s), True) for s in test_slots if pick_chunk[u, s] >= 0]  # [((kind, slot), truth)]
        items += [(("random", r), bool(h["rand_win"][u, r].max() > fire_at)) for r in range(n_random) if h["rand_chunk"][u, r] >= 0]
        rng.shuffle(items)
        rendered = [text(pick_chunk[u, s], pick_pos[u, s], None, False) if kind == "pick"
                    else text(h["rand_chunk"][u, s], h["rand_pos"][u, s], None, False) for (kind, s), _ in items]
        raw_test = await chat(score_messages(label, rendered, unit_word), model, 2 * len(items) + 5)
        said = parse_score_answer(raw_test, len(items))
        truth = [t for _, t in items]
        if said is None:
            score = None
        else:
            pos = [s for s, t in zip(said, truth) if t]
            neg = [not s for s, t in zip(said, truth) if not t]
            rates = [np.mean(x) for x in (pos, neg) if x]  # balanced accuracy over the classes present
            score = float(np.mean(rates))
        return {"unit": u, "label": label, "score": score,
                "test": {"order": [list(k) for k, _ in items], "truth": truth, "said": said},
                "raw": {"label": raw_label, "test": raw_test}}

    # {name: open jsonl handle}, appended from the main thread as futures complete
    files = {name: open(labels_dir / f"{name}.jsonl", "a") for name in hook_names}
    failed = []  # [{"hook": str, "unit": int, "error": str}]
    recent = deque(maxlen=BREAKER_WINDOW)  # [bool] failed?, last BREAKER_WINDOW finished units

    def write_errors():
        (labels_dir / "errors.json").write_text(json.dumps({"n_failed": len(failed), "failed": failed}, indent=1))

    tasks = [asyncio.create_task(label_one(name, u)) for name, u in todo]
    for i, fut in enumerate(asyncio.as_completed(tasks), 1):
        name, rec = await fut
        recent.append("error" in rec)
        if "error" in rec:
            failed.append({"hook": name, **rec})
            if len(recent) == BREAKER_WINDOW and sum(recent) > BREAKER_MAX_FAIL_FRAC * BREAKER_WINDOW:
                write_errors()
                raise RuntimeError(f"{sum(recent)}/{BREAKER_WINDOW} recent units failed, api/network down: {rec['error']}")
        else:
            files[name].write(json.dumps(rec, ensure_ascii=False) + "\n")
            files[name].flush()
        if i % 500 == 0 or i == len(tasks):
            print(f"  {i}/{len(tasks)} units done, {len(failed)} failed", flush=True)
    for f in files.values():
        f.close()
    write_errors()
    return len(failed)
