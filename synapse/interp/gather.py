# runs a hooked model over a tokenized chunk dataset and streams every batch through a TopKTracker,
# writing only the per-unit label picks + quantile grid (see synapse/interp/DOC.md, dynamic analysis).
# model-agnostic: the caller picks the HookedModel subclass and the hook names.

import numpy as np
import torch

from synapse.interp.topk_tracker import TopKTracker


def gather_picks(hooked, source_bin, source_shape, hook_names, n_chunks, out_dir, batch_chunks):
    # hooked: HookedModel instance, model already on device
    # source_bin: tokenized .bin of shape source_shape = (num_chunks, ctx_len) int32
    # the first n_chunks chunks are gathered in order (the source is already shuffled at chunk level)
    num_chunks, ctx_len = source_shape
    assert n_chunks <= num_chunks, f"asked for {n_chunks} chunks, source has {num_chunks}"
    device = next(hooked.model.parameters()).device
    # (num_chunks, ctx_len) - token ids, read lazily
    tokens = np.memmap(source_bin, dtype=np.int32, mode="r", shape=source_shape)

    # warm-up forward on the first batch just to read each hook's unit dim D
    hooked.run(torch.from_numpy(np.array(tokens[:batch_chunks])).to(device), hook_names)
    tracker = TopKTracker({name: hooked.acts[name].shape[-1] for name in hook_names}, ctx_len, n_chunks, device)

    for start in range(0, n_chunks, batch_chunks):
        end = min(start + batch_chunks, n_chunks)
        # (b, ctx_len)
        batch = torch.from_numpy(np.array(tokens[start:end])).to(device)
        hooked.run(batch, hook_names)
        chunk_ids = torch.arange(start, end, device=device)
        for name in hook_names:
            assert hooked.acts[name].shape[:2] == (end - start, ctx_len), f"{name}: {hooked.acts[name].shape}"
            tracker.update(name, hooked.acts[name], chunk_ids)
        print(f"  chunks {end}/{n_chunks}")

    tracker.save(out_dir, {"n_chunks": n_chunks, "context_chunk_size": ctx_len, "source_dataset": str(source_bin)})
    print(f"wrote picks for {len(hook_names)} hooks over {n_chunks} chunks to {out_dir}")
