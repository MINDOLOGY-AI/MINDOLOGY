# the training loop. batch-indexed (global batch counter, epochs only pick the shuffle seed), resumable from save_path.
# model protocol: model.compute_loss(batch) -> (loss, metrics {str: float}); the batch is whatever the dataset yields
# (tensor, tuple, or dict), moved to cuda here. params stay in the dtype the caller set (fp32 for finetuning);
# the forward runs under bf16 autocast when autocast_dtype is set.
# eval_fn(model) -> (metrics {str: float}, stop bool) runs every batches_per_eval batches; stop=True ends training.
# eval_dataset, if given, contributes one batch of eval_<metric> per log like simple_train.
# checkpoints are epoch_0_batch_N.pt via train_helpers.save_checkpoint (only the newest is kept).

import os
import time

import torch

from synapse.train.train_helpers import save_checkpoint, load_latest_checkpoint, log_batch, to_cuda
from synapse.train.data_to_loaders import dataset_to_dataloader


def train(
    model,
    dataset,
    batch_size,
    optimizer,
    save_path,
    max_batches,
    scheduler=None,
    accumulation_steps=1,
    clip_grad_norm=1.0,
    autocast_dtype=torch.bfloat16,
    batches_per_log=10,
    batches_per_save=500,
    batches_per_eval=None,
    eval_fn=None,
    eval_dataset=None,
    eval_batch_size=None,
    num_workers=2,
):
    # returns the global batch index training stopped at
    assert (batches_per_eval is None) == (eval_fn is None), "batches_per_eval and eval_fn go together"
    os.makedirs(save_path, exist_ok=True)
    model.cuda()
    model.train()
    autocast = torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None)

    batch_idx = load_latest_checkpoint(save_path, model, optimizer, scheduler)
    batches_per_epoch = len(dataset) // batch_size
    assert batches_per_epoch > 0, f"dataset of {len(dataset)} rows gives no full batch of {batch_size}"

    # {metric name: [values logged since last save]}
    losses = {}
    eval_iter = None
    if eval_dataset is not None:
        eval_iter = iter(dataset_to_dataloader(eval_dataset, eval_batch_size or batch_size, cur_epoch=0, num_workers=num_workers))

    # resume mid-epoch: the loader for the current epoch is reseeded and fast-skipped to the right batch.
    # datasets that change per epoch (ReplayDataset) get told the epoch before their loader is built
    epoch = batch_idx // batches_per_epoch
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    data_iter = iter(dataset_to_dataloader(dataset, batch_size, cur_epoch=epoch, start_batch=batch_idx % batches_per_epoch, num_workers=num_workers))

    while batch_idx < max_batches:
        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)
        # (1,) on device, summed over accumulation steps, synced only when logging
        acc_loss = torch.zeros((), device="cuda")
        acc_metrics = {}  # {str: float}, last micro-batch's metrics
        for _ in range(accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if hasattr(dataset, "set_epoch"):
                    dataset.set_epoch(epoch)
                data_iter = iter(dataset_to_dataloader(dataset, batch_size, cur_epoch=epoch, num_workers=num_workers))
                batch = next(data_iter)
            with autocast:
                loss, acc_metrics = model.compute_loss(to_cuda(batch))
            (loss / accumulation_steps).backward()
            acc_loss += loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        batch_idx += 1

        if batch_idx % batches_per_log == 0:
            losses.setdefault("total", []).append(acc_loss.item() / accumulation_steps)
            for k, v in acc_metrics.items():
                losses.setdefault(k, []).append(v)
            if eval_iter is not None:
                try:
                    eval_batch = next(eval_iter)
                except StopIteration:
                    eval_iter = iter(dataset_to_dataloader(eval_dataset, eval_batch_size or batch_size, cur_epoch=epoch, num_workers=num_workers))
                    eval_batch = next(eval_iter)
                model.eval()
                with torch.no_grad(), autocast:
                    eval_loss, eval_metrics = model.compute_loss(to_cuda(eval_batch))
                model.train()
                losses.setdefault("eval_total", []).append(eval_loss.item())
                for k, v in eval_metrics.items():
                    losses.setdefault(f"eval_{k}", []).append(v)
            losses.setdefault("lr", []).append(optimizer.param_groups[0]["lr"])
            log_batch(start_time, epoch, batch_idx, None, batches_per_epoch, losses, global_step=batch_idx, max_steps=max_batches)

        stop = False
        if eval_fn is not None and batch_idx % batches_per_eval == 0:
            model.eval()
            metrics, stop = eval_fn(model)
            model.train()
            for k, v in metrics.items():
                losses.setdefault(k, []).append(v)
            print(f"[eval @ batch {batch_idx}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()) + (" -> stop" if stop else ""))

        if batch_idx % batches_per_save == 0 or stop or batch_idx >= max_batches:
            save_checkpoint(save_path, model, optimizer, scheduler, 0, batch_idx, losses)
            losses = {k: [] for k in losses}
        if stop:
            break

    return batch_idx
