import re
import time
from pathlib import Path

import torch
import synapse.train.train_helpers as train_helpers
import synapse.train.data_to_loaders as data_to_loaders


def simple_train(
    model,
    dataset,
    batch_size,
    optimizer,
    epochs,
    save_path,
    batches_per_log,
    batches_per_save,
    eval_dataset = None,
    eval_batch_size = None,
    max_steps = None,
    scheduler = None,
    resume = False,
):
    # scheduler: optional torch lr scheduler, stepped once per optimizer step
    # resume: if save_path holds a checkpoint (epoch_<e>_batch_<b>.pt), load model / optimizer / scheduler from it and
    #         continue at the batch after it (same seeded data order, so no batch is seen twice)
    model.cuda()
    model.train()
    if eval_batch_size is None:
        eval_batch_size = batch_size
    losses = {}
    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = data_to_loaders.dataset_to_dataloader(eval_dataset, eval_batch_size, cur_epoch=0)
    eval_iter = iter(eval_dataloader) if eval_dataloader is not None else None

    global_step = 0
    # (epoch, batch) of the first batch to train
    start_epoch, start_batch = 0, 0
    if resume:
        ckpts = list(Path(save_path).glob("epoch_*_batch_*.pt"))
        # save_checkpoint keeps only the latest one
        assert len(ckpts) <= 1, f"several checkpoints in {save_path}: {ckpts}"
        if ckpts:
            ck = torch.load(ckpts[0], map_location="cuda")
            model.load_state_dict(ck["model_state_dict"])
            optimizer.load_state_dict(ck["optimizer_state_dict"])
            if scheduler is not None:
                scheduler.load_state_dict(ck["scheduler_state_dict"])
            e, b = map(int, re.fullmatch(r"epoch_(\d+)_batch_(\d+)\.pt", ckpts[0].name).groups())
            start_epoch, start_batch = e, b + 1
            # batches per epoch (the dataloader drops the last incomplete batch)
            global_step = e * (len(dataset) // batch_size) + b + 1
            print(f"resumed from {ckpts[0].name}: epoch {e}, next batch {start_batch}, step {global_step}", flush=True)
    for cur_epoch in range(start_epoch, epochs):
        skip = start_batch if cur_epoch == start_epoch else 0
        train_dataloader = data_to_loaders.dataset_to_dataloader(dataset, batch_size, cur_epoch, start_batch=skip)
        # batch numbers stay absolute within the epoch after a resume
        for cur_batch_num, cur_batch_data in enumerate(train_dataloader, start=skip):
            start_time = time.time()
            cur_batch_data = _to_cuda(cur_batch_data)
            optimizer.zero_grad()
            result = model.compute_loss(cur_batch_data)
            if isinstance(result, tuple):
                loss, metrics = result
            else:
                loss = result
                metrics = {}
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            if (cur_batch_num + 1) % batches_per_log == 0:
                for k, v in metrics.items():
                    losses.setdefault(k, []).append(v)
                losses.setdefault("total", []).append(loss.item())

                if eval_iter is not None:
                    try:
                        cur_eval_data = next(eval_iter)
                    except StopIteration:
                        eval_dataloader = data_to_loaders.dataset_to_dataloader(eval_dataset, eval_batch_size, cur_epoch=cur_epoch)  # type: ignore
                        eval_iter = iter(eval_dataloader)
                        cur_eval_data = next(eval_iter)
                    cur_eval_data = _to_cuda(cur_eval_data)
                    model.eval()
                    with torch.no_grad():
                        eval_result = model.compute_loss(cur_eval_data)
                    model.train()
                    if isinstance(eval_result, tuple):
                        _eval_loss, eval_metrics = eval_result
                    else:
                        _eval_loss = eval_result
                        eval_metrics = {}
                    for k, v in eval_metrics.items():
                        losses.setdefault(f"eval_{k}", []).append(v)

                train_helpers.log_batch(start_time, cur_epoch, cur_batch_num, epochs, skip + len(train_dataloader), losses, global_step=global_step, max_steps=max_steps)
            if (cur_batch_num + 1) % batches_per_save == 0:
                train_helpers.save_checkpoint(save_path, model, optimizer, scheduler, cur_epoch, cur_batch_num, losses)
                losses = {k: [] for k in losses}
            if max_steps is not None and global_step >= max_steps:
                return


def simple_eval(model, dataset, batch_size, batches=None):
    dataloader = data_to_loaders.dataset_to_dataloader(dataset, batch_size, cur_epoch=0)
    model.eval()
    all_metrics = {}
    count = 0
    with torch.no_grad():
        for cur_batch_data in dataloader:
            cur_batch_data = _to_cuda(cur_batch_data)
            result = model.compute_loss(cur_batch_data)
            if isinstance(result, tuple):
                _, metrics = result
            else:
                continue
            for k, v in metrics.items():
                all_metrics[k] = all_metrics.get(k, 0.0) + v
            count += 1
            if batches is not None and count >= batches:
                break
    model.train()
    if count == 0:
        return {}
    return {k: v / count for k, v in all_metrics.items()}


def _to_cuda(data):
    if isinstance(data, torch.Tensor):
        return data.to('cuda')
    if isinstance(data, (list, tuple)):
        return type(data)(_to_cuda(d) for d in data)
    if isinstance(data, dict):
        return {k: _to_cuda(v) for k, v in data.items()}
    return data
