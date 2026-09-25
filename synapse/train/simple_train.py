import torch
import synapse.train.train_helpers as train_helpers
import synapse.train.data_to_loaders as data_to_loaders
import time


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
):
    # scheduler: optional torch lr scheduler, stepped once per optimizer step
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
    for cur_epoch in range(epochs):
        train_dataloader = data_to_loaders.dataset_to_dataloader(dataset, batch_size, cur_epoch)
        for cur_batch_num, cur_batch_data in enumerate(train_dataloader):
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

                train_helpers.log_batch(start_time, cur_epoch, cur_batch_num, epochs, len(train_dataloader), losses, global_step=global_step, max_steps=max_steps)
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
