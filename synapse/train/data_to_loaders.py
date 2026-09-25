# takes in bin, numpy data, makes a dataset and then dataloader.  

import os.path
import math

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# Deterministic per-epoch shuffle with fast resume by skipping batches.
# Use pin_memory to move data from RAM to VRAM faster during training.
# num_workers > 0 fetches multiple batches ahead in parapllel
# set to num_workers to (time_to_load / time_to_train). want training and loading rates to be equal  
def dataset_to_dataloader(
    dataset: Dataset,
    batch_size: int,
    cur_epoch: int = 0,
    start_batch: int = 0,
    num_workers: int = 2,
) -> DataLoader:
    seed = 21 # hard code seed = 21
    generator = torch.Generator().manual_seed(seed + cur_epoch)
    # use set generator to ensure the same results when we pause and resume
    indices = torch.randperm(len(dataset), generator=generator).tolist() #type: ignore  

    # fast skip. if normal iterate through skip then it's very slow because it actually loads everything.  
    start_idx = start_batch * batch_size
    if start_idx > 0:
        indices = indices[start_idx:]

    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False, # we already shuffled
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None, # how much each worker loads ahead; not allowed without workers
        pin_memory=True, #  makes moving to vram faster
        drop_last=True, #  drops last incomplete batch.
    )

class NumpySupervisedDataset(Dataset):
    # x: (N, ...) and y: (N, ...) kept in RAM.
    def __init__(self, x: np.ndarray, y: np.ndarray):
        assert len(x) == len(y), f"x and y must have same length, got {len(x)} and {len(y)}"
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        # workers read from the same numpy array; as_tensor handles both ndarray slices and scalars.
        return torch.as_tensor(self.x[idx]), torch.as_tensor(self.y[idx])


class BinUnsupervisedDataset(Dataset):
    # memmaps a flat binary file; shape includes N first, e.g. (N, seq_len) for token ids.
    def __init__(self, path: str, dtype: torch.dtype, shape: tuple[int, ...]):
        assert len(shape) >= 1, "shape must include the sample count dimension"

        self.path = path
        self.dtype = dtype
        self.np_dtype = torch.empty(0, dtype=dtype).numpy().dtype
        self.shape = shape
        self.num_samples = shape[0]
        self.sample_shape = shape[1:]
        self.sample_numel = math.prod(self.sample_shape)
        self.sample_bytes = self.sample_numel * torch.empty(0, dtype=dtype).element_size()

        expected_size = self.num_samples * self.sample_bytes
        actual_size = os.path.getsize(path)
        assert actual_size == expected_size, (
            f"file size mismatch: {path} is {actual_size} bytes, expected {expected_size} "
            f"for shape {shape} and dtype {dtype}"
        )

        # opened lazily so each dataloader worker gets its own memmap handle
        self._mm = None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        assert 0 <= idx < self.num_samples
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=self.np_dtype, mode="r", shape=self.shape)
        # memmap pages are read-only; clone into writable tensor memory
        return torch.from_numpy(self._mm[idx].copy())


class SFTDataset(Dataset):
    # supervised finetuning rows: loss on the answer only. tokenized once, right padded to the longest row, kept in ram.
    # prompt and answer are tokenized separately and concatenated so the token boundary matches generation, where the
    # model sees the prompt tokens and must produce the answer tokens after them.
    def __init__(self, pairs: list[tuple[str, str]], tokenizer, add_eos: bool = True):
        assert len(pairs) > 0
        pad_id = tokenizer.pad_token_id
        assert pad_id is not None, "tokenizer has no pad token"
        eos = [tokenizer.eos_token_id] if add_eos else []
        # [([prompt ids], [answer ids + eos])]
        tokenized = [
            (tokenizer(p, add_special_tokens=False)["input_ids"], tokenizer(a, add_special_tokens=False)["input_ids"] + eos)
            for p, a in pairs
        ]
        max_len = max(len(p) + len(a) for p, a in tokenized)
        n = len(tokenized)
        # (N, L) each
        self.input_ids = torch.full((n, max_len), pad_id, dtype=torch.long)
        self.attention_mask = torch.zeros((n, max_len), dtype=torch.long)
        self.labels = torch.full((n, max_len), -100, dtype=torch.long)
        for i, (p, a) in enumerate(tokenized):
            self.input_ids[i, : len(p) + len(a)] = torch.tensor(p + a)
            self.attention_mask[i, : len(p) + len(a)] = 1
            # -100 = cross_entropy ignore_index: prompt and pad positions get no loss
            self.labels[i, len(p) : len(p) + len(a)] = torch.tensor(a)

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int):
        return {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx], "labels": self.labels[idx]}


class ReplayDataset(Dataset):
    # epoch-wise replay: rows [0, n_main) of `dataset` are the main set and are always included; rows [n_main, len) are the
    # replay pool, of which a fresh ratio * n_main subset is drawn every epoch from (seed, epoch). train() calls set_epoch
    # before building each epoch's loader, so the draw is deterministic and resume-safe. every row must have the same shape,
    # so build main and pool into one dataset (e.g. one SFTDataset over both) before wrapping.
    def __init__(self, dataset: Dataset, n_main: int, ratio: float, seed: int = 21):
        n_pool = len(dataset) - n_main
        self.n_replay = int(round(ratio * n_main))
        assert 0 < n_main < len(dataset), f"n_main {n_main} must leave a non-empty pool in {len(dataset)} rows"
        assert 0 <= self.n_replay <= n_pool, f"ratio {ratio} asks for {self.n_replay} replay rows, pool has {n_pool}"
        self.dataset = dataset
        self.n_main = n_main
        self.n_pool = n_pool
        self.seed = seed
        # [pool row offsets drawn for the current epoch], length n_replay
        self.chosen = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        g = torch.Generator().manual_seed(self.seed + epoch)
        self.chosen = torch.randperm(self.n_pool, generator=g)[: self.n_replay].tolist()

    def __len__(self) -> int:
        return self.n_main + self.n_replay

    def __getitem__(self, idx: int):
        if idx < self.n_main:
            return self.dataset[idx]
        return self.dataset[self.n_main + self.chosen[idx - self.n_main]]
