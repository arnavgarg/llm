import os
import urllib.request

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_CACHE = os.path.join(os.path.dirname(__file__), "..", "datasets", "tiny-shakespeare", "input.txt")


def _download() -> str:
    if not os.path.exists(DATA_CACHE):
        os.makedirs(os.path.dirname(DATA_CACHE), exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_CACHE)
    with open(DATA_CACHE, "r", encoding="utf-8") as f:
        return f.read()


class TinyShakespeareDataset(Dataset):
    def __init__(
        self,
        split: str,
        context_length: int,
        tokenizer: Tokenizer,
        val_fraction: float = 0.1,
        data_fraction: float = 1.0,
    ):
        assert split in ("train", "val")

        text = _download()

        self.tokenizer = tokenizer
        self.tokenizer.fit(text)
        ids = self.tokenizer.encode(text)

        if data_fraction < 1.0:
            ids = ids[: int(len(ids) * data_fraction)]

        split_idx = int(len(ids) * (1 - val_fraction))
        self.ids = ids[:split_idx] if split == "train" else ids[split_idx:]
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.ids) - self.context_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.ids[idx : idx + self.context_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


def get_dataloaders(
    context_length: int,
    batch_size: int,
    tokenizer: Tokenizer,
    val_fraction: float = 0.1,
    data_fraction: float = 1.0,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    train_ds = TinyShakespeareDataset(
        "train", context_length, tokenizer, val_fraction, data_fraction
    )
    val_ds = TinyShakespeareDataset(
        "val", context_length, tokenizer, val_fraction, data_fraction
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader
