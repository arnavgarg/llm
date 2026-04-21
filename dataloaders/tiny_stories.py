import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

from tokenizers import Tokenizer

_SPLIT_SIZES = {"train": 2_119_719, "validation": 21_990}


class TinyStoriesDataset(IterableDataset):
    def __init__(
        self,
        split: str,
        context_length: int,
        tokenizer: Tokenizer,
        max_samples: int | None = None,
    ):
        assert split in ("train", "val")
        self.hf_split = "validation" if split == "val" else "train"
        self.context_length = context_length
        self.tokenizer = tokenizer
        self._max_samples = max_samples

    def __iter__(self):
        dataset = load_dataset(
            "roneneldan/TinyStories",
            split=self.hf_split,
            streaming=True,
            trust_remote_code=False,
        )

        buffer: list[int] = []
        count = 0
        for example in dataset:
            tokens = self.tokenizer.encode(example["text"])
            buffer.extend(tokens)
            while len(buffer) >= self.context_length + 1:
                chunk = buffer[: self.context_length + 1]
                buffer = buffer[self.context_length + 1 :]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y
                count += 1
                if self._max_samples is not None and count >= self._max_samples:
                    return

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


def get_dataloaders(
    context_length: int,
    batch_size: int,
    tokenizer: Tokenizer,
    val_steps: int | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    train_ds = TinyStoriesDataset(
        "train", context_length, tokenizer,
        max_samples=None,
    )
    val_ds = TinyStoriesDataset(
        "val", context_length, tokenizer,
        max_samples=val_steps * batch_size if val_steps is not None else None,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader
