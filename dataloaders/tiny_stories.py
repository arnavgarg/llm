import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

from tokenizers import Tokenizer

_SPLIT_SIZES = {"train": 2_119_719, "validation": 21_990}
_AVG_TOKENS_PER_STORY = 130


class TinyStoriesDataset(IterableDataset):
    def __init__(
        self,
        split: str,
        context_length: int,
        tokenizer: Tokenizer,
        steps_per_epoch: int | None = None,
        data_fraction: float = 1.0,
    ):
        assert split in ("train", "val")
        self.hf_split = "validation" if split == "val" else "train"
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.data_fraction = data_fraction

        # Derive steps_per_epoch from data_fraction if not explicitly given
        if steps_per_epoch is not None:
            self._steps_per_epoch = steps_per_epoch
        elif data_fraction < 1.0:
            total_stories = int(_SPLIT_SIZES[self.hf_split] * data_fraction)
            self._steps_per_epoch = max(1, total_stories * _AVG_TOKENS_PER_STORY // context_length)
        else:
            self._steps_per_epoch = None

    def __iter__(self):
        dataset = load_dataset(
            "roneneldan/TinyStories",
            split=self.hf_split,
            streaming=True,
            trust_remote_code=False,
        )
        if self.data_fraction < 1.0:
            n_stories = int(_SPLIT_SIZES[self.hf_split] * self.data_fraction)
            dataset = dataset.take(n_stories)

        buffer: list[int] = []
        steps = 0
        for example in dataset:
            tokens = self.tokenizer.encode(example["text"])
            buffer.extend(tokens)
            while len(buffer) >= self.context_length + 1:
                chunk = buffer[: self.context_length + 1]
                buffer = buffer[self.context_length + 1 :]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y
                steps += 1
                if self._steps_per_epoch is not None and steps >= self._steps_per_epoch:
                    return

    def __len__(self) -> int:
        if self._steps_per_epoch is not None:
            return self._steps_per_epoch
        raise TypeError(
            "TinyStoriesDataset has no fixed length; pass steps_per_epoch or data_fraction < 1.0"
        )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


def get_dataloaders(
    context_length: int,
    batch_size: int,
    tokenizer: Tokenizer,
    steps_per_epoch: int | None = None,
    val_steps: int | None = None,
    data_fraction: float = 1.0,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    # Default: 1000 train steps, 100 val steps when streaming the full dataset
    if steps_per_epoch is None and data_fraction >= 1.0:
        steps_per_epoch = 1000
    if val_steps is None and data_fraction >= 1.0:
        val_steps = 100

    train_ds = TinyStoriesDataset(
        "train", context_length, tokenizer,
        steps_per_epoch=steps_per_epoch * batch_size if steps_per_epoch else None,
        data_fraction=data_fraction,
    )
    val_ds = TinyStoriesDataset(
        "val", context_length, tokenizer,
        steps_per_epoch=val_steps * batch_size if val_steps else None,
        data_fraction=data_fraction,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader
