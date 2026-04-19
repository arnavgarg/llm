import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import Optional, Callable
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        wandb_run=None,
    ):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.wandb_run = wandb_run
        self.global_step = 0

    def _run_epoch(self, loader: DataLoader, train: bool, epoch: int) -> float:
        self.model.train(train)
        total_loss = 0.0

        desc = f"{'train' if train else 'val'} (epoch {epoch})"
        with torch.set_grad_enabled(train):
            for x, y in tqdm(loader, desc=desc, leave=False):
                x, y = x.to(self.device), y.to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    logits = self.model(x)  # (B, T, vocab_size)
                    B, T, V = logits.shape
                    loss = self.loss_fn(logits.view(B * T, V), y.view(B * T))

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    self.global_step += 1
                    if self.wandb_run is not None:
                        self.wandb_run.log({
                            "train/batch_loss": loss.item(),
                            "train/global_step": self.global_step
                        })

                total_loss += loss.item()

        return total_loss / len(loader)

    def fit(self, num_epochs: int, epoch_callback: Optional[Callable] = None):
        print(f"Training on {self.device}")
        for epoch in range(1, num_epochs + 1):
            train_loss = self._run_epoch(self.train_loader, train=True, epoch=epoch)
            val_loss = self._run_epoch(self.val_loader, train=False, epoch=epoch)
            print(f"Epoch {epoch}/{num_epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
            if self.wandb_run is not None:
                self.wandb_run.log({
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "epoch": epoch
                })
            if epoch_callback is not None:
                epoch_callback(self.model, epoch)
