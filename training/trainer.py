import os
import torch
import torch.nn as nn
import wandb
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from typing import Optional, Callable
from tqdm import tqdm

_EARLY_STOP_PATIENCE = 3


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        scheduler: LRScheduler,
        wandb_run=None,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
        val_interval: int = 500,
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
        self.scheduler = scheduler
        self.wandb_run = wandb_run
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.global_step = 0
        self.val_interval = val_interval
        self.best_val_loss = float('inf')
        self.best_ckpt_path = None
        self.best_ckpt_label = None
        self._patience_counter = 0
        self._stop_early = False

    def _run_val(self) -> float:
        self.model.train(False)
        total_loss = 0.0
        with torch.set_grad_enabled(False):
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    logits = self.model(x)
                    B, T, V = logits.shape
                    total_loss += self.loss_fn(logits.view(B * T, V), y.view(B * T)).item()
        self.model.train(True)
        return total_loss / len(self.val_loader)

    def _run_epoch(self, loader: DataLoader, train: bool, epoch: int) -> float:
        self.model.train(train)
        total_loss = 0.0
        num_batches = 0
        step_in_epoch = 0

        desc = f"{'train' if train else 'val'} (epoch {epoch})"
        with torch.set_grad_enabled(train):
            if train:
                self.optimizer.zero_grad()

            for i, (x, y) in enumerate(tqdm(loader, desc=desc, leave=False)):
                x, y = x.to(self.device), y.to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    logits = self.model(x)  # (B, T, vocab_size)
                    B, T, V = logits.shape
                    loss = self.loss_fn(logits.view(B * T, V), y.view(B * T))

                if train:
                    (loss / self.grad_accum_steps).backward()

                    if (i + 1) % self.grad_accum_steps == 0 or (i + 1) == len(loader):
                        grad_norm = clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        self.global_step += 1
                        step_in_epoch += 1
                        if self.wandb_run is not None:
                            self.wandb_run.log({
                                "train/batch_loss": loss.item(),
                                "train/grad_norm": grad_norm.item(),
                                "train/lr": self.scheduler.get_last_lr()[0],
                                "train/global_step": self.global_step
                            })

                        if self.global_step % self.val_interval == 0:
                            val_loss = self._run_val()
                            if self.wandb_run is not None:
                                self.wandb_run.log({
                                    "val/loss_step": val_loss,
                                    "train/global_step": self.global_step
                                })
                            os.makedirs("weights", exist_ok=True)
                            ckpt_path = f"weights/ckpt_e{epoch}_s{step_in_epoch}.pt"
                            torch.save(self.model.state_dict(), ckpt_path)
                            if val_loss < self.best_val_loss:
                                self.best_val_loss = val_loss
                                self.best_ckpt_path = ckpt_path
                                self.best_ckpt_label = f"{epoch}_{step_in_epoch}"
                                self._patience_counter = 0
                            else:
                                self._patience_counter += 1
                                if self._patience_counter >= _EARLY_STOP_PATIENCE:
                                    self._stop_early = True
                                    total_loss += loss.item()
                                    num_batches += 1
                                    return total_loss / num_batches

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def fit(self, num_epochs: int, epoch_callback: Optional[Callable] = None):
        print(f"Training on {self.device}")
        for epoch in range(1, num_epochs + 1):
            train_loss = self._run_epoch(self.train_loader, train=True, epoch=epoch)
            if self._stop_early:
                print(f"Early stopping at epoch {epoch}")
                if epoch_callback is not None:
                    epoch_callback(self.model, epoch)
                break
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

        if self.best_ckpt_path is not None and self.wandb_run is not None:
            artifact = wandb.Artifact(f"model_best_{self.best_ckpt_label}", type="model")
            artifact.add_file(self.best_ckpt_path, name=f"model_best_{self.best_ckpt_label}.pt")
            self.wandb_run.log_artifact(artifact)
