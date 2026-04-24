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
        early_stopping_patience: Optional[int] = 5,
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
        self.early_stopping_patience = early_stopping_patience
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_ckpt_path = None
        self.best_ckpt_label = None
        self._patience_counter = 0
        self._stop_early = False

    def _run_val(self) -> float:
        self.model.train(False)
        total_loss, num_batches = 0.0, 0
        with torch.set_grad_enabled(False):
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    logits = self.model(x)
                    B, T, V = logits.shape
                    total_loss += self.loss_fn(logits.view(B * T, V), y.view(B * T)).item()
                num_batches += 1
        self.model.train(True)
        return total_loss / max(num_batches, 1)

    def _do_optimizer_step(self) -> float:
        grad_norm = clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return grad_norm.item()

    def _log_optimizer_metrics(self, loss_val: float, grad_norm: float) -> None:
        if self.wandb_run is not None:
            self.wandb_run.log({
                "train/batch_loss": loss_val,
                "train/grad_norm": grad_norm,
                "train/lr": self.scheduler.get_last_lr()[0],
            }, step=self.global_step)

    def _check_val_checkpoint(self, label: str) -> float:
        val_loss = self._run_val()
        os.makedirs("weights", exist_ok=True)
        ckpt_path = f"weights/ckpt_{label}.pt"
        torch.save(self.model.state_dict(), ckpt_path)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_ckpt_path = ckpt_path
            self.best_ckpt_label = label
            self._patience_counter = 0
        elif self.early_stopping_patience is not None:
            self._patience_counter += 1
            if self._patience_counter >= self.early_stopping_patience:
                print(f"Early stopping at {label}")
                self._stop_early = True
        return val_loss

    def _save_best_artifact(self) -> None:
        if self.best_ckpt_path is not None and self.wandb_run is not None:
            artifact = wandb.Artifact(f"model_best_{self.best_ckpt_label}", type="model")
            artifact.add_file(self.best_ckpt_path, name=f"model_best_{self.best_ckpt_label}.pt")
            self.wandb_run.log_artifact(artifact)

    def fit(self, *args, **kwargs):
        raise NotImplementedError


class EpochTrainer(Trainer):
    def fit(self, n_epochs: int, epoch_callback: Optional[Callable] = None):
        print(f"Training on {self.device}")
        pbar = tqdm(total=n_epochs, desc="epochs")

        for epoch in range(1, n_epochs + 1):
            if self._stop_early:
                break

            self.model.train(True)
            epoch_losses: list[float] = []
            last_loss_val = 0.0

            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    logits = self.model(x)
                    B, T, V = logits.shape
                    loss = self.loss_fn(logits.view(B * T, V), y.view(B * T))
                (loss / self.grad_accum_steps).backward()
                self.global_step += 1
                last_loss_val = loss.item()
                epoch_losses.append(last_loss_val)

                if self.global_step % self.grad_accum_steps == 0:
                    grad_norm = self._do_optimizer_step()
                    self._log_optimizer_metrics(last_loss_val, grad_norm)

            # flush any remaining accumulated gradients at epoch end
            if self.global_step % self.grad_accum_steps != 0:
                grad_norm = self._do_optimizer_step()
                self._log_optimizer_metrics(last_loss_val, grad_norm)

            train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            val_loss = self._check_val_checkpoint(f"e{epoch}")
            print(f"Epoch {epoch}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

            if self.wandb_run is not None:
                self.wandb_run.log({
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "epoch": epoch,
                }, step=self.global_step)

            if epoch_callback is not None:
                epoch_callback(self.model, epoch)

            pbar.update(1)

        pbar.close()
        self._save_best_artifact()


class MaxStepsTrainer(Trainer):
    def __init__(self, *args, val_interval: int = 500, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_interval = val_interval
        self.cycle = 0

    def _log_cycle_metrics(self, cycle_losses: list[float], cycle: int, skip_val: bool = False) -> None:
        if not cycle_losses:
            return
        train_loss = sum(cycle_losses) / len(cycle_losses)
        log_entry = {"train/loss": train_loss, "cycle": cycle}
        if skip_val:
            print(f"Cycle {cycle}  train_loss={train_loss:.4f}")
        else:
            val_loss = self._run_val()
            log_entry["val/loss"] = val_loss
            print(f"Cycle {cycle}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if self.wandb_run is not None:
            self.wandb_run.log(log_entry, step=self.global_step)

    def fit(self, max_steps: int, epoch_callback: Optional[Callable] = None):
        print(f"Training on {self.device}")
        train_iter = iter(self.train_loader)
        cycle_losses: list[float] = []
        pbar = tqdm(total=max_steps, desc="train")

        while self.global_step < max_steps and not self._stop_early:
            try:
                x, y = next(train_iter)
            except StopIteration:
                self.cycle += 1
                self._log_cycle_metrics(cycle_losses, self.cycle)
                if epoch_callback is not None:
                    epoch_callback(self.model, self.cycle)
                cycle_losses = []
                train_iter = iter(self.train_loader)
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    print("Empty train loader; stopping")
                    break

            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                logits = self.model(x)
                B, T, V = logits.shape
                loss = self.loss_fn(logits.view(B * T, V), y.view(B * T))
            (loss / self.grad_accum_steps).backward()

            self.global_step += 1
            cycle_losses.append(loss.item())
            pbar.update(1)

            is_opt_step = self.global_step % self.grad_accum_steps == 0 or self.global_step == max_steps
            if is_opt_step:
                grad_norm = self._do_optimizer_step()
                self._log_optimizer_metrics(loss.item(), grad_norm)

            if self.global_step % self.val_interval == 0:
                val_loss = self._check_val_checkpoint(f"s{self.global_step}")
                if self.wandb_run is not None:
                    self.wandb_run.log({"val/loss": val_loss}, step=self.global_step)

        pbar.close()

        skip_val = self.global_step % self.val_interval == 0
        self._log_cycle_metrics(cycle_losses, self.cycle, skip_val=skip_val)
        if epoch_callback is not None:
            epoch_callback(self.model, self.cycle)

        self._save_best_artifact()
