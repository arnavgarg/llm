import os
import argparse
import torch
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from dataloaders.tiny_shakespeare import get_dataloaders
from models.gpt import GPT
from tokenizers.character import CharacterTokenizer
from tokenizers.tiktoken import TiktokenTokenizer
from training.trainer import Trainer

def get_argparser():
    parser = argparse.ArgumentParser(description="Train GPT model")
    
    # wandb configuration
    parser.add_argument("--run-name", type=str, default="train_gpt", help="Name for the wandb run")

    # network hyperparameters
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--depth", type=int, default=2)

    # tokenizer
    parser.add_argument("--tokenizer", type=str, default="character", choices=["character", "tiktoken"])
    parser.add_argument("--tiktoken-encoding", type=str, default="cl100k_base")

    # training hyperparameters
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-fraction", type=float, default=0.01)

    return parser

def main():
    parser = get_argparser()
    args = parser.parse_args()
    config = vars(args)

    run = wandb.init(project="llm", name=args.run_name, config=config)
    cfg = run.config

    if cfg.tokenizer == "tiktoken":
        tokenizer = TiktokenTokenizer(cfg.tiktoken_encoding)
    else:
        tokenizer = CharacterTokenizer()
        
    train_loader, val_loader = get_dataloaders(cfg.context_length, cfg.batch_size, tokenizer, data_fraction=cfg.data_fraction)
    vocab_size = train_loader.dataset.vocab_size
    print(f"Vocab size: {vocab_size}")
    
    model = GPT(vocab_size, cfg.context_length, cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.depth, dropout=cfg.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss = torch.nn.CrossEntropyLoss()

    wandb.summary["num_params"] = sum(p.numel() for p in model.parameters())
    wandb.summary["optimizer"] = type(optimizer).__name__
    wandb.summary["loss"] = type(loss).__name__

    wandb.watch(model, log="gradients", log_freq=100)

    total_steps = cfg.n_epochs * len(train_loader) // cfg.grad_accum_steps
    if cfg.warmup_steps > 0:
        warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=cfg.warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=total_steps - cfg.warmup_steps)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[cfg.warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    def save_weights(model_to_save, epoch):
        os.makedirs("weights", exist_ok=True)
        weights_path = f"weights/model_epoch_{epoch}.pt"
        torch.save(model_to_save.state_dict(), weights_path)
        run.save(weights_path)

    trainer = Trainer(model, train_loader, val_loader, optimizer, loss, scheduler, wandb_run=run, grad_accum_steps=cfg.grad_accum_steps, max_grad_norm=cfg.max_grad_norm)
    trainer.fit(cfg.n_epochs, epoch_callback=save_weights)

    run.finish()

if __name__ == "__main__":
    main()