import os
import argparse
import torch
import wandb

from dataloaders.tiny_shakespeare import get_dataloaders
from models.gpt import GPT
from tokenizers.character import CharacterTokenizer
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

    # training hyperparameters
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
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

    tokenizer = CharacterTokenizer()
    train_loader, val_loader = get_dataloaders(cfg.context_length, cfg.batch_size, tokenizer, data_fraction=cfg.data_fraction)
    vocab_size = train_loader.dataset.vocab_size
    print(f"Vocab size: {vocab_size}")
    model = GPT(vocab_size, cfg.context_length, cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.depth)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss = torch.nn.CrossEntropyLoss()

    wandb.summary["num_params"] = sum(p.numel() for p in model.parameters())
    wandb.summary["optimizer"] = type(optimizer).__name__
    wandb.summary["loss"] = type(loss).__name__

    wandb.watch(model, log="gradients", log_freq=100)

    def save_weights(model_to_save, epoch):
        os.makedirs("weights", exist_ok=True)
        weights_path = f"weights/model_epoch_{epoch}.pt"
        torch.save(model_to_save.state_dict(), weights_path)
        run.save(weights_path)

    trainer = Trainer(model, train_loader, val_loader, optimizer, loss, wandb_run=run, grad_accum_steps=cfg.grad_accum_steps)
    trainer.fit(cfg.n_epochs, epoch_callback=save_weights)

    run.finish()

if __name__ == "__main__":
    main()