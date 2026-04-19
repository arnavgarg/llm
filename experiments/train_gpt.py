import os
import torch
import wandb

from dataloaders.tiny_shakespeare import get_dataloaders
from models.gpt import GPT
from tokenizers.character import CharacterTokenizer
from training.trainer import Trainer

config = {
    # network hyperparameters
    "context_length": 512,
    "d_model": 32,
    "d_ff": 128,
    "num_heads": 2,
    "depth": 2,
    
    # training hyperparameters
    "batch_size": 8,
    "n_epochs": 2,
    "lr": 1e-3,
    "data_fraction": 0.01,  
}

def main():
    run = wandb.init(project="llm", name="train_gpt", config=config)
    cfg = run.config

    tokenizer = CharacterTokenizer()
    train_loader, val_loader = get_dataloaders(cfg.context_length, cfg.batch_size, tokenizer, data_fraction=cfg.data_fraction)
    vocab_size = train_loader.dataset.vocab_size
    print(f"Vocab size: {vocab_size}")
    model = GPT(vocab_size, cfg.context_length, cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.depth)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss = torch.nn.CrossEntropyLoss()

    wandb.watch(model, log="gradients", log_freq=100)

    trainer = Trainer(model, train_loader, val_loader, optimizer, loss, wandb_run=run)
    trainer.fit(cfg.n_epochs)

    # save weights
    os.makedirs("weights", exist_ok=True)
    weights_path = "weights/model.pt"
    torch.save(model.state_dict(), weights_path)
    run.save(weights_path)

    run.finish()

if __name__ == "__main__":
    main()