import argparse
import json
import os
import torch
import wandb

from models.gpt import GPT
from tokenizers.character import CharacterTokenizer
from inference.generator import TextGenerator


PROMPTS = [
    # Character voice
    "HAMLET: To be or not",
    "KING RICHARD: My kingdom for",
    "JULIET: O Romeo, Romeo",
    "IAGO: I am not what",
    # Scene openings
    "Enter ROMEO and JULIET\n\nROMEO:",
    "ACT III. SCENE II.\n\nKING:",
    "FIRST CITIZEN: Before we proceed",
    # Emotional registers
    "ROMEO: I love you no longer\n",
    "KING HENRY: We few, we happy few\n",
    "MACBETH: Is this a dagger which\n",
    "CLOWN: Why, I would not hang\n",
    # Minimal seed
    "DUKE:",
    "Enter",
    "ACT",
    # Mid-sentence
    "OTHELLO: She loved me for the",
    "PROSPERO: I have bedimmed the",
    # Stage directions
    "[Exeunt all but",
    "[Aside] He knows not",
]


def get_argparser():
    parser = argparse.ArgumentParser(description="Run inference on a trained GPT model")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pt file)")
    parser.add_argument("--wandb-run", type=str, default=None, help="wandb run path (entity/project/run_id) to pull weights from")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--output", type=str, default="inference_results.json", help="Output JSON file path")
    return parser


def main():
    parser = get_argparser()
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = CharacterTokenizer()

    weights_path = args.weights
    if args.wandb_run:
        api = wandb.Api()
        run = api.run(args.wandb_run)
        files = [f for f in run.files() if f.name.endswith(".pt")]
        if not files:
            raise ValueError(f"No .pt files found in wandb run {args.wandb_run}")
        files[0].download(replace=True)
        weights_path = files[0].name

    checkpoint = torch.load(weights_path, map_location=device)
    vocab_size = checkpoint.get("vocab_size", len(tokenizer.chars) if hasattr(tokenizer, "chars") else 65)

    model = GPT(vocab_size, args.context_length, args.d_model, args.num_heads, args.d_ff, args.depth)
    model.load_state_dict(checkpoint if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint else checkpoint["model_state_dict"])

    generator = TextGenerator(model, tokenizer, context_length=args.context_length, device=device)

    results = []
    for prompt in PROMPTS:
        print(f"\n{'='*50}\nPROMPT: {prompt!r}\n{'-'*20}")
        output = generator.generate(prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        continuation = output[len(prompt):]
        print(continuation)
        results.append({
            "prompt": prompt,
            "continuation": continuation,
            "full_output": output,
        })

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "config": {
                "weights": weights_path,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
            },
            "results": results,
        }, f, indent=2)

    print(f"\nSaved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
