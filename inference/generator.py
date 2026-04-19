import torch
import torch.nn.functional as F

class TextGenerator:
    def __init__(self, model, tokenizer, context_length: int, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 1.0) -> str:
        """
        Generate text continuation from a prompt.
        """
        # Encode prompt
        ids = self.tokenizer.encode(prompt)
        if not ids:
            # If prompt is empty, start with a random token or a specific token
            # We'll just start with an arbitrary token ID 0 for simplicity if empty
            ids = [0]
            
        x = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0) # (1, T)

        for _ in range(max_new_tokens):
            # Crop to context length if necessary
            x_cond = x if x.size(1) <= self.context_length else x[:, -self.context_length:]

            # Get logits
            logits = self.model(x_cond) # (1, T, vocab_size)
            logits = logits[:, -1, :] # (1, vocab_size)

            # Apply temperature
            if temperature == 0.0:
                # Greedy search
                next_token = torch.argmax(logits, dim=-1, keepdim=True) # (1, 1)
            else:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1) # (1, vocab_size)
                next_token = torch.multinomial(probs, num_samples=1) # (1, 1)

            # Append to sequence
            x = torch.cat((x, next_token), dim=1) # (1, T+1)

        # Decode generated sequence
        generated_ids = x.squeeze(0).tolist()
        return self.tokenizer.decode(generated_ids)
