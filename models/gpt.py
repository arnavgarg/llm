import torch
import torch.nn as nn

from attention.causal import CausalSelfAttention


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(DecoderLayer, self).__init__()
        self.self_attn = CausalSelfAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # use pre-norming and residual connections
        x = x + self.self_attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_context, d_model):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(d_context, d_model) 
    
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return self.token_embedding(x) + self.position_embedding(positions)


class GPT(nn.Module):
    def __init__(self, vocab_size, d_context, d_model, num_heads, d_ff, num_layers):
        super(GPT, self).__init__()
        self.transformer = nn.ModuleDict({
            'embedding': EmbeddingLayer(vocab_size, d_context, d_model),
            'decoder_layers': nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]),
        })
        self.norm = nn.LayerNorm(d_model)
        self.output_linear = nn.Linear(d_model, vocab_size)
        # weight tying: share the token embedding matrix with the output projection
        # both map between the same spaces (token IDs <-> d_model) so one matrix suffices
        self.output_linear.weight = self.transformer['embedding'].token_embedding.weight
    
    def forward(self, x):
        transformer_output = self.transformer['embedding'](x)
        for layer in self.transformer['decoder_layers']:
            transformer_output = layer(transformer_output)
        
        logits = self.output_linear(self.norm(transformer_output))

        return logits

