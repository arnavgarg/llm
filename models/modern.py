import math

import torch.nn as nn

from attention.causal import CausalSelfAttention
from modules.activations.swiglu import SwiGLU


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.self_attn = CausalSelfAttention(d_model, num_heads, dropout=dropout)
        self.ffn = SwiGLU(d_model, d_ff)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
    
    def forward(self, x):
        # use pre-norming and residual connections
        x = x + self.self_attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model, dropout=0.0):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.token_embedding(x))


class ModernModel(nn.Module):
    def __init__(self, vocab_size, d_context, d_model, num_heads, d_ff, num_layers, dropout=0.0):
        super(ModernModel, self).__init__()
        self.transformer = nn.ModuleDict({
            'embedding': EmbeddingLayer(vocab_size, d_model, dropout=dropout),
            'decoder_layers': nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout=dropout) for _ in range(num_layers)]),
        })
        self.norm = nn.RMSNorm(d_model)
        self.output_linear = nn.Linear(d_model, vocab_size)
        # weight tying: share the token embedding matrix with the output projection
        # both map between the same spaces (token IDs <-> d_model) so one matrix suffices
        self.output_linear.weight = self.transformer['embedding'].token_embedding.weight

        self.apply(self._init_weights)
        # scale residual-path output projections by 1/sqrt(2 * num_layers) so the
        # variance of activations flowing through the residual stream stays bounded
        residual_std = 0.02 / math.sqrt(2 * num_layers)
        for name, param in self.named_parameters():
            if name.endswith('self_attn.out_linear.weight') or name.endswith('ffn.w3.weight'):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        transformer_output = self.transformer['embedding'](x)
        for layer in self.transformer['decoder_layers']:
            transformer_output = layer(transformer_output)
        
        logits = self.output_linear(self.norm(transformer_output))

        return logits

