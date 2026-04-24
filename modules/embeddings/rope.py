import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len=4096, base=10000):
        super(RoPE, self).__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        # theta_i = 1 / base^(2i / head_dim) for i in [0, head_dim/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # precompute cos/sin for all positions up to max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        positions = torch.arange(seq_len, device=self.inv_freq.device).float()
        # outer product: (seq_len, head_dim/2)
        freqs = torch.outer(positions, self.inv_freq)
        # duplicate so shape is (seq_len, head_dim) matching interleaved pairs
        freqs = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", freqs.cos())
        self.register_buffer("sin_cache", freqs.sin())

    def _rotate_half(self, x):
        # split last dim in half and rotate: [-x2, x1]
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k):
        """
        q, k: (batch, num_heads, seq_len, head_dim)
        Returns rotated q and k of same shape.
        """
        seq_len = q.size(2)
        if seq_len > self.cos_cache.size(0):
            self._build_cache(seq_len)

        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin

        return q_rot, k_rot
