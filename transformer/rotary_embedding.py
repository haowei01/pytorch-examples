import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048):
        super(RotaryEmbedding, self).__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        position = torch.arange(0, max_position_embeddings, dtype=torch.float)
        sinusoid_inp = torch.outer(position, inv_freq)  # (max_position_embeddings, dim/2)
        self.cos_cached = torch.cos(sinusoid_inp)
        self.sin_cached = torch.sin(sinusoid_inp)

    def get_rotary_embedding(self, seq_len, device):
        return (
            self.cos_cached[:seq_len, :].to(device),
            self.sin_cached[:seq_len, :].to(device),
        )

    def apply_rotary_pos_emb(self, x, cos, sin):
        # x: [batch_size, seq_len, dim]
        x1, x2 = x[..., : self.dim // 2], x[..., self.dim // 2 :]
        x_rotated = torch.cat(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1
        )
        return x_rotated

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)
        cos, sin = self.get_rotary_embedding(seq_len, x.device)
        # B, T, C
        return self.apply_rotary_pos_emb(x, cos, sin)
