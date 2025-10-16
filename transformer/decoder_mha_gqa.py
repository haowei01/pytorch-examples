import torch
import torch.nn as nn

# https://arxiv.org/pdf/1911.02150
# Fast Transformer Decoding: One Write-Head is All You Need


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.head_dim = d_model // nhead
        self.scale_factor = self.head_dim ** 0.5
        self.nhead = nhead

        self.w_q = nn.Linear(d_model, d_model)  # projection from d_model to nhead, d_k
        self.w_k = nn.Linear(d_model, d_model)  # projection from d_model to nhead, d_k
        self.w_v = nn.Linear(d_model, d_model)  # projection from d_model to nhead, d_k
        self.w_o = nn.Linear(d_model, d_model)  # projection from nhead, d_k to d_model

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, sequence):
        # query: B, d_model
        # sequence: B, T, d_model
        B, T, C = sequence.shape
        q_projected = self.w_q(query)  # B, d_model
        k_projected = self.w_k(sequence)  # B, T, d_model
        v_projected = self.w_v(sequence)  # B, T, d_model

        # now split into multiple heads
        q = q_projected.view(B, self.nhead, self.head_dim).unsqueeze(2)        # B, nhead, 1 head_dim
        k = k_projected.view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # B, nhead, T, head_dim
        v = v_projected.view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # B, nhead, T, head_dim
        attn_weights = q @ k.transpose(-2, -1) / self.scale_factor  # B, nhead, 1, T
        attn_weights = torch.softmax(attn_weights, dim=-1)  # B, nhead, 1, T
        attn_weights = self.dropout(attn_weights)
        attn_output = attn_weights @ v  # B, nhead, 1, head_dim
        attn_output = attn_output.unsqueeze(2).view(B, C)  # B, d_model
        output = self.w_o(attn_output)  # B, d_model
        return output


class MultiHeadAttentionWithKVCache(nn.Module):
    """For MultiHeadAttention with KV Cache, the sequence has been processed before, so we store the processed K, V
    in [Batch, num_heads, seq_len, head_dim] format, and only need to process the new query.
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadAttentionWithKVCache, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.head_dim = d_model // nhead
        self.scale_factor = self.head_dim ** 0.5
        self.nhead = nhead

        self.w_q = nn.Linear(d_model, d_model)  # projection from d_model to nhead, d_k
        self.w_k = nn.Linear(d_model, d_model)  # projection from d_model to nhead, d_k
        self.w_v = nn.Linear(d_model, d_model)  # projection from d_model to nhead, d_k
        self.w_o = nn.Linear(d_model, d_model)  # projection from nhead, d_k to d_model

        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence):
        # sequence: B, T, d_model
        # y is not used, just for compatibility with nn.MultiheadAttention
        B, T, C = sequence.shape
        q_projected = self.w_q(sequence)  # B, T, d_model
        k_projected = self.w_k(sequence)  # B, T, d_model
        v_projected = self.w_v(sequence)  # B, T, d_model
        # now split into multiple heads
        q = q_projected.view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # B, nhead, T, head_dim
        k = k_projected.view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # B, nhead, T, head_dim
        v = v_projected.view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # B, nhead, T, head_dim
        attn_weights = q @ k.transpose(-2, -1) / self.scale_factor  # B, nhead, T, T
        tril_mask = torch.tril(torch.ones(T, T)).to(sequence.device)
        attn_weights = attn_weights.masked_fill(tril_mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1)  # B, nhead, T, T
        # TODO: add dropout after debugging
        # attn_weights = self.dropout(attn_weights)
        attn_output = attn_weights @ v  # B, nhead, T, head_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)  # B, T, d_model
        output = self.w_o(attn_output)  # B, T, d_model
        return output

    def generate(self, query, key_cache=None, value_cache=None):
        # query: B, d_model, to generate the total sequence of length T
        # key_cache: B, nhead, T - 1, head_dim
        # value_cache: B, nhead, T - 1, head_dim
        B, C = query.shape
        q_projected = self.w_q(query)  # B, d_model

        # now split into multiple heads
        q = q_projected.view(B, self.nhead, self.head_dim).unsqueeze(2)        # B, nhead, 1 head_dim
        query_k = self.w_k(query).view(B, self.nhead, 1, self.head_dim)  # B, nhead, 1, head_dim
        query_v = self.w_v(query).view(B, self.nhead, 1, self.head_dim)  # B, nhead, 1, head_dim
        if key_cache is not None and value_cache is not None:
            key_cache = torch.cat([key_cache, query_k], dim=2)      # B, nhead, T, head_dim
            value_cache = torch.cat([value_cache, query_v], dim=2)  # B, nhead, T, head_dim
        else:
            key_cache = query_k
            value_cache = query_v

        attn_weight = q @ key_cache.transpose(-2, -1) / self.scale_factor  # B, nhead, 1, T
        attn_weight = torch.softmax(attn_weight, dim=-1)  # B, nhead, 1, T
        # TODO: add dropout after debugging
        # attn_weight = self.dropout(attn_weight)
        attn_output = attn_weight @ value_cache  # B, nhead, 1, head_dim
        attn_output = attn_output.unsqueeze(2).view(B, C)  # B, d_model
        output = self.w_o(attn_output)  # B, d_model
        return output, key_cache, value_cache


class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiQueryAttention, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.head_dim = d_model // nhead
        self.scale_factor = self.head_dim ** 0.5
        self.nhead = nhead

        self.w_q = nn.Linear(d_model, d_model)  # projection from d_model to nhead, d_k
        self.w_k = nn.Linear(d_model, self.head_dim)  # projection from d_model to d_k
        self.w_v = nn.Linear(d_model, self.head_dim)  # projection from d_model to d_k
        self.w_o = nn.Linear(d_model, d_model)  # projection from nhead, d_k to d_model

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, sequence):
        # query: B, d_model
        # sequence: B, T, d_model
        B, T, C = sequence.shape
        q_projected = self.w_q(query)  # B, d_model
        k_projected = self.w_k(sequence)  # B, T, head_dim
        v_projected = self.w_v(sequence)  # B, T, head_dim

        # now split into multiple heads
        q = q_projected.view(B, self.nhead, self.head_dim).unsqueeze(2)        # B, nhead, 1 head_dim
        k = k_projected.view(B, T, 1, self.head_dim).transpose(1, 2)           # B, 1, T, head_dim
        v = v_projected.view(B, T, 1, self.head_dim).transpose(1, 2)           # B, 1, T, head_dim
        attn_weights = q @ k.transpose(-2, -1) / self.scale_factor  # B, nhead, 1, T
        attn_weights = torch.softmax(attn_weights, dim=-1)  # B, nhead, 1, T
        attn_weights = self.dropout(attn_weights)
        attn_output = attn_weights @ v  # B, nhead, 1, head_dim
        attn_output = attn_output.unsqueeze(2).view(B, C)  # B, d_model
        output = self.w_o(attn_output)  # B, d_model
        return output