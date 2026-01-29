import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache:
    def __init__(self, num_layers, batch_size, num_heads, seq_len, head_dim, append_size=64):
        # TODO: pre-allocate the tensor
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None
        self.pos = 0
        self.append_size = append_size  # should be 2**n

    def get_pos(self):
        return self.pos

    def reset(self):
        self.pos = 0

    def insertKV(self, key, value, layer_idx):
        # key size: B, H, T, Head_Dim
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=key.dtype, device=key.device)
        num_layers, kv_size, batch_size, num_heads, seq_len, head_dim = self.kv_shape
        add_len = key.size(-2)
        t1 = self.pos + add_len
        if t1 > seq_len:
            # need to extend the kv_cache size, add size of append_size
            size_needed = t1 + self.append_size
            size_needed = (t1 + self.append_size - 1) & (~(self.append_size - 1))  # clear lower bits
            additional_cache = torch.empty(num_layers, kv_size, batch_size, num_heads, size_needed - seq_len, head_dim,
                                           dtype=key.dtype, device=key.device)
            self.kv_cache = torch.concat([self.kv_cache, additional_cache], dim=-2)
            self.kv_shape = self.kv_cache.shape
        self.kv_cache[layer_idx, 0, :, :, self.pos: t1] = key
        self.kv_cache[layer_idx, 1, :, :, self.pos: t1] = value
        if layer_idx == self.kv_shape[0] - 1:
            # all layers of cache has been added
            self.pos += add_len
        return self.kv_cache[layer_idx, 0, :, :, :t1], self.kv_cache[layer_idx, 1, :, :, :t1]


class RotatoryEmbedding(nn.Module):
    def __init__(self, base=10000, max_len=8192, head_dim=1024):
        super().__init__()
        self.base = 10000
        base_inv_freq = 1 / torch.pow(base,
                                      torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)  # head_dim / 2
        positions = torch.arange(max_len)
        theta = torch.outer(positions, base_inv_freq)  # [max_len, head_dim // 2]
        cos, sin = torch.cos(theta), torch.sin(theta)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)


# rotatory_embedding = RotatoryEmbedding(base, max_len, head_dim).to("cuda")

# def apply_rotatory_embed(x, start_position=0):
#     B, T, C = x.size(0), x.size(-2), x.size(-1)
#     cos_pos = rotatory_embedding.cos[start_position: start_position + T, :]
#     sin_pos = rotatory_embedding.sin[start_position: start_position + T, :]
#     x1 = x[..., :C // 2]
#     x2 = x[..., C //2 :]
#     x1_rotated = cos_pos * x1 - sin_pos * x2
#     x2_rotated = sin_pos * x1 + cos_pos * x2
#     return torch.cat([x1_rotated, x2_rotated], dim=-1)

def apply_rotatory_emb(x, cos, sin):
    # B, H, T, C
    C = x.size(-1)
    x1 = x[..., :C // 2]
    x2 = x[..., C // 2:]
    x1_rotated = cos * x1 - sin * x2
    x2_rotated = sin * x1 + cos * x2
    return torch.cat([x1_rotated, x2_rotated], dim=-1).to(x.dtype)


def rms_norm(x):
    normalized_shape = (x.size(-1),)
    print(f"rms_norm shape: {normalized_shape}")
    return F.rms_norm(x, normalized_shape)


class SelfAttention(nn.Module):
    def __init__(self, layer_idx, num_query_heads, num_key_heads, dim, head_dim, dropout=0.2):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_query_heads = num_query_heads
        self.num_key_heads = num_key_heads
        self.enable_gqa = num_query_heads != num_key_heads
        self.dim = dim
        assert dim % num_query_heads == 0
        assert num_query_heads % num_key_heads == 0
        assert dim // num_query_heads == head_dim
        self.head_dim = head_dim
        self.q_proj = nn.Linear(dim, dim)  # project from dim to num_head, head_dim
        self.k_proj = nn.Linear(dim, num_key_heads * head_dim)  # project from dim to num_head, head_dim
        self.v_proj = nn.Linear(dim, num_key_heads * head_dim)  # project from dim to num_head, head_dim
        self.o_proj = nn.Linear(dim, dim)  # project from dim to num_head, head_dim

    def forward(self, query, cos_sin, kv_cache: KVCache = None, query_start_position=0):
        """
        :start_position: query start_position, TODO: some logic verification of the start_position
        :cos or sin dimension: [1, seq_pos, 1, head_dim], the
        """
        B, Tq = query.size(0), query.size(1)
        cos, sin = cos_sin
        q = self.q_proj(query).view(B, Tq, self.num_query_heads, self.head_dim)  # B, T, num_query_heads
        k = self.k_proj(query).view(B, Tq, self.num_key_heads, self.head_dim)  # B, T, num_key_heads, head_dim
        v = self.v_proj(query).view(B, Tq, self.num_key_heads, self.head_dim)  # B, T, num_key_heads, head_dim
        # only q and key applies the rotatory change, to reflect q and k's position change
        # cos_sin are passed in based on query's position
        q, k = apply_rotatory_emb(q, cos, sin), apply_rotatory_emb(k, cos, sin)
        q, k = rms_norm(q), rms_norm(k)
        # shape to be B, num_heads, T, head_dim
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if kv_cache is None:
            # during training, KV Cache is not needed
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                  enable_gqa=self.enable_gqa)  # B, num_query_heads, L, head_dim
        else:
            pos_cache = kv_cache.get_pos()
            key_cache, value_cache = kv_cache.insertKV(k, v, self.layer_idx)
            if Tq == 1:
                # this is the common case when generated token is used to generate the next token
                attn = F.scaled_dot_product_attention(q, k, v, enable_gqa=self.enable_gqa)
            else:

                Tk = key_cache.size(2)

                attn_mask = torch.ones(Tq, Tk, dtype=torch.bool, device=query.device)
                attn_mask[-Tq:] = torch.ones(Tq, Tq, dtype=torch.bool, device=query.device).tril()

                attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                                      enable_gqa=self.enable_gqa)  # B, num_query_heads, L, head_dim

        attn = attn.transpose(1, 2).contiguous().view(B, Tq, -1)
        attn = query + self.o_proj(attn)  # B, L, C

        return attn


class DecoderLayer(nn.Module):
    def __init__(self, layer_idx, num_query_heads, num_key_heads, dim, head_dim, dropout=0.2):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = SelfAttention(layer_idx, num_query_heads, num_key_heads, dim, head_dim, dropout)
        self.ffwd1 = nn.Linear(dim, 4 * dim)
        self.ffwd2 = nn.Linear(dim * 4, dim)

    def forward(self, query, kv_cache: KVCache = None):
        attn_input = rms_norm(query)
        attn_value = self.self_attn(attn_input, kv_cache)
        x = query + attn_value

        ffwd_input = rms_norm(x)
        ffwd_output = self.ffwd2(self.ffwd1(ffwd_input))
        x = x + ffwd_output
        return x, key_cache, value_cache