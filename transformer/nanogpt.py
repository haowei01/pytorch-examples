import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.scale = 1 / math.sqrt(n_embd / n_head)
        self.dropout = dropout
        # can do batch projection
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
        self.o_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.residual_droput = nn.Dropout(dropout)
        self.block_size = block_size
        self.register_buffer('mask', torch.ones((block_size, block_size), dtype=torch.bool).tril())

    def forward(self, x):
        # first do q, k, v projection
        assert len(x.shape) == 3  # TODO: handle if it is not this shape
        B, T, C = x.shape
        q_proj, k_proj, v_proj = self.qkv_proj(x).split(self.n_embd, dim=-1)
        q = q_proj.view(B, T, self.n_head, -1).transpose(1, 2)  # B, n_head, T, head_dim
        k = k_proj.view(B, T, self.n_head, -1).transpose(1, 2)  # B, n_head, T, head_dim
        v = v_proj.view(B, T, self.n_head, -1).transpose(1, 2)  # B, n_head, T, head_dim

        attn_w = q @ k.transpose(-2, -1) * self.scale  # B, n_head, T, T
        attn_w = attn_w.masked_fill(self.mask[:T, :T] == False, float('-inf')).softmax(dim=-1)

        attn_w = self.attn_dropout(attn_w)  # B, n_head, T, T

        attn_v = attn_w @ v
        attn_v = attn_v.transpose(1, 2).contiguous().view(B, T, -1)
        o = self.o_proj(attn_v)
        o = self.residual_droput(o)
        # return attn_w optional for debugging
        return o, attn_w


class MLP(nn.Module):

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return self.dropout(x)


class Block(nn.Module):

    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.causal_attn = CausalSelfAttention(n_embd, n_head, dropout, block_size)
        self.layer_norm2 = nn.LayerNorm(n_embd)
        self.ffwd = MLP(n_embd, dropout)

    def forward(self, x):
        x_norm_1 = self.layer_norm1(x)
        attn_o, attn_w = self.causal_attn(x_norm_1)
        x = x + attn_o
        x_norm_2 = self.layer_norm2(x)
        x = x + self.ffwd(x_norm_2)
        return x


class GPT(nn.Module):

    def __init__(self, vocab_size, block_size, n_layers, n_embd, n_head, dropout):
        super().__init__()
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(vocab_size, n_embd),
            'wpe': nn.Embedding(block_size, n_embd),
            'layers': nn.ModuleList([Block(n_embd, n_head, dropout, block_size) for _ in range(n_layers)])
        })
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, target=None):
        B, T = x.shape
        emb = self.transformer.wte(x)  # B, T, C

        positions = torch.arange(T).unsqueeze(0)  # 1, T
        pos_emb = self.transformer.wpe(positions)  # 1, T, C
        x = emb + pos_emb
        for layer in self.transformer.layers:
            x = layer(x)
        logit = self.lm_head(x)
        if target is None:
            # no loss returned
            return logit, None
        else:
            loss = F.cross_entropy(logit.view(B * T, vocab_size), target.view(B * T))
            return logit, loss

    def generate(self, x, max_tokens=10):
        for _ in range(max_tokens):
            logit, _ = self.forward(x)
            probs = logit[:, -1, :].softmax(dim=-1)  # B, C
            x = torch.cat([x, torch.multinomial(probs, num_samples=1)], dim=1)
        return x


######
# Test Code
######
torch.manual_seed(1337)
n_embd = 8
n_head = 2
dropout = 0.0
block_size = 64
batch = 4
x = torch.rand(batch, 6, n_embd)
# test_causal_self_attn = CausalSelfAttention(n_embd, n_head, dropout, block_size)
# print(test_causal_self_attn.mask[:6, :6])
# o, attn_w = test_causal_self_attn(x)
# print(o)
# print(attn_w)

# test_mlp = MLP(n_embd, dropout)
# x_mlp = test_mlp(x)
# print(x_mlp)

# test_block = Block(n_embd, n_head, dropout, block_size)
# x_block_out = test_block(x)
# print(x_block_out)

vocab_size = 100
sequence = torch.randint(0, vocab_size, (batch, 11))
inputs = sequence[:, :10]
target = sequence[:, 1:].contiguous()
print(sequence)
print(inputs)
print(target)

test_gpt = GPT(vocab_size, block_size, 2, n_embd, n_head, dropout)
# logit, _ = test_gpt(inputs)
# print("test gpt outputs")
# print(logit)
# print(logit.shape)
# print(logit.view(-1, 100))
# target = target.unsqueeze(2).contiguous()
# print(target.shape)
# target = target.view(-1, 1)
# print("test gpt loss")
# logit, loss = test_gpt(inputs, target)
# print(logit)
# print(loss)
optimizer = torch.optim.AdamW(test_gpt.parameters(), lr=1e-3)
for steps in range(1000):
    optimizer.zero_grad()
    logit, loss = test_gpt(inputs, target)
    if steps % 100 == 0:
        print(f"step {steps}, {loss.item()}")
    loss.backward()
    optimizer.step()

generated = test_gpt.generate(inputs[:, :1])
print(f"generated {generated}")
