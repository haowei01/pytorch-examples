import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# tokenize by the character
char_sets = sorted(list(set(text)))
vocab_size = len(char_sets)
stoi = {char: i for i, char in enumerate(char_sets)}
itos = {i: char for i, char in enumerate(char_sets)}
encode = lambda x: [stoi[char] for char in x]
decode = lambda x: ''.join([itos[i] for i in x])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# split training data vs validation data
input_tensor = torch.tensor(encode(text), dtype=torch.long).to(device)
input_size = len(input_tensor)
split_idx = int(input_size * 0.9)
train_data = input_tensor[: split_idx]
val_data = input_tensor[split_idx:]

torch.manual_seed(1337)

#### configuration of the batch_size, block_size, and the data loader
batch_size = 64
block_size = 256
embed_dim = 256
d_model = 256
nhead = 4
num_layers = 4

#### number of steps per epoch and per validation epoch
step_per_epoch = int(input_size * 0.9 // block_size // batch_size)
step_per_val_epoch = int(input_size * 0.1 // block_size // batch_size)

def get_data(split_method):
    data = train_data if split_method == 'train' else val_data
    idx = torch.randint(0, len(data) - block_size - 1, (batch_size,), dtype=torch.long)
    x = torch.stack([data[i: i + block_size] for i in idx])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad
def estimate_loss(split, model):
    steps = step_per_epoch if split == 'train' else step_per_val_epoch
    losses = []
    model.eval()
    for _ in range(steps):
        x, y = get_data(split)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return np.mean(losses)

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, d_model):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.emb_dim = embed_dim
        self.key_layer = nn.Linear(embed_dim, d_model)
        self.value_layer = nn.Linear(embed_dim, d_model)
        self.query_layer = nn.Linear(embed_dim, d_model)
    
    def forward(self, x):
        B, T, C = x.shape
        keys = self.key_layer(x)      # B, T, d_model
        values = self.value_layer(x)  # B, T, d_model
        queries = self.query_layer(x) # B, T, d_model
        mask = torch.tril(torch.ones(T, T)).to(device)
        wei = queries @ keys.transpose(-2, -1) / (self.d_model ** 0.5)  # B, T, T
        wei = wei.masked_fill(mask == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # B, T, T
        attn = wei @ values  # B, T, d_model
        return attn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, d_model, nhead=None):
        super(MultiHeadSelfAttention, self).__init__()
        self.nhead = nhead
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.multiheads = nn.ModuleList([SelfAttention(embed_dim, d_model // nhead) for _ in range(nhead)])
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.ffwd = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, embed_dim)
        )

    def forward(self, x):
        # shape of X is B, T, C
        x = self.layer_norm_1(x)
        attn = torch.cat([head(x) for head in self.multiheads], dim=-1)
        x = x + attn  # add a residual connection

        # feed forward layer
        x = self.layer_norm_2(x)
        ffwd_output = self.ffwd(x)
        layer_output = x + ffwd_output
        return layer_output


class BigramLanguageModel(nn.Module):
    def __init__(self, embed_dim, d_model, nhead=None, num_layers=None):
        super(BigramLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        # self.self_attention = MultiHeadSelfAttention(embed_dim, d_model, nhead)
        self.multi_layers_self_attention = nn.ModuleList([MultiHeadSelfAttention(embed_dim, d_model, nhead) for _ in range(num_layers)])
        self.final_layer = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x, y=None):
        vocab_emb = self.embedding(x)  # B, T, C
        B, T, C = vocab_emb.shape
        pos = torch.arange(T, dtype=torch.long).to(device).repeat(B, 1)  # B, T
        pos_emb = self.pos_embedding(pos)  # B, T, C
        x = vocab_emb + pos_emb

        for layer in self.multi_layers_self_attention:
            x = layer(x)
        
        logits = self.final_layer(x)  # B, T, vocab_size

        # logits = self.embedding(x)
        if y is None:
            loss = None
        else:
            # B, T, C = logits.shape
            logits = logits.view(B * T, vocab_size)
            targets = y.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, x, n):
        # given X, generate n tokens, x shape B, T
        with torch.no_grad():
            input_seq = x
            for idx in range(n):
                # idx from 0 to n -1, maximum look back is block_size
                logits, _ = self(input_seq[:, -block_size: ])  # logits shape B, T, C
                target_logit = logits[:, -1, :]  # target_logit shape B, C
                target_prob = F.softmax(target_logit, dim=-1)  # target_prob shape B, C
                gen_target = torch.multinomial(target_prob, 1)  # gen_target shape B, 1
                input_seq = torch.concat((input_seq, gen_target), dim=-1)  # inque_seq shape B, T + 1
        return input_seq

model = BigramLanguageModel(embed_dim, d_model, nhead, num_layers).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)



for epoch in range(50):
    model.train()
    for i in range(step_per_epoch):
        x, y = get_data('train')
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 5000 == 0:
            print(f'Epoch {epoch}, step {i}, loss {loss.item()}')
    train_loss = estimate_loss('train', model)
    print(f'Epoch {epoch}, train loss {np.mean(train_loss)}')
    
    val_loss = estimate_loss('val', model)
    print(f'{epoch} validation loss {val_loss}')
        
torch.save(model.state_dict(), f'bigram_lm_epoch_{epoch + 1}.pth')

# try generate the text at the end:
context = torch.zeros(1, 1, dtype=torch.long).to(device)
print(decode(model.generate(context, 500)[0].to('cpu').numpy()))
