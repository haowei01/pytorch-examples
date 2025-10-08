import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super(SelfAttention, self).__init__()
        self.key = nn.Linear(embed_dim, head_dim)
        self.query = nn.Linear(embed_dim, head_dim)
        self.value = nn.Linear(embed_dim, head_dim)
        self.scale = head_dim ** -0.5  # scaling factor for dot product attention
        self.attn_weights = None  # to store attention weights for visualization

    def forward(self, x):
        B, T, C = x.shape  # Batch size, Time steps, Channels (embedding dimension)
        k = self.key(x)    # (B, T, head_dim)
        q = self.query(x)  # (B, T, head_dim)
        v = self.value(x)  # (B, T, head_dim)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, T, T)
        self.attn_weights = F.softmax(attn_weights, dim=-1)  # (B, T, T)

        attn = torch.matmul(self.attn_weights, v)  # (B, T, head_dim)
        return attn


class MultiHeadSelfAttention(nn.Module):
    # Layer Norm inside the residual block: https://arxiv.org/pdf/2002.04745
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([SelfAttention(embed_dim, self.head_dim) for _ in range(num_heads)])
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.dropout_1 = nn.Dropout(0.1)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.dropout_2 = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape
        x_norm = self.layer_norm1(x)
        
        # Concatenate outputs from all attention heads
        attn_outputs = [head(x_norm) for head in self.attention_heads]
        attn_concat = torch.cat(attn_outputs, dim=-1)  # (B, T, embed_dim)

        attn_out = self.linear(attn_concat)  # (B, T, embed_dim)
        attn_out = self.dropout_1(attn_out)
        x = x + attn_out  # Residual connection

        x_norm2 = self.layer_norm2(x)
        ffn_out = self.ffn(x_norm2)  # (B, T, embed_dim)
        ffn_out = self.dropout_2(ffn_out)
        out = x + ffn_out  # Residual connection
        return out


class Encoders(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads):
        super(Encoders, self).__init__()
        self.layers = nn.ModuleList([MultiHeadSelfAttention(embed_dim, num_heads) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super(CrossAttention, self).__init__()
        self.key = nn.Linear(embed_dim, head_dim)
        self.query = nn.Linear(embed_dim, head_dim)
        self.value = nn.Linear(embed_dim, head_dim)
        self.scale = head_dim ** -0.5  # scaling factor for dot product attention
        self.attn_weights = None  # to store attention weights for visualization

    def forward(self, x, encoder_out):
        B, T_encoder, C = encoder_out.shape
        B_x, T, C_x = x.shape  # Batch size, Time steps, Channels (embedding dimension)
        q = self.query(x)  # (B, T, head_dim)
        k = self.key(encoder_out)    # (B, T_encoder, head_dim)
        v = self.value(encoder_out)  # (B, T_encoder, head_dim)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, T, T_encoder)
        self.attn_weights = F.softmax(attn_weights, dim=-1)  # (B, T, T_encoder)

        attn = torch.matmul(self.attn_weights, v)  # (B, T, head_dim)
        return attn


class MaskedSelfAttention(nn.Module):
    # layer norm inside the residual block: https://arxiv.org/pdf/2002.04745
    def __init__(self, embed_dim, head_dim):
        super(MaskedSelfAttention, self).__init__()
        self.key = nn.Linear(embed_dim, head_dim)
        self.query = nn.Linear(embed_dim, head_dim)
        self.value = nn.Linear(embed_dim, head_dim)
        self.scale = head_dim ** -0.5  # scaling factor for dot product attention
        self.attn_weights = None  # to store attention weights for visualization

    def forward(self, x):
        k = self.key(x)    # (B, T, head_dim)
        q = self.query(x)  # (B, T, head_dim)
        v = self.value(x)  # (B, T, head_dim)

        wei = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, T, T)
        mask = torch.tril(torch.ones_like(wei))  # (B, T, T)
        wei = wei.masked_fill(mask == 0, float('-inf'))

        self.attn_weights = F.softmax(wei, dim=-1)  # (B, T, T)

        attn = torch.matmul(self.attn_weights, v)  # (B, T, head_dim)
        return attn

    
class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(DecoderLayer, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.masked_self_attentions = nn.ModuleList([MaskedSelfAttention(embed_dim, self.head_dim) for _ in range(num_heads)])
        self.linear_1 = nn.Linear(embed_dim, embed_dim)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.dropout_1 = nn.Dropout(0.1)

        self.cross_attentions = nn.ModuleList([CrossAttention(embed_dim, self.head_dim) for _ in range(num_heads)])
        self.linear_2 = nn.Linear(embed_dim, embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.dropout_2 = nn.Dropout(0.1)

        self.layer_norm_3 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.dropout_3 = nn.Dropout(0.1)
    
    def forward(self, x, encoder_out):
        B, T, C = x.shape
        x_norm_1 = self.layer_norm_1(x)

        self_atten_outputs = torch.cat([head(x_norm_1) for head in self.masked_self_attentions], dim=-1)
        self_atten_outputs = self.linear_1(self_atten_outputs)
        self_atten_outputs = self.dropout_1(self_atten_outputs)
        x = x + self_atten_outputs

        x_norm_2 = self.layer_norm_2(x)
        cross_attention_outputs = torch.cat([head(x_norm_2, encoder_out) for head in self.cross_attentions], dim=-1)
        cross_attention_outputs = self.dropout_2(self.linear_2(cross_attention_outputs))
        x = x + cross_attention_outputs

        x_norm_3 = self.layer_norm_3(x)
        ff_out = self.ffn(x_norm_3)
        ff_out = self.dropout_3(ff_out)
        x = x + ff_out
        return x


class Decoders(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(Decoders, self).__init__()
        self.decoders = nn.ModuleList([DecoderLayer(embed_dim, num_heads) for _ in range(num_layers)])
    
    def forward(self, x, encoder_out):
        for decoder in self.decoders:
            x = decoder(x, encoder_out)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # sin(pos / (10000 ** (2i/embed_dim)))
        # cos(pos / (10000 ** (2i/embed_dim)))
        depth = embed_dim / 2
        depths = torch.arange(0, depth, 1).float() / depth
        angle_rates = 1 / (10000 ** depths).unsqueeze(0)  # (1, depth)

        positions = torch.arange(0, max_len).float().unsqueeze(1) # (max_len, 1)

        angle_rads = positions * angle_rates  # (max_len, depth)

        pe = torch.concat((torch.sin(angle_rads), torch.cos(angle_rads)), dim=-1)  # (max_len, embed_dim)
        self.register_buffer('pe', pe)  # (max_len, embed_dim)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return x


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, padding_idx=None):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_dim = embed_dim

    def forward(self, x):
        return self.embedding(x) * (self.embed_dim ** 0.5)  # scale the embedding by sqrt(embed_dim)


class TranslationTransformer(nn.Module):
    def __init__(
            self, src_vocab_size, tgt_vocab_size, embed_dim=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, max_len=5000,
            padding_idx=None,
            # for using tokenizer that shares the same vocab for source and target, can use the same embedding layer
            share_word_embedding=False,
    ):
        super(TranslationTransformer, self).__init__()
        self.src_embedding = WordEmbedding(src_vocab_size, embed_dim, padding_idx)
        if share_word_embedding:
            assert src_vocab_size == tgt_vocab_size, "When sharing word embedding, source and target vocab sizes must be the same"
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = WordEmbedding(tgt_vocab_size, embed_dim, padding_idx)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        self.encoder = Encoders(num_encoder_layers, embed_dim, num_heads)
        self.decoder = Decoders(embed_dim, num_heads, num_decoder_layers)
        self.final_linear = nn.Linear(embed_dim, tgt_vocab_size)
    
    def forward(self, src, tgt):
        # src: (B, T_src), tgt: (B, T_tgt)
        src_emb = self.src_embedding(src)  # (B, T_src, embed_dim)
        # positional embedding add the residual value
        src_emb = self.positional_encoding(src_emb)  # (B, T_src, embed_dim)
        encoder_out = self.encoder(src_emb)  # (B, T_src, embed_dim)

        tgt_emb = self.tgt_embedding(tgt)  # (B, T_tgt, embed_dim)
        tgt_emb = self.positional_encoding(tgt_emb)  # (B, T_tgt, embed_dim)
        decoder_out = self.decoder(tgt_emb, encoder_out)  # (B, T_tgt, embed_dim)

        out = self.final_linear(decoder_out)  # (B, T_tgt, tgt_vocab_size)
        return out
