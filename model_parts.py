# model_parts.py (append this to a new file)

import torch
import torch.nn as nn

class TokenAndPositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, config.max_seq_len, config.embedding_dim))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T = x.size()
        token_emb = self.token_embedding(x)                     # [B, T, C]
        pos_emb = self.pos_embedding[:, :T, :]                  # [1, T, C]
        return self.dropout(token_emb + pos_emb)

# model_parts.py (continue in same file)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embedding_dim % config.num_heads == 0

        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.out_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

        # Causal mask - buffer so it's not trainable
        self.register_buffer("mask", torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.qkv_proj(x)                      # [B, T, 3C]
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                 # Each: [B, T, H, D]
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]  # Each: [B, H, T, D]

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, T, T]
        attn_weights = attn_weights.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = attn_probs @ v                          # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        return self.out_proj(out)
# model_parts.py (continue in same file)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.embedding_dim
        self.net = nn.Sequential(
            nn.Linear(config.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

# model_parts.py (continue in same file)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.embedding_dim)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # Residual connection around attention
        x = x + self.ff(self.ln2(x))   # Residual connection around feedforward
        return x
