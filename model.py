# model.py

import torch
import torch.nn as nn
from model_parts import TokenAndPositionalEmbedding, TransformerBlock
from config import config

class SmallLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = TokenAndPositionalEmbedding(config)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embedding_dim)
        self.head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)  # Output logits

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits
