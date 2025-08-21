# char_data.py

import torch
from torch.utils.data import Dataset

class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.vocab = {ch: i for i, ch in enumerate(chars)}
        self.inv_vocab = {i: ch for ch, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        return [self.vocab[ch] for ch in text]

    def decode(self, indices):
        return ''.join([self.inv_vocab[i] for i in indices])

class CharDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        data = tokenizer.encode(text)
        self.data = torch.tensor(data, dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y
