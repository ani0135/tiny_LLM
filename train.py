# train.py

import torch
from torch.utils.data import DataLoader
from config import config
from char_data import CharTokenizer, CharDataset
from model import SmallLLM
import os

def save_checkpoint(model, optimizer, epoch, path="checkpoint.pt"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_checkpoint(model, optimizer, path="checkpoint.pt"):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open("data/tiny_shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    config.vocab_size = tokenizer.vocab_size

    dataset = CharDataset(text, tokenizer, config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = SmallLLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    start_epoch = 0
    checkpoint_path = "checkpoint.pt"
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path) + 1
        print(f"Loaded checkpoint from epoch {start_epoch-1}")

    model.train()
    for epoch in range(start_epoch, start_epoch + 10):
        total_loss = 0
        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 100 == 0:
                print(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")

        save_checkpoint(model, optimizer, epoch, checkpoint_path)

if __name__ == "__main__":
    train()
