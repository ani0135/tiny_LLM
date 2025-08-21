# inference.py

import torch
import torch.nn.functional as F
from model import SmallLLM
from config import config
from char_data import CharTokenizer

def generate_text(model, tokenizer, start_text, max_tokens=100, temperature=1.0, device='cpu'):
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(start_text), dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_tokens):
            if input_ids.size(1) > config.max_seq_len:
                input_ids = input_ids[:, -config.max_seq_len:]
            logits = model(input_ids)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0].tolist())

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open("data/tiny_shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    config.vocab_size = tokenizer.vocab_size

    model = SmallLLM(config)
    checkpoint = torch.load("checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    prompt = "HI"
    generated = generate_text(model, tokenizer, prompt, max_tokens=200, temperature=0.8, device=device)
    print("Generated text:\n")
    print(generated)
