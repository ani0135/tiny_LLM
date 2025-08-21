# config.py

class Config:
    # vocab_size = None          # vocab_size will be set dynamically after tokenizer is created    
    max_seq_len = 1         # Maximum context length
    embedding_dim = 512       # Hidden size (reduced to fit within 50M parameters)
    num_heads = 8             # Attention heads
    num_layers = 6            # Transformer blocks
    dropout = 0.1             # Dropout rate
    

config = Config()
