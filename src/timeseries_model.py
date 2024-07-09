import torch.nn as nn
import torch
from torch.nn import functional as F

from src.transformer import Block


class TimeSeriesTransformerModel(nn.Module):
    def __init__(self, n_embd, ctx_len, dropout, num_classes):
        super().__init__()
        self.input_projection = nn.Linear(1, n_embd)  # Project single feature to embedding dimension
        self.position_embedding_table = nn.Embedding(ctx_len, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4, mask=False),
            Block(n_embd, n_head=4, mask=False),
            Block(n_embd, n_head=4, mask=False),
            nn.LayerNorm(n_embd),
        )
        self.classification_head = nn.Linear(n_embd, num_classes)

    def forward(self, x, y=None):
        B, T, D = x.shape  # B is batch size, T is sequence length

        tok_emb = self.input_projection(x)  # B, T, n_embd tensor
        pos_emb = self.position_embedding_table(torch.arange(T, device=x.device))  # T, n_embd
        emb = tok_emb + pos_emb  # B, T, n_embd
        emb = self.blocks(emb)
        emb = emb.mean(dim=1)  # Pooling: mean of the sequence (B, n_embd)
        logits = self.classification_head(emb)  # B, num_classes

        if y is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, y)  # logits is B, num_classes and y is B

        return logits, loss
