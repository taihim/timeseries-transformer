import torch
import torch.nn as nn
from torch.nn import functional as F

from src.transformer import Block


class TransformerModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(ctx_len, n_embd)
    # self.sa_head = Head(n_embd)
    # self.sa_heads = MultiHeadAttention(4, n_embd//4)
    # self.ffwd = FeedForward(n_embd)
    self.blocks = nn.Sequential(
        Block(n_embd, n_head=4),
        Block(n_embd, n_head=4),
        Block(n_embd, n_head=4),
        nn.LayerNorm(n_embd),
    )
    self.lm_head = nn.Linear(n_embd, vocab_size)


  def forward(self, x, y=None):
    B,T = x.shape

    tok_emb = self.token_embedding_table(x) # B,T,n_embd tensor
    pos_emb = self.position_embedding_table(torch.arange(T)) # T, n_embd
    emb = tok_emb + pos_emb # B,T, n_embd
    # emb = self.sa_head(emb)
    # emb = self.sa_heads(emb)
    # emb = self.ffwd(emb)
    emb = self.blocks(emb)
    logits = self.lm_head(emb) # B,T,vocab_size (4,8,32 in this case)

    if y is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      y = y.view(B*T)
      loss = F.cross_entropy(logits, y) # logits is B,T,C, y is B,T,1
    return logits, loss

  def generate(self, idx, max_new_tokens):
    # idx is a B,T array of indices in the current context
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -ctx_len:]
      logits, _ = self(idx_cond)
      # get only last time step
      logits = logits[:, -1, :] # (B, C)
      probs = F.softmax(logits, dim=1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx