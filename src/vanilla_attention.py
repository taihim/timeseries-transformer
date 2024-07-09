import torch
import torch.nn as nn
from torch.nn import functional as F


class MaskedHead(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(ctx_len, ctx_len)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)

    wei = q @ k.transpose(-2, -1) * C ** -0.5
    wei = wei.masked_fill(self.tril[:T, :T]==0, float("-inf"))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)

    v=self.value(x)
    out = wei @ v

    return out


class UnmaskedHead(nn.Module):
  def __init__(self, head_size):
      super().__init__()
      self.key = nn.Linear(n_embd, head_size, bias=False)
      self.query = nn.Linear(n_embd, head_size, bias=False)
      self.value = nn.Linear(n_embd, head_size, bias=False)
      self.register_buffer('tril', torch.tril(torch.ones(ctx_len, ctx_len)))
      self.dropout = nn.Dropout(dropout)

  def forward(self, x):
      B, T, C = x.shape
      k = self.key(x)
      q = self.query(x)

      wei = q @ k.transpose(-2, -1) * C ** -0.5
      wei = F.softmax(wei, dim=-1)
      wei = self.dropout(wei)

      v = self.value(x)
      out = wei @ v

      return out


class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size, mask=False):
    super().__init__()
    self.heads = nn.ModuleList([MaskedHead(head_size) if mask else UnmaskedHead(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out