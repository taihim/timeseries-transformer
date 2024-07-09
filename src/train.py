import torch
import torch.nn as nn
from torch.nn import functional as F

from src.vanilla_model import TransformerModel

eval_iters = 200

# context manager tells pytorch that w/e happens in the function, we wont call backward() on it
# so it doesnt need to store the intermediate values for backprop, is more memory efficient this way
@torch.no_grad()
def estimate_loss():
  out = {}
  # set model to eval mode
  m.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = m(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  # set it back to training mode
  m.train()
  return out



if __name__ == "__main__":
    m = TransformerModel()
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    eval_interval = 300
    batch_size = 32
    for iter in range(5000):
        xb, yb = get_batch('train')

        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()