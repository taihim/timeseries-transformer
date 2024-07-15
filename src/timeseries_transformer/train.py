import torch
import torch.nn as nn
from torch.nn import functional as F

from src.vanilla_model import TransformerModel

eval_iters = 200




if __name__ == "__main__":
    model = TransformerModel()
    model.load_state_dict(torch.load("model.pt"))
    model.eval()
    model.cuda()
    for i in range(eval_iters):
        idx = torch.randint(0, 32, (4, 8)).cuda()
        idx = model.generate(idx, 8)
        print(idx)