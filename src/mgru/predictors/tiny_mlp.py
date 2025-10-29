import torch, torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, hid: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.Tanh(),
            nn.Linear(hid, out_dim),
        )
    def forward(self, x):
        return self.net(x)

