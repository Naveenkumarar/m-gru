import torch, torch.nn as nn

class GRUBaseline(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (B, L, D)
        h, _ = self.gru(x)
        y = self.head(h)
        return y

