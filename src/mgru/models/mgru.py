import torch, torch.nn as nn
from .mgru_cell import MGRUCell

class MGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 delay: int = 0, use_smith: bool = True):
        super().__init__()
        self.cell = MGRUCell(input_size, hidden_size, use_smith=use_smith)
        self.head = nn.Linear(hidden_size, output_size)
        self.delay = delay

    def forward(self, x):
        # x: (B,L,D)
        B, L, _ = x.shape
        device = x.device
        h = torch.zeros(B, self.cell.hidden_size, device=device)
        # circular buffer for delayed hidden states
        hist = [torch.zeros_like(h) for _ in range(self.delay + 1)]

        outputs = []
        for t in range(L):
            h_delay = hist[0]  # oldest
            h = self.cell(x[:, t, :], h, h_delay)
            # push into buffer
            hist.append(h)
            hist.pop(0)
            outputs.append(self.head(h).unsqueeze(1))
        return torch.cat(outputs, dim=1)

