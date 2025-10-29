import torch, torch.nn as nn

class AdaptiveGain(nn.Module):
    """
    Produces a scalar or vector gain g_t in [0,1] per time step.
    """
    def __init__(self, hidden_size: int, vector: bool = True):
        super().__init__()
        self.vector = vector
        self.proj = nn.Linear(hidden_size, hidden_size if vector else 1)

    def forward(self, h_tilde: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.proj(h_tilde))
        return g

