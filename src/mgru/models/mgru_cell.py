import torch, torch.nn as nn
from ..predictors.tiny_mlp import TinyMLP
from ..predictors.smith import SmithPredictor
from ..gates.adaptive_gain import AdaptiveGain

class MGRUCell(nn.Module):
    """
    GRU-like cell with:
      - TinyMLP predictor on [x_t, h_{t-1}]
      - (optional) Smith compensation using delayed hidden
      - Adaptive gain gate controlling update magnitude
    """
    def __init__(self, input_size: int, hidden_size: int, use_smith: bool = True, gain_vector: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_smith = use_smith

        self.Wr = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wz = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wh = nn.Linear(input_size + hidden_size, hidden_size)

        self.mlp = TinyMLP(in_dim=input_size + hidden_size, hid=max(16, hidden_size // 2), out_dim=hidden_size)
        self.smith = SmithPredictor()
        self.gain = AdaptiveGain(hidden_size, vector=gain_vector)

    def forward(self, x_t, h_prev, h_delay):
        # GRU gates
        concat = torch.cat([x_t, h_prev], dim=-1)
        r_t = torch.sigmoid(self.Wr(concat))
        z_t = torch.sigmoid(self.Wz(concat))
        h_hat = torch.tanh(self.Wh(torch.cat([x_t, r_t * h_prev], dim=-1)))

        # Predictor path
        h_mlp = self.mlp(concat)
        if self.use_smith:
            h_pred = self.smith(h_mlp, h_prev, h_delay)
        else:
            h_pred = h_mlp

        # Blend: predictor proposes h_tilde; adaptive gain scales the delta
        h_tilde = (1 - z_t) * h_prev + z_t * h_hat
        g_t = self.gain(h_tilde)  # in [0,1]
        h_new = h_tilde + g_t * (h_pred - h_tilde)
        return h_new

