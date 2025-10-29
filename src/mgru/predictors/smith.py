import torch

class SmithPredictor:
    """
    Stateless helper: combine current MLP prediction with delayed hidden estimate.
    h_bar_{t-d} is provided by rolling buffer outside.
    """
    def __call__(self, h_mlp: torch.Tensor, h_prev: torch.Tensor, h_delayed: torch.Tensor):
        # Smith: h_hat = h_mlp + (h_prev - h_delayed)
        return h_mlp + (h_prev - h_delayed)

