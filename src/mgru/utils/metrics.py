import torch
from typing import Dict

def regression_metrics(y_hat: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    mse = torch.mean((y_hat - y) ** 2).item()
    rmse = mse ** 0.5
    return {"mse": mse, "rmse": rmse}

