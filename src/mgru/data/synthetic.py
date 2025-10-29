from typing import Tuple
import torch
from torch.utils.data import Dataset

class DelayedSine(Dataset):
    """
    Generates (x_t, y_t) where y_t = sin(2π f (t-d)/T) with noise.
    Useful to test delay compensation.
    """
    def __init__(self, length: int = 2048, seq_len: int = 64, delay: int = 0, freq: float = 3.0,
                 noise: float = 0.05, device: str = "cpu"):
        self.length = length
        self.seq_len = seq_len
        self.delay = delay
        self.freq = freq
        self.noise = noise
        self.device = device

        t = torch.arange(0, length + seq_len + 2, dtype=torch.float32, device=device)
        base = torch.sin(2.0 * torch.pi * self.freq * t / 100.0)
        y = torch.roll(base, shifts=delay)
        x = base + self.noise * torch.randn_like(base)
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_seq = self.x[idx:idx+self.seq_len].unsqueeze(-1)  # (L,1)
        y_seq = self.y[idx:idx+self.seq_len].unsqueeze(-1)
        return x_seq, y_seq

