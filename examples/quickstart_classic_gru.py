import torch
from mgru.models.gru_baseline import GRUBaseline
from mgru.data.synthetic import DelayedSine

ds = DelayedSine(length=512, seq_len=64, delay=0)
model = GRUBaseline(1, 64, 1)
x, y = ds[0]
y_hat = model(x.unsqueeze(0))
print(y_hat.shape)

