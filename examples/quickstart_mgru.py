import torch
from mgru.models.mgru import MGRU
from mgru.data.synthetic import DelayedSine

ds = DelayedSine(length=512, seq_len=64, delay=3)
model = MGRU(1, 64, 1, delay=3, use_smith=True)
x, y = ds[0]
y_hat = model(x.unsqueeze(0))
print(y_hat.shape)

