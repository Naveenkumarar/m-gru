from mgru.models.mgru import MGRU
from mgru.models.gru_baseline import GRUBaseline
import torch

def test_forward_shapes():
    x = torch.randn(2, 16, 1)
    m1 = GRUBaseline(1, 32, 1)
    m2 = MGRU(1, 32, 1, delay=2, use_smith=True)
    assert m1(x).shape == (2, 16, 1)
    assert m2(x).shape == (2, 16, 1)

