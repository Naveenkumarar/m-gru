from mgru.data.synthetic import DelayedSine
from mgru.trainers.supervised_trainer import train_regression

def test_train_one_epoch(tmp_path):
    ds = DelayedSine(length=128, seq_len=16, delay=2)
    model = train_regression(ds, model_name="mgru", epochs=1, run_dir=str(tmp_path))
    assert model is not None

