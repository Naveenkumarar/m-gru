import json, pathlib
from typing import Literal, Optional
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..models.gru_baseline import GRUBaseline
from ..models.mgru import MGRU
from ..utils.metrics import regression_metrics

def make_model(model: Literal["gru","mgru"], input_size: int, hidden_size: int,
               output_size: int, delay: int, use_smith: bool):
    if model == "gru":
        return GRUBaseline(input_size, hidden_size, output_size)
    return MGRU(input_size, hidden_size, output_size, delay=delay, use_smith=use_smith)

def train_regression(dataset, model_name="mgru", hidden_size=64, delay=0, use_smith=True,
                     batch_size=32, lr=1e-3, epochs=5, run_dir=".runs/mgru",
                     num_workers=0, device="cpu"):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    input_size = output_size = 1
    model = make_model(model_name, input_size, hidden_size, output_size, delay, use_smith).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    run_path = pathlib.Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    logf = (run_path / "log.jsonl").open("w")

    step = 0
    for epoch in range(epochs):
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            rec = {"step": step, "loss": float(loss.item())}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            pbar.set_postfix(rec)
    torch.save(model.state_dict(), run_path / "latest.pt")
    logf.close()
    return model

@torch.no_grad()
def evaluate_regression(dataset, model, device="cpu"):
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, batch_size=64, shuffle=False)
    preds, targs = [], []
    for x, y in dl:
        x = x.to(device)
        y_hat = model(x).cpu()
        preds.append(y_hat); targs.append(y)
    import torch as T
    y_hat = T.cat(preds, dim=0); y = T.cat(targs, dim=0)
    return regression_metrics(y_hat, y)

