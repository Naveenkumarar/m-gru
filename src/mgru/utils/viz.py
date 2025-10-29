import json, pathlib, matplotlib.pyplot as plt

def plot_run(run_dir: str):
    p = pathlib.Path(run_dir) / "log.jsonl"
    steps, loss = [], []
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open() as f:
        for line in f:
            rec = json.loads(line)
            steps.append(rec["step"])
            loss.append(rec["loss"])
    plt.figure()
    plt.plot(steps, loss)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training loss")
    plt.tight_layout()
    out = pathlib.Path(run_dir) / "loss.png"
    plt.savefig(out)
    print(f"Saved {out}")

