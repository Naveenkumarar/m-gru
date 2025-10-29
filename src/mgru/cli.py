import argparse, pathlib, torch
from rich import print
from .data.synthetic import DelayedSine
from .trainers.supervised_trainer import train_regression, evaluate_regression
from .models.mgru import MGRU
from .models.gru_baseline import GRUBaseline
from .utils.viz import plot_run

def main():
    parser = argparse.ArgumentParser("mgru")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--model", choices=["gru","mgru"], default="mgru")
    p_train.add_argument("--hidden_size", type=int, default=64)
    p_train.add_argument("--delay", type=int, default=0)
    p_train.add_argument("--use_smith", action="store_true")
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--run", default=".runs/mgru")
    p_train.add_argument("--device", default="cpu")

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--model", choices=["gru","mgru"], default="mgru")
    p_eval.add_argument("--ckpt", required=True)
    p_eval.add_argument("--delay", type=int, default=0)
    p_eval.add_argument("--device", default="cpu")

    p_plot = sub.add_parser("plot")
    p_plot.add_argument("--run", required=True)

    args = parser.parse_args()

    if args.cmd == "train":
        ds = DelayedSine(length=1024, seq_len=64, delay=args.delay)
        train_regression(
            dataset=ds,
            model_name=args.model,
            hidden_size=args.hidden_size,
            delay=args.delay,
            use_smith=args.use_smith,
            epochs=args.epochs,
            run_dir=args.run,
            device=args.device,
        )
        print(f"[green]Done. Run stored at {args.run}[/green]")

    elif args.cmd == "eval":
        ds = DelayedSine(length=256, seq_len=64, delay=args.delay)
        input_size = output_size = 1
        hidden = 64
        if args.model == "gru":
            model = GRUBaseline(input_size, hidden, output_size)
        else:
            model = MGRU(input_size, hidden, output_size, delay=args.delay, use_smith=True)
        state = torch.load(args.ckpt, map_location=args.device)
        model.load_state_dict(state)
        model.to(args.device).eval()
        print(evaluate_regression(ds, model, device=args.device))

    elif args.cmd == "plot":
        plot_run(args.run)

