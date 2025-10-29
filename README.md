# Memento-GRU (M-GRU)

A minimal, research-friendly package implementing:
- GRU baseline
- Tiny-MLP predictor
- Smith-style delay compensation
- Adaptive feedback gain
- Trainer, CLI, synthetic dataset, and plots

## Install (editable)
```bash
pip install -e .
```

## Quickstart

```bash
mgru train --model gru --epochs 2
mgru train --model mgru --epochs 2 --delay 3
mgru eval --model mgru --ckpt .runs/mgru/latest.pt
mgru plot --run .runs/mgru
```

See examples/ for Python APIs.

