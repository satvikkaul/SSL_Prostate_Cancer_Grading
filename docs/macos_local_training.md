# macOS Local Training Notes

This project now supports a practical macOS local workflow for smoke tests and short pilot runs.

## Goal

Use the Mac for:
- environment verification
- data pipeline checks
- 1-epoch or low-epoch pilot runs
- debugging model startup and checkpoint loading

Use Colab for:
- full SSL pretraining runs
- final comparison runs
- heavy MoCo / SimCLR experiments

## Verified Local Environment

The project `venv` was updated to:
- `tensorflow==2.18.1`
- `tensorflow-metal==1.2.0` via `requirements-macos.txt`

This fixes the broken import path seen with `tensorflow==2.20.0` + `tensorflow-metal`.

## Current Status On This Machine

On the current machine:
- Chip: Apple M5 Pro
- macOS: 26.3.2
- TensorFlow import works
- `tensorflow-metal` installs
- TensorFlow now detects the Metal device as `GPU:0`

So local runs can be used for:
- smoke tests
- short pilot runs
- limited fine-tuning

Full SSL sweeps are still better suited to Colab.

## Setup

From the repo root:

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements-macos.txt
```

## Verify TensorFlow

```bash
MPLCONFIGDIR=/tmp/mpl ./venv/bin/python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices()); print(tf.config.list_physical_devices('GPU'))"
```

Expected on this machine now:
- TensorFlow imports successfully
- GPU list includes `GPU:0`

## Recommended Local Smoke Tests

### MoCo pretraining

```bash
./venv/bin/python training/moco/pretrain_moco.py --epochs 1 --batch_size 8
```

### MoCo fine-tuning

```bash
./venv/bin/python training/moco/finetune_moco.py --checkpoint output/models/moco/encoder_q_epoch010.weights.h5
```

### MoCo evaluation

```bash
./venv/bin/python evaluation/moco/eval_moco.py
```

## Recommended Local Settings

For Mac local runs:
- keep epochs low
- use smaller batch sizes (`8` or lower)
- prefer MoCo pilot runs over full sweeps
- use `./venv/bin/python` explicitly
- verify thermals/memory before longer runs

## Notes

- The repo scripts now default to writable Matplotlib cache handling for the MoCo path.
- The MoCo CPU-side stall on the first step was fixed by avoiding the problematic graph-compiled loss path on CPU-only Apple Silicon runs.
- If GPU detection disappears after a future update, re-run the TensorFlow verification command before starting longer local jobs.
