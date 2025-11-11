# HET: Hypercomplex Eulerian Transformer  
**Train**: `het_train.py` • **Inference**: `het_inference.py`

A compact research rig for byte-level causal LMs with optional **Quaternion Attention** and **Quaternion RoPE**, plus T4-safe AMP, cosine LR with warmup, sliding-window corpus streaming, and fused quaternion linears for fast inference.

---

## Contents
- [Features](#features)
- [Install](#install)
- [Data](#data)
- [Training (`het_train.py`)](#training-het_trainpy)
  - [Quickstart](#quickstart)
  - [Important constraints](#important-constraints)
  - [Key behaviors](#key-behaviors)
  - [All CLI flags](#all-cli-flags)
  - [Examples](#training-examples)
- [Inference (`het_inference.py`)](#inference-het_inferencepy)
  - [Quickstart](#inference-quickstart)
  - [One-shot mode](#one-shot-mode)
  - [REPL mode](#repl-mode)
  - [All CLI flags](#inference-cli-flags)
  - [Examples](#inference-examples)
- [Checkpoints & Formats](#checkpoints--formats)
- [AMP / dtypes](#amp--dtypes)
- [Fused export](#fused-export)
- [Eval & Logging](#eval--logging)
- [Troubleshooting](#troubleshooting)
- [Performance tips](#performance-tips)
- [Reproducibility](#reproducibility)
- [Ablations (optional)](#ablations-optional)

---

## Features
- **Byte tokenizer (256-vocab)** — no preprocessing; good for raw text and ragged corpora.
- **Sliding-window streaming** — overlap windows across the full corpus with `sliding_keep_pct`.
- **Quaternion stack**  
  - **Quaternion Attention** (`--quat_attention`): Q/K/V/O via quaternion linears.  
  - **Quaternion RoPE** (`--quat_rope`, `--qrope_axis`, `--qrope_conjugate`).
- **Cosine LR with warmup**, AdamW, grad clipping.
- **AMP**: bf16/fp16 autocast on CUDA; safe on T4 (bf16 falls back to fp16).
- **Resume vs Warm-start**: `--resume_ckpt` (full state) vs `--load_weights` (model only).
- **Eval & checkpointing**: saves only on **val-loss improvement**.
- **Fused quaternion linears** for inference export.

---

## Install
```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate     # (PowerShell) .\\.venv\\Scripts\\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu124  # choose your CUDA/CPU build
pip install numpy
```

> Torch install line depends on your platform/CUDA. Use PyTorch’s selector if unsure.

---

## Data
Provide a single text file:
```text
./Corpus.txt
```
The trainer uses a byte-level tokenizer, so **no preprocessing** required.

---

## Training (`het_train.py`)

### Quickstart
Baseline (standard attention + standard RoPE), CUDA, AMP, 2 epochs:
```bash
python het_train.py --corpus ./Corpus.txt --device cuda --amp \
  --d_model 256 --n_layers 12 --n_heads 8 \
  --seq_len 512 --batch_size 48 --epochs 2 \
  --lr 7e-5 --min_lr 1e-5 --warmup 300 \
  --sliding_keep_pct 0.14
```

Quaternion Attention + Quaternion RoPE (cycle axis):
```bash
python het_train.py --corpus ./Corpus.txt --device cuda --amp \
  --d_model 256 --n_layers 12 --n_heads 8 --seq_len 512 --batch_size 48 \
  --epochs 2 --lr 7e-5 --min_lr 1e-5 --warmup 300 \
  --sliding_keep_pct 0.14 --quat_attention --quat_rope --qrope_axis cycle
```

Resume from a full checkpoint:
```bash
python het_train.py --corpus ./Corpus.txt --device cuda --amp \
  --resume_ckpt ckpts_quat/model_step1200.pt
```

Warm-start weights **only** (no optimizer/step state):
```bash
python het_train.py --corpus ./Corpus.txt --device cuda --amp \
  --load_weights ckpts_quat/model_step1200.pt
```

### Important constraints
- **Quaternion Attention requires 4-channel head dims**:
  ```
  d_model % (4 * n_heads) == 0
  ```
  The script enforces this and exits with an error if violated.
- **Quaternion RoPE requires Quaternion Attention**: `--quat_attention` must be set when using `--quat_rope`.

### Key behaviors
- **Cosine LR** with `--warmup` (in steps). If `--steps > 0`, train for exactly that many steps; otherwise derive from `--epochs`.
- **Sliding windows**: stride = `seq_len - floor(seq_len * sliding_keep_pct)`. Controls token reuse vs. coverage.
- **Eval/Checkpoints**: every `--eval_every` steps; checkpoint saved **only when** `val_loss` improves; name `model_step{N}.pt`.
- **Device & dtype**: compute dtype via AMP (`--dtype fp16|bf16|fp32`); weights stay FP32.

### All CLI flags
```
--corpus (str, required)
--out_dir (str, default: ckpts_quat)
--d_model (int, 512)
--n_layers (int, 8)
--n_heads (int, 8)
--ffn_mult (int, 4)
--dropout (float, 0.0)
--quat_attention (flag)
--quat_rope (flag; requires --quat_attention)
--qrope_axis (cycle|i, default cycle)
--qrope_conjugate (flag)
--seq_len (int, 512)
--batch_size (int, 8)
--steps (int, 0)        # if >0, overrides epochs
--epochs (float, 1.0)
--eval_every (int, 200)
--lr (float, 1e-3)
--min_lr (float, 1e-4)
--warmup (int, 200)
--weight_decay (float, 0.1)
--beta1 (float, 0.9)
--beta2 (float, 0.95)
--grad_clip (float, 1.0)
--amp (flag)            # enable CUDA autocast
--dtype (fp32|fp16|bf16; default fp16)
--device (str, default cuda if available else cpu)
--seed (int, 1337)
--compile (flag)        # torch.compile if available
--train_split (float, 0.9)
--sliding_keep_pct (float, 0.14)
--num_workers (int, 2)
--pin_memory (flag)
--fast_export (flag)            # write fused model for inference
--ffn_only_fast_export (flag)   # fuse only FFN layers
--load_weights (path or None)   # model weights only
--resume_ckpt (path or None)    # full resume (model+optimizer+step)
```

### Training examples
Small quaternion model, bf16 on A100 (falls back to fp16 on T4):
```bash
python het_train.py --corpus ./Corpus.txt --device cuda --amp --dtype bf16 \
  --d_model 256 --n_layers 6 --n_heads 8 --seq_len 512 --batch_size 64 \
  --epochs 3 --lr 7e-5 --min_lr 1e-5 --warmup 300 \
  --sliding_keep_pct 0.14 --quat_attention --quat_rope --qrope_axis cycle
```

Export fused weights for fast inference:
```bash
python het_train.py --corpus ./Corpus.txt --device cuda --amp \
  --d_model 256 --n_layers 6 --n_heads 8 --epochs 3 \
  --fast_export
# => ckpts_quat/model_fused_infer.pt
```

---

## Inference (`het_inference.py`)

### Inference Quickstart
Run one-shot generation from a training checkpoint:
```bash
python het_inference.py --ckpt ckpts_quat/model_step1200.pt --device cuda \
  --dtype fp16 --prompt "Summarize the central themes of The Lord of the Rings." \
  --temperature 0.65 --top_p 0.95 --max_new 128
```

REPL chat:
```bash
python het_inference.py --ckpt ckpts_quat/model_step1200.pt --device cuda --dtype fp16
```

> The inference model is reconstructed from the saved `config` embedded in the checkpoint. If you exported a fused model, pass `--fast` to use `FastQuaternionLinear`.

### One-shot mode
Provide `--prompt` to generate once and exit. Useful for batch eval or scripts.

### REPL mode
No `--prompt` → a simple chat REPL with byte-level history.  
`--sliding_keep_pct` (in REPL) limits how much prior transcript is retained between turns.

### Inference CLI flags
```
--ckpt (str, required)           # checkpoint from het_train.py
--device (str, default cuda if available else cpu)
--dtype (fp32|fp16|bf16; default fp16)
--fast (flag)                    # use FastQuaternionLinear
--max_seq_len (int, 2048)
--max_new (int, 256)
--temperature (float, 0.8 default)
--top_k (int, 50)                # set 0 to disable
--top_p (float, 0.9)
--repetition_penalty (float, 1.1)
--sliding_keep_pct (float, 1.0)  # REPL history retention [0..1]
--prompt (str or None)           # one-shot if set
--sys_prefix (str, default "You are a helpful model.")
--user_prefix (str, default "User: ")
--asst_prefix (str, default "Reply: ")
--stop_str (str, default "\\nUser: ")
```

### Inference examples
Pure nucleus sampling (no top-k), low temp:
```bash
python het_inference.py --ckpt ckpts_quat/model_step1200.pt --device cuda \
  --dtype fp16 --prompt "Describe Rivendell at dawn." \
  --temperature 0.65 --top_k 0 --top_p 0.95 --max_new 192
```

Fast fused linears:
```bash
python het_inference.py --ckpt ckpts_quat/model_fused_infer.pt --device cuda \
  --dtype fp16 --fast --prompt "Write a single haiku about journeys." \
  --temperature 0.7 --top_p 0.95 --max_new 64
```

---

## Checkpoints & Formats
Training checkpoints (improvement-only) are named:
```
out_dir/model_step{N}.pt
```
They contain:
```python
{
  "model": state_dict,             # FP32 weights
  "optimizer": opt_state,          # (only in training ckpts)
  "config": asdict(TrainConfig),   # full train-time config
  "step": int,
  "best_val": float,
  "vocab_size": 256
}
```
Fused export (`--fast_export`) writes:
```
out_dir/model_fused_infer.pt
# contains {"model": fused_state_dict, "config": cfg, "vocab_size": 256}
```

---

## AMP / dtypes
- **Training** uses `torch.amp.autocast('cuda', dtype=fp16|bf16)` when `--amp` is set.
- **bf16** support is queried; if unsupported (e.g., T4), it **falls back to fp16** with a warning.
- **Weights remain FP32**; AMP only affects compute.

---

## Fused export
- `swap_to_fast()` can replace `QuaternionLinear` with `FastQuaternionLinear` for inference.
- Enable at train end via `--fast_export` (and optionally `--ffn_only_fast_export`).
- Load with `het_inference.py` and pass `--fast` to ensure fused path usage.

---

## Eval & Logging
- Training logs stream to console and `out_dir/train.log`.
- Eval every `--eval_every` steps prints and logs:
  ```
  [eval] step {N} | val_loss {x.xxxx}
  ```
- Model is saved **only** if `val_loss` improves `best_val`.

---

## Troubleshooting
**Error:** *“Q-RoPE requires --quat_attention…”*  
→ Add `--quat_attention` when using `--quat_rope`.

**Error:** *“d_model must be divisible by 4*n_heads…”*  
→ Ensure `d_model % (4 * n_heads) == 0` whenever `--quat_attention` is set.

**NaNs or loss spikes**  
- Lower `--lr`, increase `--warmup`, or disable AMP (`--dtype fp32`, omit `--amp`) to test.
- Reduce `--batch_size` or `--seq_len` to fit memory; keep overlap reasonable.

**Slow dataloader**  
- Increase `--num_workers`; set `--pin_memory` on CUDA.

**No checkpoints saved**  
- If `val_loss` never improves, no file is written. Reduce `--eval_every` or extend training (`--epochs` or `--steps`).

---

## Performance tips
- Prefer **CUDA + AMP** (`--amp`, `--dtype bf16` if GPU supports it).  
- Use **overlap** (`--sliding_keep_pct ~0.1–0.2`) for better token efficiency on small corpora.
- Keep `--eval_every` modest (e.g., 300–800) to avoid frequent stalls on big models.
- For quaternions: keep heads numerous enough that `d_model / n_heads` is a multiple of 4.

---

## Reproducibility
- Set `--seed`. The code seeds Python & Torch; cuDNN benchmarking is **enabled** for speed, so exact determinism isn’t guaranteed. For stricter determinism, you may toggle PyTorch backend flags manually (speed will drop).

---

## Ablations (optional)
If you’re using the companion `het_ablate.py`, it runs A0–A3 variants and (optionally) **inference** per best checkpoint. It mirrors train logs, dumps CSV/JSON summaries, and can preview generations in `ablation_summary.txt`. Make sure `het_ablate.py` lives next to `het_train.py` and `het_inference.py`.

---

## License / Citation
This is a research scaffold. If you publish results, note quaternion attention and RoPE variants and include training details (overlap, cosine LR, warmup, AMP settings, and eval cadence).
#   H E T  
 