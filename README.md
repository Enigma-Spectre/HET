# HET: Hypercomplex Eulerian Transformer

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

HET is a compact research rig for training and experimenting with byte-level causal language models. It features a novel **Tiered Memory** architecture and support for **Quaternion** layers, designed for researchers and hobbyists who want to explore advanced transformer variants with a minimal, self-contained codebase.

The entire project is contained in a few core Python files, requiring only PyTorch.

## Key Features

*   **Advanced Tiered Memory:** A unique memory system with Short (S), Medium (M), and Long-term (L) banks that allows the model to develop a persistent, stateful understanding of concepts.
*   **Hypercomplex Layers:** Optional **Quaternion Attention** and **Quaternion RoPE** for exploring parameter-efficient, rotation-based network dynamics.
*   **Byte-Level Tokenizer:** Operates directly on UTF-8 bytes (256 vocab size). No preprocessing or external tokenizer is needed—just a single text file.
*   **Optimized for Single GPU:** Designed to be T4-safe with built-in Automatic Mixed Precision (AMP) and gradient accumulation, making it perfect for platforms like Google Colab.
*   **Self-Contained & Minimalist:** No complex dependencies. If you have PyTorch, you can run this.

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Enigma-Spectre/HET.git
    cd HET
    ```

2.  **Set up a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install PyTorch:**
    Install PyTorch according to your CUDA version. For example:
    ```bash
    pip install torch torchvision torchaudio
    ```

## Quickstart Guide

### 1. Prepare Your Data

Combine all your training text into a single file named `corpus.txt` and place it in the root directory. The model will learn from the raw byte patterns.

### 2. Train a New Model

The following command trains a small, memory-enabled model from scratch. It's configured to run well on a single T4 GPU (like in Google Colab).

```bash
python mem_het_train.py \
  --corpus "./corpus.txt" \
  --out_dir "./checkpoints" \
  --d_model 512 \
  --n_layers 12 \
  --n_heads 8 \
  --seq_len 512 \
  --batch_size 8 \
  --accum_steps 8 \
  --epochs 3 \
  --lr 5e-4 \
  --warmup 400 \
  --amp \
  --use_memory \
  --mem_layer_schedule 10 11 \
  --mem_s_sizes 64 \
  --mem_m_sizes 128 \
  --mem_l_sizes 128 \
  --mem_l_learnable
```

### 3. Chat with Your Model

Once training saves a checkpoint, you can interact with your model using the inference script.

```bash
# Replace with the actual path to your saved checkpoint
CHECKPOINT_PATH="./checkpoints/model_step1000.pt"

python mem_het_inference.py \
  --ckpt $CHECKPOINT_PATH \
  --fast \
  --temperature 0.2 \
  --top_p 0.9 \
  --user_prefix "User: " \
  --asst_prefix "Reply: "
```
The script will load the model and present you with a `>` prompt where you can start chatting.

## Advanced Features

### Tiered Memory Configuration

The memory architecture is highly configurable.

*   `--use_memory`: Enables the memory blocks.
*   `--mem_layer_schedule`: Specify which layers to augment with memory (e.g., `10 11` for the last two layers of a 12-layer model).
*   `--mem_s_sizes`, `--mem_m_sizes`, `--mem_l_sizes`: Set the sizes of the Short, Medium, and Long-term memory banks.
*   `--mem_l_learnable`: Makes the Long-term memory bank a trainable parameter.

### Quaternion Layers

Explore hypercomplex networks with these flags. **Requires `--quat_attention` to be enabled first.**

*   `--quat_attention`: Use `QuaternionLinear` for the Q, K, V, and O projections in the attention mechanism.
*   `--quat_rope`: Use Quaternion Rotary Position Embeddings instead of standard RoPE.

## File Overview

*   **`mem_het_train.py`**: The primary script for training and fine-tuning models with the tiered memory architecture.
*   **`mem_het_inference.py`**: A script for loading a trained checkpoint and interacting with it in a REPL-style chat.
*   **`het_ablate.py`**: An optional runner for conducting ablation studies and experiments.
*   **`het_train.py` / `het_inference.py`**: The original, non-memory versions of the scripts.
