# MoD SLM — Small Language Model for the Indian Ministry of Defence

A decoder-only Transformer Language Model built **completely from scratch**
in pure PyTorch. No HuggingFace pretrained models. No fine-tuning.

Trained on MoD policy documents, legal text, budget documents, RTI guidelines,
and audit reports.

---

## Folder Structure

```
slm/
├── config.py              ← All hyperparameters (edit HERE only)
├── merge_corpus.py        ← Step 0: merge raw corpus files
├── tokenizer.py           ← Step 1: BPE tokenizer (train + save + load)
├── model.py               ← Transformer model definition
├── dataset.py             ← PyTorch Dataset + DataLoader
├── train.py               ← Step 2: full training loop
├── inference.py           ← Step 3: interactive query interface
├── requirements.txt       ← Python dependencies
│
├── 1 and 2 .txt           ← Raw corpus file 1 & 2
├── mate3_legal_corpus.txt ← Raw corpus file 3
├── mate4_audit_corpus.txt ← Raw corpus file 4
├── mate5_rti_corpus.txt   ← Raw corpus file 5
│
├── mod_slm_stage1_corpus.txt   ← Merged corpus (created by merge_corpus.py)
├── tokenizer_vocab.json        ← BPE vocab (created by tokenizer.py)
├── tokenizer_merges.txt        ← BPE merges (created by tokenizer.py)
│
├── checkpoints/
│   ├── checkpoint_epoch_0.pt
│   ├── checkpoint_epoch_1.pt
│   └── ...
│
├── best_model.pt          ← Best model weights (lowest training loss)
└── training_log.txt       ← Loss log
```

---

## Step 1 — Install Requirements

First, install PyTorch **with CUDA support** (this is mandatory for GPU training).

**For CUDA 11.8 (most common on RTX 3050):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies:
```bash
pip install numpy tqdm
```

> **Do NOT** run `pip install torch` without the `--index-url` flag.
> That installs a CPU-only build and training will be 20–50× slower.

---

## Step 2 — Put Corpus Files in the Correct Location

Place all raw `.txt` corpus files in the **same directory** as `config.py`.
The filenames are already configured in `config.py`:

```python
CORPUS_FILES = [
    "1 and 2 .txt",
    "mate3_legal_corpus.txt",
    "mate4_audit_corpus.txt",
    "mate5_rti_corpus.txt",
]
```

If you have additional files, add them to this list.

---

## Step 3 — Merge Corpus Files

Merge all raw corpus files into one:

```bash
python merge_corpus.py
```

This creates `mod_slm_stage1_corpus.txt` in the project directory.

---

## Step 4 — Train the Tokenizer

```bash
python tokenizer.py
```

This reads `mod_slm_stage1_corpus.txt`, trains a BPE tokenizer with 8,000
tokens, and saves:
- `tokenizer_vocab.json`
- `tokenizer_merges.txt`

**Expected output:**
```
[Tokenizer] Reading corpus from 'mod_slm_stage1_corpus.txt' ...
[Tokenizer] Corpus size: 189,000 characters
[Tokenizer] Initial char vocab size: 130
[Tokenizer] Training BPE to vocab size 8000 ...
  Merge 500/7870  vocab_size=630
  Merge 1000/7870 vocab_size=1130
  ...
[Tokenizer] Final vocab size: 8000
```

Training takes **2–5 minutes** depending on corpus size.

---

## Step 5 — Train the Model

```bash
python train.py
```

Training runs for 10 epochs with:
- Mixed precision (AMP) enabled automatically on CUDA
- Checkpoint saved after every epoch to `checkpoints/`
- Best model saved to `best_model.pt`
- Loss logged to `training_log.txt`
- Sample generated text printed every 500 steps

**Expected training time on RTX 3050 (4 GB VRAM):**

| Corpus Size | Approx Time / Epoch | Total (10 epochs) |
|-------------|---------------------|--------------------|
| ~200 K chars | 2–5 min             | 20–50 min          |
| ~1 M chars   | 10–20 min           | 1.5–3 hr           |
| ~10 M chars  | 90–150 min          | 15–25 hr           |

**Expected loss values:**
- Epoch 1: loss ≈ 6.0–7.0 (random-walk territory)
- Epoch 3: loss ≈ 4.0–5.0 (model starts learning structure)
- Epoch 7+: loss ≈ 2.5–3.5 (coherent phrase fragments)
- Below 2.0: model has fit the corpus well

**When to stop training:** Stop when validation/training loss stops
decreasing for 2+ epochs. With this small corpus the model may overfit
after epoch 5–7.

---

## Step 6 — Run Inference

```bash
python inference.py
```

This loads `best_model.pt` and opens an interactive prompt:

```
================================================
  MoD SLM — Interactive Inference
  Type 'quit' or 'exit' to stop.
================================================

> You: The Ministry of Defence allocates funds for

  Model: capital expenditure under the defence budget as per
         the revised estimates approved by Parliament...
```

Type `quit` to exit.

---

## Step 7 — Resume Training from Checkpoint

Training resumes automatically if a checkpoint exists.  Just run:

```bash
python train.py
```

The script scans `checkpoints/checkpoint_epoch_N.pt` and picks the latest.
It restores the model weights, optimiser state, global step, and best loss.

To start from scratch, delete the `checkpoints/` folder:
```bash
rmdir /s /q checkpoints
```

---

## How to Check if GPU Is Being Used

At the start of training you will see:
```
  GPU    : NVIDIA GeForce RTX 3050 Laptop GPU
  VRAM   : 4.0 GB
  Device : CUDA
```

During training, run in a separate terminal:
```bash
nvidia-smi
```

You should see GPU utilisation > 50 % and memory usage ~2–3 GB.

In Python:
```python
import torch
print(torch.cuda.is_available())          # True
print(torch.cuda.get_device_name(0))      # NVIDIA GeForce RTX 3050 Laptop GPU
print(torch.cuda.memory_allocated(0) / 1e9)  # GB in use
```

---

## Common Errors and Fixes (RTX 3050 Specific)

### `CUDA out of memory`
```
→ Open config.py
→ Reduce BATCH_SIZE from 8 to 4 (or 2)
→ Or reduce CONTEXT_LENGTH from 256 to 128
→ Run: torch.cuda.empty_cache() in a Python shell to free cached memory
```

### `cuDNN error: CUDNN_STATUS_NOT_INITIALIZED`
```
→ Your CUDA toolkit version may not match your PyTorch build.
→ Reinstall PyTorch for the correct CUDA version (11.8 or 12.1).
→ Check CUDA version: nvcc --version
```

### `torch.cuda.is_available()` returns `False`
```
→ PyTorch was installed without CUDA (common with plain pip install torch).
→ Uninstall and reinstall with the --index-url flag (see Step 1).
```

### `FileNotFoundError: tokenizer_vocab.json`
```
→ Run: python tokenizer.py   (before train.py)
```

### `FileNotFoundError: best_model.pt`
```
→ Run: python train.py   (before inference.py)
```

### Training loss is stuck at ~log(vocab_size) ≈ 9.0
```
→ The model is predicting uniformly (untrained).
→ Wait at least 200–500 steps; it will drop sharply once embeddings tune.
→ Check that DEVICE is 'cuda' and not 'cpu'.
```

### Very slow training (< 10 batches/sec)
```
→ Confirm GPU is being used (see section above).
→ Confirm mixed precision is active (requires CUDA).
→ Set NUM_WORKERS=0 in config.py if DataLoader is causing issues on Windows.
```

---

## Model Architecture Summary

| Parameter        | Value   |
|-----------------|---------|
| Vocabulary size | 8,000   |
| Embedding dim   | 256     |
| Attention heads | 8       |
| Layers          | 6       |
| Context length  | 256     |
| FFN dimension   | 1,024   |
| Dropout         | 0.1     |
| **Total params**| **~18 M** |

All parameters are adjustable in `config.py`.
