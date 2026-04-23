# FILE: config.py
"""
Central configuration for the MoD SLM project.
All hyperparameters, file paths, and device settings live here.
Change values here ONLY -- no other file should have magic numbers.
"""

import torch
import os

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# FILE PATHS
# =============================================================================
# Raw corpus text files (will be merged automatically)
CORPUS_FILES = [
    "data/raw/1 and 2 .txt",
    "data/processed/mate3_legal_corpus.txt",
    "data/processed/mate4_audit_corpus.txt",
    "data/processed/mate5_rti_corpus.txt",
    "data/processed/synthetic_mod_corpus.txt",
    "data/processed/synthetic_mod_corpus_large.txt",
    "data/processed/synthetic_mod_faq.txt",
    "data/raw/Artificial_Intelligence.txt",
    "data/raw/Biology.txt",
    "data/raw/Chemistry.txt",
    "data/raw/History.txt",
    "data/raw/Space.txt",
    "data/raw/geography.txt",
    "data/raw/physics.txt",
]

# Merged corpus written here before tokenizer training
CORPUS_PATH = "data/processed/mod_slm_stage1_corpus.txt"

# Tokenizer save paths
VOCAB_PATH      = "models/custom/tokenizer_vocab.json"
MERGES_PATH     = "models/custom/tokenizer_merges.txt"

# Model checkpoint directory and best model path
CHECKPOINT_DIR  = "models/checkpoints"
BEST_MODEL_PATH = "models/custom/best_model.pt"

# Training log
TRAINING_LOG    = "models/training_log.txt"

# =============================================================================
# TOKENIZER PARAMETERS
# =============================================================================
VOCAB_SIZE = 5000   # BPE vocabulary size

# Special token assignments
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3

# =============================================================================
# MODEL ARCHITECTURE  (tuned for RTX 3050 4 GB VRAM)
# =============================================================================
EMBEDDING_DIM    = 256    # Width of every embedding / hidden state
NUM_HEADS        = 8      # Attention heads  (head_dim = 256 / 8 = 32)
NUM_LAYERS       = 6      # Transformer blocks stacked
CONTEXT_LENGTH   = 256    # Maximum sequence length in tokens
FEEDFORWARD_DIM  = 1024   # Inner dimension of the FFN (4 x embedding_dim)
DROPOUT          = 0.1    # Dropout probability throughout the model

# =============================================================================
# DATASET / DATALOADER
# =============================================================================
STRIDE      = 128   # Sliding window stride (50 % overlap)
BATCH_SIZE  = 8     # Safe for 4 GB VRAM with the architecture above
NUM_WORKERS = 2     # DataLoader worker processes
PIN_MEMORY  = True  # Speeds up CPU->GPU transfer

# =============================================================================
# TRAINING  HYPERPARAMETERS
# =============================================================================
NUM_EPOCHS        = 200
LEARNING_RATE     = 3e-4
WEIGHT_DECAY      = 0.01
BETAS             = (0.9, 0.95)
WARMUP_STEPS      = 100
MIN_LR            = 1e-5          # Cosine floor
GRAD_CLIP_NORM    = 1.0           # Gradient clipping max norm
LOG_EVERY_STEPS   = 50            # Print loss every N steps
SAMPLE_EVERY_STEPS = 500          # Print generated sample every N steps

# =============================================================================
# INFERENCE DEFAULTS
# =============================================================================
MAX_NEW_TOKENS    = 200
TEMPERATURE       = 0.2
TOP_K             = 20
