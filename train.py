# FILE: train.py
"""
Full training loop for the MoD SLM.

Features
--------
- AdamW optimiser with cosine-decay + linear warm-up schedule
- Mixed-precision training via torch.cuda.amp (GradScaler + autocast)
- Gradient clipping (max_norm 1.0)
- Checkpoint saving every epoch + best model tracking
- Resume from checkpoint automatically
- Sample text generation every 500 steps
- Training log written to training_log.txt
"""

import os
import math
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from config import (
    DEVICE,
    # Paths
    CORPUS_PATH, VOCAB_PATH, MERGES_PATH,
    CHECKPOINT_DIR, BEST_MODEL_PATH, TRAINING_LOG,
    # Model
    VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS,
    CONTEXT_LENGTH, FEEDFORWARD_DIM, DROPOUT,
    # Dataset
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, STRIDE,
    # Optimiser / scheduler
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BETAS,
    WARMUP_STEPS, MIN_LR, GRAD_CLIP_NORM,
    # Logging
    LOG_EVERY_STEPS, SAMPLE_EVERY_STEPS,
    # Inference defaults
    MAX_NEW_TOKENS, TEMPERATURE, TOP_K,
    # Special tokens
    PAD_TOKEN_ID, BOS_TOKEN_ID,
)
from tokenizer import BPETokenizer
from model import SLMModel, count_parameters
from dataset import create_dataloader


# =============================================================================
# LEARNING RATE SCHEDULE — cosine decay with linear warmup
# =============================================================================

def get_lr(step: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    """
    Compute the learning rate at a given global step.

    During the warmup phase the LR increases linearly from 0 to ``max_lr``.
    After warmup it follows a cosine curve decaying to ``min_lr``.

    Parameters
    ----------
    step : int
        Current global training step.
    warmup_steps : int
        Number of warmup steps.
    max_lr : float
        Peak learning rate (reached at end of warmup).
    min_lr : float
        Floor learning rate (approached as step → ∞).

    Returns
    -------
    float
        Learning rate for the current step.
    """
    if step < warmup_steps:
        # Linear ramp-up
        return max_lr * (step + 1) / warmup_steps

    # Cosine decay after warmup
    # We map step ∈ [warmup_steps, ∞) to t ∈ [0, 1]
    progress = (step - warmup_steps) / max(1, 10_000 - warmup_steps)
    progress = min(progress, 1.0)   # cap at 1
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """
    Manually update the learning rate in all parameter groups.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimiser whose LR should be updated.
    lr : float
        New learning rate value.
    """
    for group in optimizer.param_groups:
        group['lr'] = lr


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def save_checkpoint(
    model: SLMModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    best_loss: float,
    filename: str,
) -> None:
    """
    Save a training checkpoint to disk.

    The checkpoint dictionary contains everything needed to resume training
    from exactly the same point.

    Parameters
    ----------
    model : SLMModel
        The model whose weights to save.
    optimizer : torch.optim.Optimizer
        The optimiser state (momentum buffers, etc.).
    epoch : int
        Epoch index just completed.
    step : int
        Global step count.
    best_loss : float
        Best validation/training loss seen so far.
    filename : str
        Destination file path.
    """
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    torch.save({
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch':                epoch,
        'step':                 step,
        'best_loss':            best_loss,
    }, filename)


def load_checkpoint(
    model: SLMModel,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
) -> tuple[int, int, float]:
    """
    Load a training checkpoint from disk and restore model + optimiser state.

    Parameters
    ----------
    model : SLMModel
        Model to load weights into.
    optimizer : torch.optim.Optimizer
        Optimiser to restore state into.
    checkpoint_path : str
        Path to the checkpoint ``.pt`` file.

    Returns
    -------
    tuple[int, int, float]
        ``(start_epoch, global_step, best_loss)``
    """
    print(f"[Train] Loading checkpoint from '{checkpoint_path}' ...")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    start_epoch = ckpt['epoch'] + 1
    global_step = ckpt['step']
    best_loss   = ckpt['best_loss']

    print(f"[Train] Resumed from epoch {ckpt['epoch']} | "
          f"step {global_step} | best_loss {best_loss:.4f}")
    return start_epoch, global_step, best_loss


# =============================================================================
# SAMPLE GENERATION (called during training to monitor quality)
# =============================================================================

@torch.no_grad()
def generate_sample(model: SLMModel, tokenizer: BPETokenizer, prompt: str) -> str:
    """
    Generate a short text sample from the model to monitor training progress.

    Parameters
    ----------
    model : SLMModel
        The model (will be temporarily set to eval mode).
    tokenizer : BPETokenizer
        Tokenizer for encoding the prompt.
    prompt : str
        Short text prompt to condition generation on.

    Returns
    -------
    str
        Generated continuation text (prompt excluded).
    """
    model.eval()

    ids     = [BOS_TOKEN_ID] + tokenizer.encode(prompt)
    inp     = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    out     = model.generate(inp, max_new_tokens=80, temperature=TEMPERATURE, top_k=TOP_K)
    new_ids = out[0, len(ids):].tolist()

    model.train()
    return tokenizer.decode(new_ids)


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train() -> None:
    """
    Full training loop for the MoD SLM.

    Execution order
    ---------------
    1. Load tokenizer from saved files.
    2. Initialise model and move to GPU.
    3. Load dataset and DataLoader.
    4. Set up AdamW optimiser.
    5. Optionally resume from the latest checkpoint.
    6. Loop over epochs → batches:
       a. Mixed-precision forward pass.
       b. Loss computation (CrossEntropy, ignoring PAD).
       c. Backward pass with GradScaler.
       d. Gradient clipping.
       e. Optimiser step.
       f. LR schedule update.
       g. Logging and checkpointing.
    """

    print("=" * 65)
    print("  MoD SLM — Training")
    print(f"  Device : {DEVICE.upper()}")
    print("=" * 65)

    # ---- GPU diagnostics ---------------------------------------------------
    if DEVICE == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU    : {gpu_name}")
        print(f"  VRAM   : {vram_gb:.1f} GB")
    else:
        print("  WARNING: CUDA not detected — training on CPU (very slow).")

    # ---- Load tokenizer ----------------------------------------------------
    print("\n[Train] Loading tokenizer ...")
    tokenizer = BPETokenizer()
    tokenizer.load(VOCAB_PATH, MERGES_PATH)

    # ---- Build model -------------------------------------------------------
    print("\n[Train] Building model ...")
    model = SLMModel(
        vocab_size      = VOCAB_SIZE,
        embedding_dim   = EMBEDDING_DIM,
        num_heads       = NUM_HEADS,
        num_layers      = NUM_LAYERS,
        context_length  = CONTEXT_LENGTH,
        feedforward_dim = FEEDFORWARD_DIM,
        dropout         = DROPOUT,
    ).to(DEVICE)

    count_parameters(model)

    # ---- Dataset and DataLoader -------------------------------------------
    print("\n[Train] Preparing dataset ...")
    try:
        loader, _ = create_dataloader(
            corpus_path    = CORPUS_PATH,
            tokenizer      = tokenizer,
            context_length = CONTEXT_LENGTH,
            stride         = STRIDE,
            batch_size     = BATCH_SIZE,
            shuffle        = True,
            num_workers    = NUM_WORKERS,
            pin_memory     = PIN_MEMORY and (DEVICE == 'cuda'),
        )
    except MemoryError:
        print(
            "\n[ERROR] Out of memory while building the dataset.\n"
            "  → Try reducing BATCH_SIZE or CONTEXT_LENGTH in config.py"
        )
        return

    # ---- Optimiser ---------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
        betas        = BETAS,
    )

    # ---- Loss function (ignore PAD token in loss) --------------------------
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

    # ---- Mixed precision scaler (only on CUDA) ----------------------------
    use_amp = (DEVICE == 'cuda')
    scaler  = GradScaler(enabled=use_amp)

    # ---- Checkpoint directory ----------------------------------------------
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ---- Resume from checkpoint if one exists ------------------------------
    start_epoch = 0
    global_step = 0
    best_loss   = float('inf')

    # Find the most recent epoch checkpoint
    latest_ckpt = None
    for ep in range(NUM_EPOCHS - 1, -1, -1):
        path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{ep}.pt')
        if os.path.exists(path):
            latest_ckpt = path
            break

    if latest_ckpt:
        start_epoch, global_step, best_loss = load_checkpoint(
            model, optimizer, latest_ckpt
        )
    else:
        print("[Train] No checkpoint found — starting from scratch.")

    # ---- Training log file -------------------------------------------------
    log_file = open(TRAINING_LOG, 'a', encoding='utf-8')
    log_file.write(f"\n{'='*60}\n  Training started\n{'='*60}\n")

    # ---- Main training loop ------------------------------------------------
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_loss   = 0.0
        epoch_steps  = 0
        epoch_start  = time.time()

        print(f"\n{'─'*60}")
        print(f"  Epoch {epoch + 1} / {NUM_EPOCHS}")
        print(f"{'─'*60}")

        for batch_idx, (input_ids, target_ids) in enumerate(loader):

            # Move batch to GPU
            input_ids  = input_ids.to(DEVICE,  non_blocking=True)
            target_ids = target_ids.to(DEVICE, non_blocking=True)

            # Update learning rate
            lr = get_lr(global_step, WARMUP_STEPS, LEARNING_RATE, MIN_LR)
            set_lr(optimizer, lr)

            optimizer.zero_grad(set_to_none=True)

            # ---- Forward pass with mixed precision --------------------------
            try:
                with autocast(enabled=use_amp):
                    logits = model(input_ids)          # (B, T, vocab_size)

                    # Reshape for CrossEntropyLoss: (B*T, vocab_size) and (B*T,)
                    B, T, V = logits.shape
                    loss = criterion(
                        logits.view(B * T, V),
                        target_ids.view(B * T),
                    )

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(
                        "\n[OOM ERROR] CUDA out of memory!\n"
                        "  → Reduce BATCH_SIZE in config.py (currently "
                        f"{BATCH_SIZE}).\n"
                        "  → Or reduce CONTEXT_LENGTH (currently "
                        f"{CONTEXT_LENGTH}).\n"
                        "  → Or reduce NUM_LAYERS (currently "
                        f"{NUM_LAYERS})."
                    )
                    torch.cuda.empty_cache()
                    return
                raise

            # ---- Backward pass + gradient clipping -------------------------
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            loss_val     = loss.item()
            epoch_loss  += loss_val
            epoch_steps += 1
            global_step += 1

            # ---- Periodic logging -------------------------------------------
            if global_step % LOG_EVERY_STEPS == 0:
                avg_loss = epoch_loss / epoch_steps
                msg = (
                    f"  Epoch {epoch+1:02d} | "
                    f"Step {global_step:6d} | "
                    f"LR {lr:.2e} | "
                    f"Loss {loss_val:.4f} | "
                    f"Avg {avg_loss:.4f}"
                )
                print(msg)
                log_file.write(msg + '\n')
                log_file.flush()

            # ---- Periodic sample generation --------------------------------
            if global_step % SAMPLE_EVERY_STEPS == 0:
                sample_prompt = "The Ministry of Defence shall"
                sample_text   = generate_sample(model, tokenizer, sample_prompt)
                print(f"\n  [Sample @ step {global_step}]")
                print(f"  Prompt  : {sample_prompt}")
                print(f"  Output  : {sample_text}")
                print()

        # ---- End of epoch ---------------------------------------------------
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        elapsed        = time.time() - epoch_start
        epoch_msg = (
            f"\n  ✓ Epoch {epoch+1} complete | "
            f"Avg Loss: {avg_epoch_loss:.4f} | "
            f"Time: {elapsed/60:.1f} min"
        )
        print(epoch_msg)
        log_file.write(epoch_msg + '\n')

        # ---- Save per-epoch checkpoint -------------------------------------
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pt')
        save_checkpoint(model, optimizer, epoch, global_step, best_loss, ckpt_path)
        print(f"  [Checkpoint] Saved → '{ckpt_path}'")

        # ---- Save best model -----------------------------------------------
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_checkpoint(model, optimizer, epoch, global_step, best_loss, BEST_MODEL_PATH)
            print(f"  [Best Model] New best loss {best_loss:.4f} → saved to '{BEST_MODEL_PATH}'")

    # ---- Training complete -------------------------------------------------
    log_file.write("\nTraining complete.\n")
    log_file.close()
    print("\n" + "=" * 65)
    print("  Training complete.")
    print(f"  Best loss   : {best_loss:.4f}")
    print(f"  Best model  : {BEST_MODEL_PATH}")
    print("=" * 65)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    train()
