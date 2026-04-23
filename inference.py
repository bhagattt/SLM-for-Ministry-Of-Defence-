# FILE: inference.py
"""
Interactive inference script for the MoD SLM.

Loads the best trained model and runs an interactive terminal loop where
the user can type prompts and receive generated text.

Usage
-----
    python inference.py

Type "quit" (or "exit") to stop.
"""

import sys
import torch

from src.config import (
    DEVICE,
    VOCAB_PATH, MERGES_PATH, BEST_MODEL_PATH,
    # Model architecture (must match what was trained)
    VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS,
    CONTEXT_LENGTH, FEEDFORWARD_DIM, DROPOUT,
    # Generation defaults
    MAX_NEW_TOKENS, TEMPERATURE, TOP_K,
    # Special tokens
    BOS_TOKEN_ID,
)
from src.tokenizer import BPETokenizer
from src.model import SLMModel


# =============================================================================
# LOAD HELPERS
# =============================================================================

def load_tokenizer() -> BPETokenizer:
    """
    Load the trained BPE tokenizer from disk.

    Returns
    -------
    BPETokenizer
        Ready-to-use tokenizer.

    Raises
    ------
    FileNotFoundError
        If tokenizer files are missing.
    """
    tokenizer = BPETokenizer()
    tokenizer.load(VOCAB_PATH, MERGES_PATH)
    return tokenizer


def load_model(tokenizer: BPETokenizer) -> SLMModel:
    """
    Load the trained SLM from ``best_model.pt``.

    Parameters
    ----------
    tokenizer : BPETokenizer
        Used to verify the vocabulary size matches the checkpoint.

    Returns
    -------
    SLMModel
        Model in evaluation mode, moved to the appropriate device.

    Raises
    ------
    FileNotFoundError
        If ``best_model.pt`` does not exist.
    RuntimeError
        If the checkpoint cannot be loaded.
    """
    import os
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at '{BEST_MODEL_PATH}'.\n"
            "Train the model first:  python train.py"
        )

    print(f"[Inference] Loading model from '{BEST_MODEL_PATH}' ...")

    # Build the model architecture
    model = SLMModel(
        vocab_size      = VOCAB_SIZE,
        embedding_dim   = EMBEDDING_DIM,
        num_heads       = NUM_HEADS,
        num_layers      = NUM_LAYERS,
        context_length  = CONTEXT_LENGTH,
        feedforward_dim = FEEDFORWARD_DIM,
        dropout         = DROPOUT,
    )

    # Load checkpoint
    ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])

    model = model.to(DEVICE)
    model.eval()

    epoch     = ckpt.get('epoch', '?')
    best_loss = ckpt.get('best_loss', float('nan'))
    print(f"[Inference] Checkpoint: epoch {epoch} | best_loss {best_loss:.4f}")
    print(f"[Inference] Running on: {DEVICE.upper()}")

    return model


# =============================================================================
# GENERATE RESPONSE
# =============================================================================

@torch.no_grad()
def generate(
    model: SLMModel,
    tokenizer: BPETokenizer,
    query: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float  = TEMPERATURE,
    top_k: int          = TOP_K,
) -> str:
    """
    Encode a query, generate a response, and decode it.

    The BOS token is prepended to the encoded prompt so that the model
    knows it is at the start of a sequence.

    Parameters
    ----------
    model : SLMModel
        Trained model in eval mode.
    tokenizer : BPETokenizer
        Tokenizer to encode/decode.
    query : str
        User's input text prompt.
    max_new_tokens : int
        Maximum number of new tokens to generate.
    temperature : float
        Sampling temperature in (0, inf).
        Lower -> more focused/deterministic.
        Higher -> more random/creative.
    top_k : int
        Top-k vocabulary cut-off for sampling.

    Returns
    -------
    str
        The generated continuation only (input query is not repeated).
    """
    # Encode query with BOS token prepended
    prompt_ids = [BOS_TOKEN_ID] + tokenizer.encode(query)
    prompt_len = len(prompt_ids)

    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

    # Truncate prompt if longer than context_length - 1 (need at least 1 new token)
    max_prompt = CONTEXT_LENGTH - 1
    if input_tensor.shape[1] > max_prompt:
        input_tensor = input_tensor[:, -max_prompt:]
        prompt_len   = input_tensor.shape[1]

    # Generate
    output_ids = model.generate(
        input_tensor,
        max_new_tokens = max_new_tokens,
        temperature    = temperature,
        top_k          = top_k,
    )

    # Extract only the newly generated token IDs (strip the prompt)
    generated_ids = output_ids[0, prompt_len:].tolist()

    return tokenizer.decode(generated_ids)


# =============================================================================
# INTERACTIVE LOOP
# =============================================================================

def run_interactive(model: SLMModel, tokenizer: BPETokenizer) -> None:
    """
    Run an interactive question-answering loop in the terminal.

    The user types a prompt and receives model-generated text.
    Typing ``quit`` or ``exit`` ends the session.

    Parameters
    ----------
    model : SLMModel
        Trained model in eval mode.
    tokenizer : BPETokenizer
        Tokenizer for encoding / decoding.
    """
    print("\n" + "=" * 65)
    print("  MoD SLM -- Interactive Inference")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 65)

    while True:
        try:
            query = input("\n> You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Inference] Session ended.")
            break

        if not query:
            continue

        if query.lower() in ('quit', 'exit', 'q'):
            print("[Inference] Goodbye.")
            break

        print("\n  Generating ...\n")

        try:
            response = generate(model, tokenizer, query)
            print(f"  Model: {response}")
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print("[OOM] CUDA out of memory. Try a shorter query.")
                torch.cuda.empty_cache()
            else:
                print(f"[Error] {e}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # ---- Load tokenizer ----------------------------------------------------
    try:
        tokenizer = load_tokenizer()
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        sys.exit(1)

    # ---- Load model --------------------------------------------------------
    try:
        model = load_model(tokenizer)
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] Failed to load model: {e}")
        sys.exit(1)

    # ---- Run interactive loop -----------------------------------------------
    run_interactive(model, tokenizer)
