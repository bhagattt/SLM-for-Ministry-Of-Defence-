# FILE: dataset.py
"""
PyTorch Dataset and DataLoader for language model training.

Reads the merged MoD corpus, tokenises it with the BPE tokenizer, and
produces (input, target) pairs using a sliding window approach.

    input  = tokens[i   : i + context_length]
    target = tokens[i+1 : i + context_length + 1]

This implements standard *next-token prediction* (teacher forcing).
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    CORPUS_PATH, CONTEXT_LENGTH, STRIDE,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    BOS_TOKEN_ID, EOS_TOKEN_ID,
)
from tokenizer import BPETokenizer


# =============================================================================
# DATASET CLASS
# =============================================================================

class MoDCorpusDataset(Dataset):
    """
    Sliding-window language model dataset for the MoD SLM corpus.

    The full tokenised corpus is stored in memory as a single flat list of
    token IDs.  Each call to ``__getitem__`` returns a window of
    ``context_length`` tokens as the input and the same window shifted right
    by one as the target.

    Parameters
    ----------
    corpus_path : str
        Path to the plain-text corpus file.
    tokenizer : BPETokenizer
        A trained (or loaded) tokenizer instance.
    context_length : int
        Number of tokens per training sample.
    stride : int
        Step size between consecutive windows.  Use ``context_length`` for
        no overlap, or a smaller value for data augmentation via overlap.
    """

    def __init__(
        self,
        corpus_path: str,
        tokenizer: BPETokenizer,
        context_length: int = CONTEXT_LENGTH,
        stride: int         = STRIDE,
    ) -> None:
        super().__init__()

        self.context_length = context_length
        self.stride         = stride

        # ---- Read corpus ----------------------------------------------------
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(
                f"Corpus file not found: '{corpus_path}'\n"
                "Make sure merge_corpus.py has been run first."
            )

        print(f"[Dataset] Reading corpus from '{corpus_path}' ...")
        with open(corpus_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()

        print(f"[Dataset] Corpus size: {len(text):,} characters")

        # ---- Tokenise -------------------------------------------------------
        print("[Dataset] Tokenising corpus (this may take a minute) ...")
        # Prepend BOS and append EOS for the entire document
        self.tokens: list[int] = (
            [BOS_TOKEN_ID] + tokenizer.encode(text) + [EOS_TOKEN_ID]
        )
        print(f"[Dataset] Total tokens in corpus: {len(self.tokens):,}")

        # ---- Compute valid starting indices ---------------------------------
        # We need context_length input tokens plus one more for the target.
        max_start = len(self.tokens) - context_length - 1
        if max_start < 0:
            raise ValueError(
                "Corpus is too small for the configured context_length. "
                "Use a larger corpus or reduce CONTEXT_LENGTH in config.py."
            )

        self.start_indices: list[int] = list(range(0, max_start + 1, stride))
        print(f"[Dataset] Total training samples: {len(self.start_indices):,}")

    def __len__(self) -> int:
        """Return the total number of training samples."""
        return len(self.start_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve one (input, target) pair.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``input_ids``  — shape ``(context_length,)``
            ``target_ids`` — shape ``(context_length,)``
            where target_ids[i] == input_ids[i + 1].
        """
        start = self.start_indices[idx]
        end   = start + self.context_length

        input_ids  = self.tokens[start     : end    ]
        target_ids = self.tokens[start + 1 : end + 1]

        return (
            torch.tensor(input_ids,  dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


# =============================================================================
# DATALOADER FACTORY
# =============================================================================

def create_dataloader(
    corpus_path: str    = CORPUS_PATH,
    tokenizer: BPETokenizer | None = None,
    context_length: int = CONTEXT_LENGTH,
    stride: int         = STRIDE,
    batch_size: int     = BATCH_SIZE,
    shuffle: bool       = True,
    num_workers: int    = NUM_WORKERS,
    pin_memory: bool    = PIN_MEMORY,
) -> tuple[DataLoader, MoDCorpusDataset]:
    """
    Build and return a DataLoader and the underlying Dataset.

    Parameters
    ----------
    corpus_path : str
        Path to the plain-text corpus.
    tokenizer : BPETokenizer or None
        Trained tokenizer; if ``None``, one will be loaded from the default
        ``VOCAB_PATH`` / ``MERGES_PATH`` locations.
    context_length : int
        Tokens per sample.
    stride : int
        Sliding window step size.
    batch_size : int
        Samples per mini-batch.
    shuffle : bool
        Whether to shuffle the dataset each epoch.
    num_workers : int
        Number of DataLoader worker processes.
    pin_memory : bool
        Whether to pin memory for faster GPU transfer.

    Returns
    -------
    tuple[DataLoader, MoDCorpusDataset]
        The DataLoader and the underlying dataset object.
    """
    if tokenizer is None:
        from config import VOCAB_PATH, MERGES_PATH
        tokenizer = BPETokenizer()
        tokenizer.load(VOCAB_PATH, MERGES_PATH)

    dataset = MoDCorpusDataset(corpus_path, tokenizer, context_length, stride)

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        # Drop the last incomplete batch for training stability
        drop_last   = True,
    )

    print(f"[DataLoader] batch_size={batch_size} | "
          f"batches_per_epoch={len(loader):,} | "
          f"num_workers={num_workers}")
    return loader, dataset


# =============================================================================
# QUICK TEST — run as a script
# =============================================================================

if __name__ == '__main__':
    from tokenizer import BPETokenizer
    from config import VOCAB_PATH, MERGES_PATH, CORPUS_PATH

    print("Loading tokenizer ...")
    tok = BPETokenizer()
    tok.load(VOCAB_PATH, MERGES_PATH)

    print("Creating DataLoader ...")
    loader, ds = create_dataloader(CORPUS_PATH, tok)

    # Inspect one batch
    inputs, targets = next(iter(loader))
    print(f"Input batch shape : {inputs.shape}")
    print(f"Target batch shape: {targets.shape}")
    print(f"Sample input[0]   : {inputs[0, :10].tolist()} ...")
    print(f"Sample target[0]  : {targets[0, :10].tolist()} ...")
    print("Dataset test passed.")
