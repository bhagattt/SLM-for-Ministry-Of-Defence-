# FILE: tokenizer.py
"""
Byte Pair Encoding (BPE) tokenizer built from scratch in pure Python.

No tiktoken, no HuggingFace tokenizers -- just the raw BPE algorithm.

Special token IDs (must match config.py):
    <PAD>  ->  0
    <UNK>  ->  1
    <BOS>  ->  2
    <EOS>  ->  3
"""

import re
import json
import os
from collections import defaultdict, Counter
from src.config import (
    VOCAB_SIZE, VOCAB_PATH, MERGES_PATH,
    PAD_TOKEN_ID, UNK_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID,
)


# =============================================================================
# HELPER -- TEXT PRE-TOKENISATION
# =============================================================================

def _pre_tokenize(text: str) -> list[str]:
    """
    Split raw text into 'words' before BPE merging.

    We use a simple regex that keeps punctuation attached to words where
    it matters, and splits on whitespace.  Each resulting token is then
    stored as a space-separated sequence of individual characters so that
    the BPE algorithm can work on character pairs.

    Returns
    -------
    list[str]
        Each element is a word represented as characters joined by spaces,
        e.g. ``"hello"`` becomes ``"h e l l o"``.
    """
    # Split on whitespace, keeping non-empty tokens
    raw_words = re.findall(r'\S+', text)
    # Represent each word as a sequence of characters (BPE starting point)
    return [' '.join(list(word)) for word in raw_words]


def _get_vocab(words: list[str]) -> dict[str, int]:
    """
    Count how many times each (space-separated character sequence) word
    appears in the corpus.

    Parameters
    ----------
    words : list[str]
        Pre-tokenised words as returned by ``_pre_tokenize``.

    Returns
    -------
    dict[str, int]
        Mapping from character-sequence word to its frequency.
    """
    vocab: dict[str, int] = defaultdict(int)
    for word in words:
        vocab[word] += 1
    return dict(vocab)


def _get_pairs(vocab: dict[str, int]) -> dict[tuple[str, str], int]:
    """
    Count all adjacent character-pair frequencies across all words.

    Parameters
    ----------
    vocab : dict[str, int]
        Word-to-frequency mapping where each word is space-separated chars.

    Returns
    -------
    dict[tuple[str, str], int]
        Pair -> total count.
    """
    pairs: dict[tuple[str, str], int] = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return dict(pairs)


def _merge_vocab(pair: tuple[str, str], vocab: dict[str, int]) -> dict[str, int]:
    """
    Apply one BPE merge: replace every occurrence of ``pair`` inside every
    word with a single merged symbol.

    Parameters
    ----------
    pair : tuple[str, str]
        The character pair to merge, e.g. ``('e', 's')``.
    vocab : dict[str, int]
        Current word-to-frequency mapping.

    Returns
    -------
    dict[str, int]
        Updated vocab after the merge.
    """
    new_vocab: dict[str, int] = {}
    bigram = re.escape(' '.join(pair))
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    replacement = ''.join(pair)
    for word, freq in vocab.items():
        new_word = pattern.sub(replacement, word)
        new_vocab[new_word] = freq
    return new_vocab


# =============================================================================
# BPE TOKENIZER CLASS
# =============================================================================

class BPETokenizer:
    """
    A Byte Pair Encoding tokenizer trained completely from scratch.

    Attributes
    ----------
    vocab : dict[str, int]
        Token string -> token ID mapping (filled after training or loading).
    id_to_token : dict[int, str]
        Token ID -> token string (reverse of vocab).
    merges : list[tuple[str, str]]
        Ordered list of merge rules produced during BPE training.
    """

    # Names of the four reserved special tokens
    SPECIAL_TOKENS = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------

    def train_bpe(self, corpus_path: str, vocab_size: int = VOCAB_SIZE) -> None:
        """
        Train BPE on a plain-text corpus file.

        The algorithm:
          1. Read corpus and pre-tokenise into character sequences.
          2. Initialise the vocabulary with all unique characters + specials.
          3. Repeatedly find the most frequent adjacent pair and merge it.
          4. Stop when ``vocab_size`` is reached.

        Parameters
        ----------
        corpus_path : str
            Path to the plain-text training corpus.
        vocab_size : int
            Target vocabulary size (includes special tokens).
        """
        print(f"[Tokenizer] Reading corpus from '{corpus_path}' ...")
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(
                f"Corpus file not found: {corpus_path}\n"
                "Please run merge_corpus.py first or place the corpus at this path."
            )

        with open(corpus_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()

        print(f"[Tokenizer] Corpus size: {len(text):,} characters")

        # ---- Step 1: Pre-tokenisation ------------------------------------
        words = _pre_tokenize(text)
        word_freq = _get_vocab(words)
        print(f"[Tokenizer] Unique word types: {len(word_freq):,}")

        # ---- Step 2: Character-level initial vocabulary ------------------
        # Start vocab with special tokens at fixed IDs
        self.vocab = {}
        for tok in self.SPECIAL_TOKENS:
            self.vocab[tok] = len(self.vocab)

        # Collect all individual characters from the corpus
        char_set: set[str] = set()
        for word in word_freq:
            for ch in word.split():
                char_set.add(ch)

        for ch in sorted(char_set):
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)

        print(f"[Tokenizer] Initial char vocab size: {len(self.vocab)}")
        print(f"[Tokenizer] Training BPE to vocab size {vocab_size} ...")

        # ---- Step 3: BPE merge loop --------------------------------------
        self.merges = []
        num_merges = vocab_size - len(self.vocab)

        for i in range(num_merges):
            pairs = _get_pairs(word_freq)
            if not pairs:
                print("[Tokenizer] No more pairs to merge. Stopping early.")
                break

            # Pick the most frequent pair (ties broken alphabetically for reproducibility)
            best_pair = max(pairs, key=lambda p: (pairs[p], p))
            word_freq = _merge_vocab(best_pair, word_freq)

            merged_token = ''.join(best_pair)
            self.merges.append(best_pair)
            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)

            if (i + 1) % 500 == 0:
                print(f"  Merge {i+1}/{num_merges}  vocab_size={len(self.vocab)}")

        # ---- Step 4: Build reverse mapping --------------------------------
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        print(f"[Tokenizer] Final vocab size: {len(self.vocab)}")

    # ------------------------------------------------------------------
    # ENCODING
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """
        Encode a string into a list of token IDs using the learned merges.

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        list[int]
            Token IDs corresponding to the encoded text.
        """
        if not self.vocab:
            raise RuntimeError("Tokenizer is not trained/loaded. Call train_bpe() or load() first.")

        words = _pre_tokenize(text)
        token_ids: list[int] = []

        for word_chars in words:
            # Start with the characters split by spaces
            symbols = word_chars.split()

            # Apply each merge rule in order
            for pair in self.merges:
                new_symbols: list[str] = []
                i = 0
                while i < len(symbols):
                    if (i < len(symbols) - 1
                            and symbols[i] == pair[0]
                            and symbols[i + 1] == pair[1]):
                        new_symbols.append(pair[0] + pair[1])
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                symbols = new_symbols

            # Map each symbol to its token ID
            for sym in symbols:
                token_ids.append(self.vocab.get(sym, UNK_TOKEN_ID))

        return token_ids

    # ------------------------------------------------------------------
    # DECODING
    # ------------------------------------------------------------------

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a list of token IDs back to a human-readable string.

        Special tokens (<PAD>, <BOS>, <EOS>) are stripped from the output.
        <UNK> tokens are replaced with the replacement character ``\ufffd``.

        Parameters
        ----------
        token_ids : list[int]
            Sequence of token IDs to decode.

        Returns
        -------
        str
            Decoded text.
        """
        skip_ids = {PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID}
        tokens: list[str] = []

        for tid in token_ids:
            if tid in skip_ids:
                continue
            if tid == UNK_TOKEN_ID:
                tokens.append('\ufffd')
            else:
                tokens.append(self.id_to_token.get(tid, '\ufffd'))

        # Re-join -- BPE tokens that came from the same word are adjacent
        # and have no internal spaces; we just join and then normalise spaces.
        raw = ' '.join(tokens)

        # Collapse multiple spaces into one and strip leading/trailing space
        text = re.sub(r' +', ' ', raw).strip()
        return text

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------

    def save(self, vocab_path: str = VOCAB_PATH, merges_path: str = MERGES_PATH) -> None:
        """
        Persist the trained tokenizer to disk.

        Creates two files:
          - ``vocab_path``: JSON mapping token string -> token ID.
          - ``merges_path``: Plain text, one merge rule per line.

        Parameters
        ----------
        vocab_path : str
            Destination path for the vocabulary JSON file.
        merges_path : str
            Destination path for the merges text file.
        """
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        with open(merges_path, 'w', encoding='utf-8') as f:
            for a, b in self.merges:
                f.write(f"{a} {b}\n")

        print(f"[Tokenizer] Saved vocab  -> '{vocab_path}'")
        print(f"[Tokenizer] Saved merges -> '{merges_path}'")

    def load(self, vocab_path: str = VOCAB_PATH, merges_path: str = MERGES_PATH) -> None:
        """
        Load a previously trained tokenizer from disk.

        Parameters
        ----------
        vocab_path : str
            Path to the vocabulary JSON file.
        merges_path : str
            Path to the merges text file.

        Raises
        ------
        FileNotFoundError
            If either file is missing.
        """
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(
                f"Vocabulary file not found: '{vocab_path}'\n"
                "Run: python tokenizer.py  (to train the tokenizer first)"
            )
        if not os.path.exists(merges_path):
            raise FileNotFoundError(
                f"Merges file not found: '{merges_path}'\n"
                "Run: python tokenizer.py  (to train the tokenizer first)"
            )

        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        self.id_to_token = {int(v): k for k, v in self.vocab.items()}

        self.merges = []
        with open(merges_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        self.merges.append((parts[0], parts[1]))

        print(f"[Tokenizer] Loaded vocab  ({len(self.vocab):,} tokens) from '{vocab_path}'")
        print(f"[Tokenizer] Loaded merges ({len(self.merges):,} rules)  from '{merges_path}'")

    # ------------------------------------------------------------------
    # CONVENIENCE PROPERTIES
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        return len(self.vocab)

    def __len__(self) -> int:
        return len(self.vocab)


# =============================================================================
# CONVENIENCE WRAPPERS (module-level functions used by other scripts)
# =============================================================================

_tokenizer: BPETokenizer | None = None


def _get_tokenizer() -> BPETokenizer:
    """Return the module-level singleton tokenizer (must be loaded first)."""
    if _tokenizer is None:
        raise RuntimeError(
            "Module-level tokenizer is not initialised.\n"
            "Call load() before using encode() / decode()."
        )
    return _tokenizer


def train_bpe(corpus_path: str, vocab_size: int = VOCAB_SIZE) -> BPETokenizer:
    """
    Train a new BPE tokenizer and return it.

    Also sets the module-level singleton so that ``encode`` / ``decode``
    work immediately after training.

    Parameters
    ----------
    corpus_path : str
        Path to the plain-text training corpus.
    vocab_size : int
        Target vocabulary size.

    Returns
    -------
    BPETokenizer
        Trained tokenizer instance.
    """
    global _tokenizer
    _tokenizer = BPETokenizer()
    _tokenizer.train_bpe(corpus_path, vocab_size)
    return _tokenizer


def encode(text: str) -> list[int]:
    """Encode text using the module-level singleton tokenizer."""
    return _get_tokenizer().encode(text)


def decode(token_ids: list[int]) -> str:
    """Decode token IDs using the module-level singleton tokenizer."""
    return _get_tokenizer().decode(token_ids)


def save(vocab_path: str = VOCAB_PATH, merges_path: str = MERGES_PATH) -> None:
    """Save the module-level singleton tokenizer to disk."""
    _get_tokenizer().save(vocab_path, merges_path)


def load(vocab_path: str = VOCAB_PATH, merges_path: str = MERGES_PATH) -> None:
    """Load a tokenizer from disk into the module-level singleton."""
    global _tokenizer
    _tokenizer = BPETokenizer()
    _tokenizer.load(vocab_path, merges_path)


# =============================================================================
# ENTRY POINT -- run as a script to train and save the tokenizer
# =============================================================================

if __name__ == '__main__':
    import sys
    from src.config import CORPUS_PATH, VOCAB_SIZE, VOCAB_PATH, MERGES_PATH

    corpus = sys.argv[1] if len(sys.argv) > 1 else CORPUS_PATH

    print("=" * 60)
    print("  MoD SLM -- BPE Tokenizer Training")
    print("=" * 60)

    tok = train_bpe(corpus, VOCAB_SIZE)
    tok.save(VOCAB_PATH, MERGES_PATH)

    # Quick sanity check
    sample = "The Ministry of Defence allocates budget under capital expenditure."
    ids = tok.encode(sample)
    recovered = tok.decode(ids)
    print(f"\n[Sanity check]")
    print(f"  Input   : {sample}")
    print(f"  Token IDs ({len(ids)}): {ids[:20]} ...")
    print(f"  Decoded : {recovered}")
    print("\n[Tokenizer] Training complete.")
