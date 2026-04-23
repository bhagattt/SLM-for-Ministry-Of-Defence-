# FILE: merge_corpus.py
"""
Utility script that merges all MoD corpus text files into one file.

Run this ONCE before training the tokenizer or the model.

Usage
-----
    python merge_corpus.py
"""

import os
from src.config import CORPUS_FILES, CORPUS_PATH


def merge_corpus(
    input_files: list[str] = CORPUS_FILES,
    output_path: str       = CORPUS_PATH,
) -> None:
    """
    Concatenate all corpus text files into a single merged file.

    Each source file is separated by two newlines in the output so that
    the BPE tokenizer does not accidentally stitch the end of one document
    to the start of the next.

    Parameters
    ----------
    input_files : list[str]
        List of paths to the raw corpus text files.
    output_path : str
        Destination path for the merged corpus.
    """
    print(f"Merging {len(input_files)} corpus files -> '{output_path}'")
    total_chars = 0

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for path in input_files:
            if not os.path.exists(path):
                print(f"  [SKIP] Not found: '{path}'")
                continue

            with open(path, 'r', encoding='utf-8', errors='replace') as in_f:
                text = in_f.read().strip()

            char_count   = len(text)
            total_chars += char_count
            print(f"  [OK]   '{path}'  ({char_count:,} chars)")

            out_f.write(text)
            out_f.write('\n\n')   # Document boundary

    print(f"\nMerge complete -- total characters: {total_chars:,}")
    print(f"Saved to: '{output_path}'")


if __name__ == '__main__':
    merge_corpus()
