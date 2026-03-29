# FILE: test_model.py
"""
Automated evaluation script for the MoD SLM.

Takes known snippets from the training corpus, uses the first half as a 
prompt, and compares the model's generation with the actual text.
"""

import torch
from config import (
    DEVICE, VOCAB_PATH, MERGES_PATH, BEST_MODEL_PATH,
    VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS,
    CONTEXT_LENGTH, FEEDFORWARD_DIM, DROPOUT, 
    BOS_TOKEN_ID, TEMPERATURE, TOP_K
)
from tokenizer import BPETokenizer
from model import SLMModel

# Test cases: (Prompt, Expected Continuation)
# We test MoD (Specific), Physics, AI, and History to see the SLM's range.
TEST_CASES = [
    # 1. MoD (The flagship)
    (
        "The Chief of Defence Staff (CDS) office ensures",
        "the convergence of military strategic thought and procurement"
    ),
    # 2. MoD (Agnipath)
    (
        "Agniveers are recruited for a period of",
        "four years, after which up to 25 percent"
    ),
    # 3. Physics (Newton)
    (
        "Newton's first law states that an object at rest",
        "stays at rest, and an object in motion continues"
    ),
    # 4. History
    (
        "The historical analysis suggests that most empires",
        "confronted significant administrative challenges"
    ),
    # 5. Artificial Intelligence
    (
        "Neural networks are inspired by the structure of the",
        "human brain and consist of interconnected nodes"
    ),
    # 6. MoD (Navy)
    (
        "INS Vikrant is India's first indigenously built",
        "aircraft carrier"
    )
]

def run_tests():
    print("="*60)
    print("  MoD SLM — Automated Corpus Testing")
    print("="*60)

    # 1. Load Tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(VOCAB_PATH, MERGES_PATH)

    # 2. Load Model
    model = SLMModel(
        vocab_size      = VOCAB_SIZE,
        embedding_dim   = EMBEDDING_DIM,
        num_heads       = NUM_HEADS,
        num_layers      = NUM_LAYERS,
        context_length  = CONTEXT_LENGTH,
        feedforward_dim = FEEDFORWARD_DIM,
        dropout         = DROPOUT,
    )
    
    try:
        ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded model from {BEST_MODEL_PATH} (Epoch {ckpt['epoch']}, Loss {ckpt['best_loss']:.4f})")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model = model.to(DEVICE)
    model.eval()

    # 3. Run Samples
    for i, (prompt, expected) in enumerate(TEST_CASES):
        print(f"\nTEST {i+1}:")
        print(f"  Prompt   : {prompt}")
        print(f"  Expected : ... {expected}")
        
        # Tokenize and generate
        input_ids = [BOS_TOKEN_ID] + tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_tensor, 
                max_new_tokens=25, 
                temperature=0.4, # Lower temp for more deterministic testing
                top_k=20
            )
        
        # Decode only the NEW parts
        new_ids = output_ids[0, len(input_ids):].tolist()
        generated_text = tokenizer.decode(new_ids)
        
        print(f"  Model    : {generated_text}")
        
        # Simple overlap check
        words_expected = set(expected.lower().replace(',', '').replace('.', '').split())
        words_generated = set(generated_text.lower().replace(',', '').replace('.', '').split())
        overlap = words_expected.intersection(words_generated)
        
        if overlap:
            print(f"  [Match] Found keywords: {', '.join(overlap)}")
        else:
            print("  [No direct keyword match]")
        print("-" * 40)

    print("\n" + "="*60)
    print("  Testing Complete.")
    print("="*60)

if __name__ == "__main__":
    run_tests()
