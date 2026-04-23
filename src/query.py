# FILE: query.py
import torch
import sys
from src.config import (
    DEVICE, VOCAB_PATH, MERGES_PATH, BEST_MODEL_PATH,
    VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS,
    CONTEXT_LENGTH, FEEDFORWARD_DIM, DROPOUT, 
    BOS_TOKEN_ID
)
from src.tokenizer import BPETokenizer
from src.model import SLMModel

def query(prompt):
    tokenizer = BPETokenizer()
    tokenizer.load(VOCAB_PATH, MERGES_PATH)
    model = SLMModel(VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, CONTEXT_LENGTH, FEEDFORWARD_DIM, DROPOUT)
    ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE).eval()
    
    input_ids = [BOS_TOKEN_ID] + tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], device=DEVICE)
    with torch.no_grad():
        out = model.generate(input_tensor, max_new_tokens=50, temperature=0.3, top_k=5)
    
    print(f"\nPrompt: {prompt}")
    print(f"SLM Response: {tokenizer.decode(out[0, len(input_ids):].tolist())}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query(" ".join(sys.argv[1:]))
