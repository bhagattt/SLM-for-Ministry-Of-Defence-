# FILE: hf_query.py
import torch
import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

MODEL_DIR = "./hf_mod_model"

def query(prompt):
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    model     = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    device = 0 if torch.cuda.is_available() else -1
    
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    
    out = generator(
        prompt, 
        max_new_tokens=60, 
        temperature=0.2, 
        top_k=40, 
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    
    print(f"\nPrompt: {prompt}")
    print(f"HF-SLM Response: {out[0]['generated_text'][len(prompt):].strip()}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query(" ".join(sys.argv[1:]))
