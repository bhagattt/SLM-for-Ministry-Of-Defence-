# FILE: hf_inference.py
"""
Hugging Face Inference Script: MoD-SLM (GPT-2 Fine-tuned)
---------------------------------------------------------
Loads the fine-tuned GPT2 model and provides an interactive interface.
"""

import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Path to the fine-tuned model
MODEL_DIR = "models/hf_fine_tuned"

def run_inference():
    print("="*60)
    print("  MoD SLM -- HF GPT-2 Inference Interface")
    print("  Note: This model uses the pre-trained GPT-2 backbone.")
    print("  Type 'quit' or 'exit' to stop.")
    print("="*60)

    # 1. Load Fine-tuned Model and Tokenizer
    print(f"[Inference] Loading from {MODEL_DIR} ...")
    
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
        model     = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    except Exception as e:
        print(f"\n[Error] Model not found: {e}")
        print("Train the model first: python hf_train.py")
        return

    # 2. Setup Device
    device = 0 if torch.cuda.is_available() else -1
    print(f"[Inference] Running on: {'CUDA (GPU)' if device == 0 else 'CPU'}")

    # 3. Create Pipeline
    generator = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        device=device
    )

    # 4. Interactive Loop
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

        # Generate output (optimized for coherence)
        output = generator(
            query, 
            max_new_tokens  = 100,
            num_return_sequences = 1,
            temperature     = 0.4,       # Lower temp for more "factual" feel
            top_k           = 50,
            top_p           = 0.95,      # Nucleus sampling for diversity
            pad_token_id    = tokenizer.eos_token_id,
            truncation      = True
        )

        # Print the NEWly generated text (strip the prompt)
        full_text = output[0]['generated_text']
        new_text  = full_text[len(query):].strip()
        
        print(f"  Model: {new_text}")

if __name__ == "__main__":
    run_inference()
