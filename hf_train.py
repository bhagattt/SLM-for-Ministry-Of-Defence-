# FILE: hf_train.py
"""
Hugging Face Fine-tuning Script: MoD-SLM (GPT-2)
-----------------------------------------------
This script fine-tunes a pre-trained GPT-2 model (base) on the MoD corpus.
Optimized for RTX 3050 (4GB VRAM) via Mixed Precision (FP16).
"""

import os
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AutoConfig
)

# Paths
CORPUS_PATH     = "data/processed/mod_slm_stage1_corpus.txt"
OUTPUT_DIR      = "models/hf_fine_tuned"
LOGGING_DIR     = "models/logs"

def train():
    print("="*60)
    print("  MoD SLM -- GPT-2 Fine-Tuning (Hugging Face)")
    print("="*60)

    # 1. Load Pre-trained GPT2 Model and Tokenizer
    model_name = "gpt2"
    print(f"[1/4] Loading {model_name} backbone ...")
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model     = GPT2LMHeadModel.from_pretrained(model_name)

    # Set pad_token to eos_token since GPT2 doesn't have a pad_token
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Prepare Dataset
    print(f"[2/4] Preparing dataset from {CORPUS_PATH} ...")
    
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = CORPUS_PATH,
        block_size = 128  # Safe length for 4GB VRAM
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer, 
        mlm       = False
    )

    # 3. Training Arguments (RTX 3050 Optimized)
    print("[3/4] Configuring Trainer (Optimized for 4GB VRAM) ...")
    
    training_args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        overwrite_output_dir        = True,
        num_train_epochs            = 10,           # More epochs since we have a pre-trained backbone
        per_device_train_batch_size = 4,            # Smaller batch for 4GB VRAM
        save_steps                  = 500,
        save_total_limit            = 2,
        logging_steps               = 50,
        fp16                        = True,         # Use FP16 for speed/VRAM
        learning_rate               = 5e-5,         # standard fine-tuning LR
        weight_decay                = 0.01,
        logging_dir                 = LOGGING_DIR,
        report_to                   = "none"        # Disable WandB/Tensorboard for now
    )

    # 4. Run Training
    print("[4/4] Starting training ...")
    
    trainer = Trainer(
        model           = model,
        args            = training_args,
        data_collator   = data_collator,
        train_dataset   = dataset,
    )

    trainer.train()

    # Save and finalize
    print(f"\n[OK] Fine-tuning complete! Model saved to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train()
