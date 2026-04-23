# MoD-SLM: Specialized Micro Language Model

A specialized Small Language Model (SLM) trained from scratch and fine-tuned for the Indian Ministry of Defence (MoD) policy and procedural knowledge.

## Overview
This project features a custom Transformer-based language model architecture designed to run on consumer hardware (e.g., RTX 3050 4GB). It includes:
1.  **Custom Transformer**: A decoder-only architecture built from scratch in PyTorch.
2.  **Custom BPE Tokenizer**: A Byte Pair Encoding tokenizer implemented from the ground up.
3.  **GPT-2 Fine-tuning**: A secondary implementation using a pre-trained GPT-2 backbone fine-tuned on MoD data.
4.  **Interactive UI**: A Streamlit interface for querying the specialized knowledge base.

---

## Project Structure
```text
MoD_SLM/
├-- data/                       # Dataset directory
│   ├-- raw/                    # General subject corpora (Physics, History, etc.)
│   └-- processed/              # MoD-specific specialized corpora & merged file
├-- models/                     # Model weights and logs
│   ├-- custom/                 # Custom Transformer weights & tokenizer files
│   ├-- hf_fine_tuned/          # HuggingFace fine-tuned GPT-2 model
│   └-- checkpoints/            # Training checkpoints
├-- src/                        # Core source code
│   ├-- config.py               # Central configuration (Hyperparameters & Paths)
│   ├-- model.py                # Transformer architecture
│   ├-- tokenizer.py            # BPE Tokenizer implementation
│   ├-- dataset.py              # PyTorch Dataset/DataLoader
│   ├-- merge_corpus.py         # Utility to merge raw text files
│   └-- query.py                # Command-line query utility
├-- app.py                      # Streamlit Interactive Interface (Main App)
├-- train.py                    # Training script for the custom model
├-- inference.py                # Interactive inference for the custom model
├-- hf_train.py                 # Fine-tuning script for GPT-2
├-- hf_inference.py             # Inference script for the HF model
├-- requirements.txt            # Project dependencies
└-- README.md                   # You are here
```

---

## Setup & Installation

### 1. Requirements
Ensure you have Python 3.9+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
Merge the raw text files into a single training corpus:
```bash
python src/merge_corpus.py
```

### 3. Tokenizer Training (Optional)
If you wish to retrain the BPE tokenizer:
```bash
python src/tokenizer.py
```

---

## Training

### Custom Model
To train the custom Transformer model from scratch:
```bash
python train.py
```
Training logs and checkpoints will be saved in the `models/` directory.

### HuggingFace GPT-2 Fine-tuning
To fine-tune a pre-trained GPT-2 backbone on MoD data:
```bash
python hf_train.py
```

---

## Running Inference

### Interactive Streamlit App
The most user-friendly way to interact with the model:
```bash
streamlit run app.py
```

### Command-Line Inference
For the custom model:
```bash
python inference.py
```

For the HuggingFace model:
```bash
python hf_inference.py
```

---

## Configuration
All hyperparameters (embedding dimensions, layer counts, learning rates) and file paths are centralized in `src/config.py`. Modify this file to tune the model or point to different data sources.

## License
Specialized project for Ministry of Defence educational purposes.
