# MoD-MLM (Micro Language Model) - Complete Guide

## 1. Project Introduction
This project is building a **Micro Language Model (MLM)** from the ground up using pure **PyTorch**. It is designed specifically for the **Indian Ministry of Defence (MoD)** and optimized for low-resource environments.

No HuggingFace, no pre-trained weights--everything from the architecture to the Byte Pair Encoding (BPE) tokenizer was developed from scratch.

---

## 2. File Structure & Components
Each file in this repository has a specific, isolated role:

| File | Role | Description |
| :--- | :--- | :--- |
| `config.py` | **Heart of the Project** | All hyperparameters (Layers, Heads, Dim, LR, Epochs) and file paths. Change values here to affect the whole project. |
| `tokenizer.py` | **BPE Engine** | Custom Byte Pair Encoding implementation. Trains on your corpus and converts text ↔ tokens. |
| `model.py` | **Transformer Core** | The GPT-style decoder-only Transformer with Multi-Head Attention, Pre-LayerNorm, and Weight Tying. |
| `dataset.py` | **Data Pipeline** | Handles sliding-window tokenization and creates the PyTorch `DataLoader` for batch training. |
| `train.py` | **Training Loop** | Manages the training process, mixed precision (AMP), gradient clipping, and checkpoint saving. |
| `inference.py` | **Interactive App** | A terminal-based interface to "talk" to the trained model. |
| `query.py` | **One-Shot Tester** | A quick script to get a model response for a single prompt via CLI. |
| `merge_corpus.py` | **Data Processor** | Combines multiple raw text files into a single master corpus for training. |
| `test_model.py` | **Logic Verifier** | Compares model output against known snippets from the training data. |

---

## 3. Technical Specifications
### Model Architecture (7M Parameters)
- **Type**: Causal Language Model (Transformer Decoder)
- **Embedding Dimension**: 256
- **Attention Heads**: 8 (32 dims per head)
- **Transformer Blocks**: 6 Layers
- **Context Length**: 256 Tokens (Sliding window enabled)
- **Vocabulary Support**: 10,000 unique tokens (BPE)
- **Weight Tying**: Shared weights between Input Embeddings and Output Linear Head.

### Training Strategy
- **Mixed Precision (AMP)**: Uses `float16` for faster math and lower VRAM usage.
- **AdamW Optimizer**: Better weight decay implementation for model stability.
- **Cosine LR Scheduler**: Automatically lowers learning rate as loss plateaus.
- **Dropout (0.1)**: Prevents the model from memorizing text (overfitting).

---

## 4. How to Use & Train
### Step 1: Prepare Data
Place your `.txt` files in the directory and list them in `config.py`. Run:
```bash
python merge_corpus.py
```

### Step 2: Train Tokenizer
Ensure your `VOCAB_SIZE` is set, then run:
```bash
python tokenizer.py
```

### Step 3: Train Model
Start the training process. The model will auto-detect your GPU and use AMP:
```bash
python train.py
```

### Step 4: Run Inference
Talk to your trained SLM:
```bash
python inference.py
```

---

## 5. Training Results (Stage 1)
- **Total Training Tokens**: ~150,000 (from 650k characters)
- **VRAM Usage**: ~1.8 GB (Ideal for 4GB RTX 3050)
- **Initial Loss**: 8.85
- **Final Loss (50 Epochs)**: **1.4624**
- **Convergence Time**: Very fast (approx. 10-15 mins on GPU)

---

## 6. Project Philosophy
- **Privacy First**: No internet needed. All training and inference is local.
- **Efficiency**: State-of-the-art results within extreme hardware constraints.
- **Customization**: The BPE tokenizer is trained *only* on the MoD and Technical data, making it much more efficient at "Defence Speak" than general-purpose LLMs.

---

## 7. Project Architecture Flow
```mermaid
graph TD
    A[Raw MoD Corpus (.txt)] --> B(merge_corpus.py)
    B --> C[mod_slm_stage1_corpus.txt]
    C --> D{Choose Strategy}
    
    %% Scratch Pathway
    D -->|From Scratch| E(tokenizer.py - BPE)
    E --> F(train.py - 7M MLM)
    F --> G[best_model.pt]
    G --> H(inference.py / query.py)
    
    %% HF Pathway
    D -->|Fine-Tuning| I(hf_train.py - GPT2-124M)
    I --> J[hf_mod_model/]
    J --> K(hf_inference.py / hf_query.py)
    
    %% App Layer
    H --> L[app.py - Streamlit UI]
    K --> L
    
    subgraph "Inference Layer"
    L
    end
```

### *Generated for the Indian Ministry of Defence MLM Project*
