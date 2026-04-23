# TECHNICAL REPORT: MoD-SLM (Stage 1)
## Project: Small Language Model for Ministry of Defence

### 1. Overview
The MoD-SLM is a custom-built, decoder-only Transformer model designed specifically for low-VRAM environments (NVIDIA RTX 3050 4GB). Unlike off-the-shelf models, every component--from the BPE Tokenizer to the Transformer Attention blocks--was implemented from scratch in pure PyTorch.

---

### 2. Architecture Specifications
| Component | Specification |
| :--- | :--- |
| **Model Type** | Decoder-Only Transformer (GPT-Style) |
| **Parameters** | 6.85 Million |
| **Embedding Dimension** | 256 |
| **Heads (Multi-Head Attn)** | 8 (32 dims per head) |
| **Transformer Layers** | 6 |
| **Feed-Forward Dimension** | 1024 |
| **Context Length** | 256 Tokens |
| **Vocabulary Size** | 10,000 (Custom BPE) |

---

### 3. Training Infrastructure
- **Hardware**: NVIDIA RTX 3050 Laptop GPU (4.3 GB VRAM)
- **Mixed Precision**: `torch.cuda.amp` (FP16) enabled for 35% faster training and 50% lower VRAM.
- **Gradient Clipping**: Scaled to `1.0` for stability.
- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.95$, Weight Decay=0.01).
- **Scheduler**: Cosine Annealing with Warmup (100 steps).

---

### 4. Corpus Composition
The model was trained on a multi-domain corpus (~653,000 characters), including:
- **MoD Original**: Policy circulars, Leave Rules, and Audit reports.
- **MoD Synthetic**: Agnipath scheme details, Naval/Army/Air Force hierarchy, weaponry (Rafale, INS Vikrant), and CDS/DMA duties.
- **General Knowledge**: AI, Physics, Biology, Chemistry, Space, and History.

---

### 5. Training Results
- **Final Training Loss**: 1.4624
- **Training Duration**: 50 Epochs (Full Convergence)
- **Convergence Rate**: Loss reduced from ~8.85 (random) to 1.46 (context-aware).

---

### 6. Key Features
- **Weight Tying**: Shares weights between embedding and output layers to reduce param count.
- **Pre-LayerNorm**: Improved gradient flow for deep structures.
- **Causal Masking**: Ensures the model only predicts the "Next Token" based on the past.
- **Interactive Inference**: Real-time terminal interface for model testing.

---

### 7. Deployment Instructions
1. **Inference**: Use `python inference.py` for interactive querying.
2. **Batch Query**: Use `python query.py "Prompt here"` for quick testing.
3. **Hardware Requirement**: Min 2GB VRAM (GPU) or 4GB RAM (CPU).
