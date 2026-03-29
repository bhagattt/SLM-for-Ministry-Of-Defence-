# FILE: model.py
"""
Decoder-only Transformer Language Model (GPT-style) built from scratch
in pure PyTorch.

Architecture summary (tuned for RTX 3050 4 GB VRAM):
    embedding_dim  = 256
    num_heads      = 8   (head_dim = 32)
    num_layers     = 6
    context_length = 256
    feedforward    = 1024
    vocab_size     = 8000
    params         ≈ 15–25 M
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS,
    CONTEXT_LENGTH, FEEDFORWARD_DIM, DROPOUT,
    BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID,
    DEVICE,
)


# =============================================================================
# CLASS 1 — MULTI-HEAD SELF-ATTENTION
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Scaled dot-product multi-head self-attention with causal masking.

    Each head computes:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V

    The causal mask ensures that position *i* can only attend to positions
    ≤ *i*, which is essential for autoregressive language modelling.

    Parameters
    ----------
    embedding_dim : int
        Model hidden size (must be divisible by num_heads).
    num_heads : int
        Number of parallel attention heads.
    dropout : float
        Dropout probability applied to attention weights.
    context_length : int
        Maximum sequence length; used to pre-build the causal mask.
    """

    def __init__(
        self,
        embedding_dim: int  = EMBEDDING_DIM,
        num_heads: int      = NUM_HEADS,
        dropout: float      = DROPOUT,
        context_length: int = CONTEXT_LENGTH,
    ) -> None:
        super().__init__()

        assert embedding_dim % num_heads == 0, (
            f"embedding_dim ({embedding_dim}) must be divisible by "
            f"num_heads ({num_heads})"
        )

        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.head_dim      = embedding_dim // num_heads  # dimension per head

        # Query, Key, Value projections (combined for efficiency)
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Output projection that merges all heads
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

        # Pre-build a causal mask: upper triangle = -inf, lower = 0
        # Shape: (1, 1, context_length, context_length) for easy broadcasting
        mask = torch.triu(
            torch.ones(context_length, context_length) * float('-inf'),
            diagonal=1
        )
        # Register as a buffer so it moves to GPU with the model automatically
        self.register_buffer('causal_mask', mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-head self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, seq_len, embedding_dim)``.

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape as input.
        """
        B, T, C = x.shape   # batch, sequence length, channels (embedding_dim)

        # --- Compute Q, K, V projections -----------------------------------
        Q = self.q_proj(x)   # (B, T, C)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # --- Reshape to multi-head format ----------------------------------
        # (B, T, C) → (B, T, num_heads, head_dim) → (B, num_heads, T, head_dim)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # --- Scaled dot-product attention ----------------------------------
        scale = math.sqrt(self.head_dim)
        # (B, num_heads, T, T)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # Apply causal mask (slice to actual sequence length T)
        attn_scores = attn_scores + self.causal_mask[:, :, :T, :T]

        # Softmax + dropout on attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # --- Weighted sum of values ----------------------------------------
        # (B, num_heads, T, head_dim)
        context = torch.matmul(attn_weights, V)

        # --- Merge heads and project ----------------------------------------
        # (B, num_heads, T, head_dim) → (B, T, C)
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        output = self.out_proj(context)

        return output


# =============================================================================
# CLASS 2 — FEED-FORWARD NETWORK
# =============================================================================

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network (FFN) used inside each Transformer block.

    Architecture:
        Linear(embedding_dim → feedforward_dim) → GELU → Dropout
        → Linear(feedforward_dim → embedding_dim) → Dropout

    The intermediate dimension is typically 4× the embedding dimension,
    which gives the model capacity to learn complex feature interactions.

    Parameters
    ----------
    embedding_dim : int
        Input and output dimension.
    feedforward_dim : int
        Hidden (expanded) dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        embedding_dim: int   = EMBEDDING_DIM,
        feedforward_dim: int = FEEDFORWARD_DIM,
        dropout: float       = DROPOUT,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len, embedding_dim)``.

        Returns
        -------
        torch.Tensor
            Same shape as input.
        """
        return self.net(x)


# =============================================================================
# CLASS 3 — TRANSFORMER BLOCK
# =============================================================================

class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block using *pre-layer normalisation*.

    Pre-norm (LayerNorm before the sub-layer, not after) is more stable
    during training than the original post-norm design from "Attention is
    All You Need".

    Structure:
        x = x + MHA(LayerNorm(x))      ← attention sub-layer + residual
        x = x + FFN(LayerNorm(x))      ← feed-forward sub-layer + residual

    Parameters
    ----------
    embedding_dim : int
        Hidden size.
    num_heads : int
        Attention heads.
    feedforward_dim : int
        FFN inner dimension.
    dropout : float
        Dropout probability.
    context_length : int
        Max sequence length (for causal mask pre-computation).
    """

    def __init__(
        self,
        embedding_dim: int   = EMBEDDING_DIM,
        num_heads: int       = NUM_HEADS,
        feedforward_dim: int = FEEDFORWARD_DIM,
        dropout: float       = DROPOUT,
        context_length: int  = CONTEXT_LENGTH,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn  = MultiHeadSelfAttention(
            embedding_dim, num_heads, dropout, context_length
        )

        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ff    = FeedForward(embedding_dim, feedforward_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of one Transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len, embedding_dim)``.

        Returns
        -------
        torch.Tensor
            Same shape as input.
        """
        # Attention sub-layer with pre-norm and residual connection
        x = x + self.attn(self.norm1(x))
        # Feed-forward sub-layer with pre-norm and residual connection
        x = x + self.ff(self.norm2(x))
        return x


# =============================================================================
# CLASS 4 — FULL SLM MODEL
# =============================================================================

class SLMModel(nn.Module):
    """
    Small Language Model: a decoder-only Transformer (GPT-style).

    Key design choices:
    - **Learned positional embeddings** (not sinusoidal) — simpler and
      empirically comparable on short contexts.
    - **Weight tying** between the input token embedding matrix and the
      final output projection — reduces parameters and often improves
      performance by coupling the embedding and un-embedding spaces.
    - **Pre-LayerNorm** in every block for training stability.

    Parameters
    ----------
    vocab_size : int
        Size of the tokenizer vocabulary.
    embedding_dim : int
        Hidden dimension throughout the model.
    num_heads : int
        Number of attention heads per block.
    num_layers : int
        Number of Transformer blocks stacked.
    context_length : int
        Maximum sequence length.
    feedforward_dim : int
        FFN inner dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int      = VOCAB_SIZE,
        embedding_dim: int   = EMBEDDING_DIM,
        num_heads: int       = NUM_HEADS,
        num_layers: int      = NUM_LAYERS,
        context_length: int  = CONTEXT_LENGTH,
        feedforward_dim: int = FEEDFORWARD_DIM,
        dropout: float       = DROPOUT,
    ) -> None:
        super().__init__()

        self.context_length = context_length

        # ---- Embedding layers -----------------------------------------------
        # Token embedding: maps each token ID to a vector of size embedding_dim
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Positional embedding: one learned vector per position
        self.position_embedding = nn.Embedding(context_length, embedding_dim)

        self.embed_dropout = nn.Dropout(dropout)

        # ---- Transformer blocks ---------------------------------------------
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, feedforward_dim, dropout, context_length)
            for _ in range(num_layers)
        ])

        # ---- Final normalisation + output head ------------------------------
        self.final_norm = nn.LayerNorm(embedding_dim)

        # Output projection: embedding_dim → vocab_size
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

        # ---- Weight tying ---------------------------------------------------
        # Share weights between token embedding and the output projection.
        # This means lm_head.weight IS token_embedding.weight (no copy).
        self.lm_head.weight = self.token_embedding.weight

        # ---- Weight initialisation -----------------------------------------
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialise model weights using a scaled normal distribution.

        Following GPT-2, residual projections are scaled by 1/sqrt(num_layers)
        to prevent the residual stream from growing with depth.
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = 0.02
                # Scale residual path projections
                if 'out_proj' in name or 'lm_head' in name:
                    std *= (2 * NUM_LAYERS) ** -0.5
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute logits for every position in the sequence.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, seq_len, vocab_size)``.
        """
        B, T = input_ids.shape
        assert T <= self.context_length, (
            f"Input length {T} exceeds context_length {self.context_length}"
        )

        # Create position indices [0, 1, 2, ..., T-1] for each item in batch
        positions = torch.arange(T, device=input_ids.device)   # (T,)

        # Combine token and positional embeddings
        token_emb = self.token_embedding(input_ids)      # (B, T, C)
        pos_emb   = self.position_embedding(positions)   # (T, C)  → broadcasts
        x = self.embed_dropout(token_emb + pos_emb)

        # Pass through each Transformer block
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.final_norm(x)

        # Project to vocabulary logits
        logits = self.lm_head(x)   # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float  = 0.7,
        top_k: int          = 40,
    ) -> torch.Tensor:
        """
        Autoregressively generate tokens given a prompt.

        Sampling strategy: top-k sampling with temperature scaling.
          1. Compute logits for the current sequence.
          2. Scale by temperature (higher → more random, lower → more greedy).
          3. Keep only the top-k logits; set the rest to -inf.
          4. Sample from the resulting distribution.
          5. Append the sampled token and repeat.

        Parameters
        ----------
        input_ids : torch.Tensor
            Prompt token IDs, shape ``(1, prompt_len)``.
        max_new_tokens : int
            Number of new tokens to generate.
        temperature : float
            Sampling temperature. Set to 1.0 for unscaled.
        top_k : int
            Number of top logits to consider when sampling.

        Returns
        -------
        torch.Tensor
            Token IDs including the original prompt plus generated tokens.
            Shape: ``(1, prompt_len + max_new_tokens)``.
        """
        self.eval()

        for _ in range(max_new_tokens):

            # Truncate if the context is too long
            ctx = input_ids[:, -self.context_length:]

            # Forward pass — only need the last position's logits
            logits = self.forward(ctx)           # (1, T, vocab_size)
            logits = logits[:, -1, :]            # (1, vocab_size)

            # Temperature scaling
            logits = logits / max(temperature, 1e-8)

            # Top-k filtering
            if top_k > 0:
                top_k_actual = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, top_k_actual)
                threshold  = values[:, -1].unsqueeze(-1)
                logits     = logits.masked_fill(logits < threshold, float('-inf'))

            # Sample from the filtered distribution
            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)   # (1, 1)

            # Append to the running sequence
            input_ids = torch.cat([input_ids, next_tok], dim=1)

            # Stop if EOS token is generated
            if next_tok.item() == EOS_TOKEN_ID:
                break

        return input_ids


# =============================================================================
# UTILITY — PARAMETER COUNTER
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """
    Count and print the number of trainable parameters in the model.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to inspect.

    Returns
    -------
    int
        Total number of trainable parameters.
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {total:,}  ({total/1e6:.2f} M)")
    return total


# =============================================================================
# QUICK SMOKE TEST — run as a script
# =============================================================================

if __name__ == '__main__':
    print("Building SLMModel with default config ...")
    model = SLMModel()
    count_parameters(model)

    # Dummy forward pass
    dummy_input = torch.randint(0, VOCAB_SIZE, (2, CONTEXT_LENGTH))
    logits = model(dummy_input)
    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {logits.shape}  (expect: 2, {CONTEXT_LENGTH}, {VOCAB_SIZE})")

    # Dummy generation
    prompt = torch.tensor([[BOS_TOKEN_ID]], dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"Generated token IDs: {generated[0].tolist()}")
    print("Model smoke test passed.")
