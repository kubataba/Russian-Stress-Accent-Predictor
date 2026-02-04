"""
model.py
Transformer model architecture for Russian stress accent prediction

This file contains the model classes used for training.
For inference, these classes are also included in accentor.py for standalone operation.
"""

import torch
import torch.nn as nn
import math


class CharacterEmbedding(nn.Module):
    """Character-level embeddings with positional encoding"""
    
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        
        # Character embeddings
        self.char_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Pre-computed positional encoding
        self.register_buffer('pos_encoding', self._create_positional_encoding(max_len, d_model))
        
        self.dropout = nn.Dropout(0.1)
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with embeddings and positional encoding"""
        seq_len = x.size(1)
        pos_enc = self.pos_encoding[:, :seq_len, :]
        x = self.char_embed(x) * math.sqrt(self.d_model)
        x = x + pos_enc
        return self.dropout(x)


class StressAccentTransformer(nn.Module):
    """
    Character-level Transformer for stress accent prediction
    
    Architecture:
    - Shared character embeddings for encoder and decoder
    - Transformer encoder-decoder with multi-head attention
    - Character-level generation with greedy decoding
    
    Args:
        vocab_size: Size of character vocabulary
        d_model: Embedding dimension (default: 256)
        nhead: Number of attention heads (default: 8)
        num_encoder_layers: Number of encoder layers (default: 4)
        num_decoder_layers: Number of decoder layers (default: 4)
        dim_feedforward: Dimension of feedforward network (default: 1024)
        dropout: Dropout rate (default: 0.1)
        max_len: Maximum sequence length (default: 256)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 256
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Shared embedding for source and target
        self.embed = CharacterEmbedding(vocab_size, d_model, max_len)
        
        # Transformer encoder-decoder
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Important: batch dimension comes first
            norm_first=False   # Post-norm (standard Transformer)
        )
        
        # Output projection to vocabulary
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier uniform initialization for better training stability"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate causal mask for decoder (prevents looking at future tokens)
        
        Args:
            sz: Sequence length
            
        Returns:
            Causal mask of shape (sz, sz)
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))
    
    def forward(
        self, 
        src: torch.Tensor, 
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        src_padding_mask: torch.Tensor = None,
        tgt_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            src: Source sequence (batch_size, src_len)
            tgt: Target sequence (batch_size, tgt_len)
            tgt_mask: Causal mask for target (optional)
            src_padding_mask: Padding mask for source (optional)
            tgt_padding_mask: Padding mask for target (optional)
            
        Returns:
            Logits of shape (batch_size, tgt_len, vocab_size)
        """
        # Embed sequences
        src_emb = self.embed(src)
        tgt_emb = self.embed(tgt)
        
        # Pass through transformer
        output = self.transformer(
            src_emb, 
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # Project to vocabulary
        return self.output_proj(output)
    
    @torch.no_grad()
    def generate(
        self, 
        src: torch.Tensor, 
        max_len: int = 256,
        start_token_id: int = 1,
        end_token_id: int = 2
    ) -> torch.Tensor:
        """
        Generate output sequence using greedy decoding
        
        Args:
            src: Source sequence (batch_size, src_len)
            max_len: Maximum generation length
            start_token_id: ID of start token (default: 1)
            end_token_id: ID of end token (default: 2)
            
        Returns:
            Generated sequence (batch_size, gen_len)
        """
        batch_size = src.size(0)
        device = src.device
        
        # Encode source
        src_emb = self.embed(src)
        memory = self.transformer.encoder(src_emb)
        
        # Start with <s> token
        tgt = torch.ones(batch_size, 1, dtype=torch.long, device=device) * start_token_id
        
        # Generate tokens sequentially
        for _ in range(max_len):
            # Embed current target sequence
            tgt_emb = self.embed(tgt)
            
            # Create causal mask
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            # Decode
            output = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            
            # Get logits for last token
            logits = self.output_proj(output[:, -1, :])
            
            # Greedy decoding: take argmax
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if all sequences have end token
            if (next_token == end_token_id).all():
                break
        
        return tgt
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Utility function for model summary
def print_model_summary(model: StressAccentTransformer):
    """Print model architecture summary"""
    print("=" * 70)
    print("Model Architecture Summary")
    print("=" * 70)
    
    print(f"\nVocabulary size: {model.vocab_size}")
    print(f"Embedding dimension: {model.d_model}")
    print(f"Total parameters: {model.count_parameters():,}")
    
    print(f"\nModel structure:")
    print(model)
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...\n")
    
    vocab_size = 224  # Example vocab size
    model = StressAccentTransformer(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_len=256
    )
    
    print_model_summary(model)
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    src_len = 10
    tgt_len = 12
    
    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))
    
    # Create causal mask
    tgt_mask = model.generate_square_subsequent_mask(tgt_len)
    
    # Forward pass
    output = model(src, tgt, tgt_mask=tgt_mask)
    
    print(f"Input shapes:")
    print(f"  src: {src.shape}")
    print(f"  tgt: {tgt.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {tgt_len}, {vocab_size})")
    
    assert output.shape == (batch_size, tgt_len, vocab_size), "Output shape mismatch!"
    print("\n✅ Model test passed!")
    
    # Test generation
    print("\nTesting generation...")
    generated = model.generate(src, max_len=20)
    print(f"Generated sequence shape: {generated.shape}")
    print("✅ Generation test passed!")
