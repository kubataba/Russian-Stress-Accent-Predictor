"""
accentor.py
Russian stress accent predictor with two output formats: apostrophe (') and synthesis (+ before vowel)

Features:
- Original model format: apostrophe (') after stressed vowel
- Two output formats: apostrophe (') and synthesis (+ before vowel)
- Simple and consistent interface
- Self-contained - includes all model classes
"""

import torch
import torch.nn as nn
import math
import re
import sys
import os
import json
from pathlib import Path
from typing import List, Union, Optional, Tuple, Dict
import warnings
import atexit


# ============================================================================
# Model Classes for standalone operation
# ============================================================================

def load_vocab(path: str) -> Dict[str, int]:
    """Load vocabulary from JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


class CharacterEmbedding(nn.Module):
    """Character-level embeddings with positional encoding"""
    
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        
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
    Exact architecture from the trained model
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
        
        # Transformer encoder-decoder with exact training configuration
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Important: matches training config
            norm_first=False   # Matches training config
        )
        
        # Output projection to vocabulary
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier uniform initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Causal mask for decoder (prevents looking at future tokens)"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, **kwargs) -> torch.Tensor:
        """Standard forward pass"""
        src_emb = self.embed(src)
        tgt_emb = self.embed(tgt)
        output = self.transformer(src_emb, tgt_emb, **kwargs)
        return self.output_proj(output)
    
    @torch.no_grad()
    def generate(self, src: torch.Tensor, max_len: int = 256, 
                 start_token_id: int = 1, end_token_id: int = 2) -> torch.Tensor:
        """Generate output using greedy decoding"""
        batch_size = src.size(0)
        device = src.device
        
        # Encode source
        src_emb = self.embed(src)
        memory = self.transformer.encoder(src_emb)
        
        # Start with <s> token
        tgt = torch.ones(batch_size, 1, dtype=torch.long, device=device) * start_token_id
        
        # Generate tokens sequentially
        for _ in range(max_len):
            tgt_emb = self.embed(tgt)
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            output = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = self.output_proj(output[:, -1, :])
            
            # Greedy decoding
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if all sequences have end token
            if (next_token == end_token_id).all():
                break
        
        return tgt


# ============================================================================
# Main Accentor Class
# ============================================================================

class Accentor:
    """
    Main accentor class
    Preserves original logic from accentor.py with model classes included
    """
    
    def __init__(
        self,
        model_path: str = 'model/acc_model.pt',
        vocab_path: str = 'model/vocab.json',
        device: str = None,
        quantize: bool = False,
        max_len: int = 200
    ):
        """
        Initialize the accentor
        
        Args:
            model_path: Path to model weights
            vocab_path: Path to vocabulary JSON
            device: 'cuda', 'mps', 'cpu', or None for auto-detect
            quantize: Use quantization for faster CPU inference
            max_len: Maximum sequence length
        """
        # Set threads for CPU (optimized for inference)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        
        # Auto-detect device
        if device=="auto" or device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"✅ Using CUDA device: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                device = 'mps'
                print("✅ Using Apple MPS (Metal)")
            else:
                device = 'cpu'
                print("✅ Using CPU")
        
        self.device = torch.device(device)
        self.max_len = max_len
        
        # Simple cache for frequently used texts
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Register cleanup on exit
        atexit.register(self._cleanup_cache)
        
        # Check for required files
        self._check_files(model_path, vocab_path)
        
        # Load vocabulary
        print(f"📖 Loading vocabulary from {vocab_path}...")
        self.vocab = load_vocab(vocab_path)
        self.idx2char = {v: k for k, v in self.vocab.items()}
        print(f"   Vocabulary size: {len(self.vocab)}")
        
        # Initialize model with exact training configuration
        print(f"🤖 Initializing model...")
        self.model = StressAccentTransformer(
            vocab_size=len(self.vocab),
            d_model=256,           # Must match trained model
            nhead=8,               # Must match trained model
            num_encoder_layers=4,  # Must match trained model
            num_decoder_layers=4,  # Must match trained model
            dim_feedforward=1024,  # Must match trained model
            dropout=0.1,           # Must match trained model
            max_len=256            # Must match trained model
        )
        
        # Load model weights
        print(f"📦 Loading weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        
        # Apply quantization if requested (CPU only)
        if quantize and self.device.type == 'cpu':
            print("⚡ Applying quantization for faster CPU inference...")
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear},
                dtype=torch.qint8
            )
        
        # Move to device and set evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ Accentor initialized successfully!\n")
    
    def _check_files(self, model_path: str, vocab_path: str):
        """Check if required files exist"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    
    def _cleanup_cache(self):
        """Cleanup cache on program exit"""
        if hasattr(self, '_cache'):
            self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def clear_cache(self):
        """Manually clear the cache"""
        self._cache.clear()
        print("🗑️  Cache cleared")
    
    def _apostrophe_to_plus(self, text: str) -> str:
        """
        Convert apostrophe format to synthesis format
        
        Args:
            text: Text in apostrophe format (vowel')
            
        Returns:
            Text in synthesis format (+vowel)
        """
        # Efficient conversion using regex
        result = text
        
        # Process from end to beginning to avoid index issues
        i = len(result) - 1
        while i >= 0:
            if result[i] == "'" and i > 0:
                # Look backward for vowel
                for j in range(i-1, -1, -1):
                    if result[j].lower() in 'аеёиоуыэюя':
                        # Replace ' with + before vowel
                        result = result[:j] + '+' + result[j:i] + result[i+1:]
                        i = j  # Skip to vowel position
                        break
            i -= 1
        
        return result
    
    def _plus_to_apostrophe(self, text: str) -> str:
        """
        Convert synthesis format to apostrophe format
        
        Args:
            text: Text in synthesis format (+vowel)
            
        Returns:
            Text in apostrophe format (vowel')
        """
        result = text
        i = 0
        
        while i < len(result):
            if result[i] == '+' and i + 1 < len(result):
                # Find next vowel
                for j in range(i+1, len(result)):
                    if result[j].lower() in 'аеёиоуыэюя':
                        # Move + after vowel
                        result = result[:i] + result[i+1:j+1] + "'" + result[j+1:]
                        i = j + 1  # Skip to after apostrophe
                        break
            i += 1
        
        return result
    
    def _text_to_tensor(self, text: str) -> torch.Tensor:
        """
        Convert text to tensor for model input
        Only used if you want to use the actual model
        """
        # Add <s> token
        indices = [self.vocab.get('<s>', 1)]
        
        # Convert characters (limit to max_len-2 for <s> and </s>)
        for char in text[:254]:  # 256 - 2 for special tokens
            idx = self.vocab.get(char, self.vocab.get('<unk>', 3))
            indices.append(idx)
        
        # Add </s> token
        indices.append(self.vocab.get('</s>', 2))
        
        # Pad to 256 for batching
        pad_idx = self.vocab.get('<pad>', 0)
        while len(indices) < 256:
            indices.append(pad_idx)
        
        # Truncate if too long
        indices = indices[:256]
        
        return torch.tensor([indices], dtype=torch.long, device=self.device)
    
    def _tensor_to_text(self, tensor: torch.Tensor) -> str:
        """
        Convert tensor back to text
        Only used if you want to use the actual model
        """
        if tensor.dim() == 2:
            indices = tensor[0].tolist()
        else:
            indices = tensor.tolist()
        
        chars = []
        for idx in indices:
            # Stop at </s> token
            if idx == self.vocab.get('</s>', 2):
                break
            # Skip special tokens
            if idx in [self.vocab.get('<pad>', 0), self.vocab.get('<s>', 1)]:
                continue
            
            char = self.idx2char.get(idx, '')
            if char:  # Only add if character exists
                chars.append(char)
        
        return ''.join(chars)
    
    def cache_info(self) -> dict:
        """Get cache statistics"""
        return {
            'size': len(self._cache),
            'hits': getattr(self, '_cache_hits', 0),
            'misses': getattr(self, '_cache_misses', 0)
        }
    
    @torch.no_grad()
    def _predict_batch(self, texts: List[str]) -> List[str]:
        """
        Batch prediction for multiple texts (optimized for speed)
        
        Args:
            texts: List of input texts
            
        Returns:
            List of texts with stress marks
        """
        if not texts:
            return []
        
        # Encode all texts to tensors (already padded to 256)
        src_tensors = []
        for text in texts:
            src = self._text_to_tensor(text)  # Returns (1, 256)
            src_tensors.append(src.squeeze(0))  # Remove batch dim -> (256,)
        
        # Stack into batch (batch_size, 256)
        src_batch = torch.stack(src_tensors).to(self.device)
        
        # Encode source
        src_emb = self.model.embed(src_batch)
        memory = self.model.transformer.encoder(src_emb)
        
        batch_size = len(texts)
        
        # Initialize target with <s> token
        tgt = torch.ones(batch_size, 1, dtype=torch.long, device=self.device)
        tgt = tgt * self.vocab.get('<s>', 1)
        
        # Track which sequences are done
        done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # Generate tokens
        max_len = 256
        for _ in range(max_len):
            # Embed current target
            tgt_emb = self.model.embed(tgt)
            
            # Create causal mask
            tgt_mask = self.model.generate_square_subsequent_mask(
                tgt.size(1)
            ).to(self.device)
            
            # Decode
            output = self.model.transformer.decoder(
                tgt_emb, 
                memory, 
                tgt_mask=tgt_mask
            )
            
            # Get next token
            logits = self.model.output_proj(output[:, -1, :])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Mark done sequences
            done = done | (next_token.squeeze(-1) == self.vocab.get('</s>', 2))
            
            # Stop if all done
            if done.all():
                break
        
        # Decode all sequences
        results = []
        for i in range(batch_size):
            text = self._tensor_to_text(tgt[i])
            results.append(text)
        
        return results
    
    @torch.no_grad()
    def _predict_raw(self, text: str) -> str:
        """
        Get raw prediction from model (returns apostrophe format)
        
        Args:
            text: Input text without accents
            
        Returns:
            Text in apostrophe format (vowel')
        """
        # Check cache first
        if text in self._cache:
            return self._cache[text]
        
        # Use batch prediction with single text
        result = self._predict_batch([text])[0]
        
        # Cache the result (limit cache size)
        if len(self._cache) < 10000:
            self._cache[text] = result
        
        return result
    
    def __call__(self, texts: Union[str, List[str]], 
                 format: str = 'apostrophe', 
                 batch_size: int = 8) -> Union[str, List[str], Tuple[str, str]]:
        """
        Main call method - add stress accents to text(s)
        
        Args:
            texts: Single text string or list of texts
            format: Output format - 'apostrophe', 'synthesis', or 'both'
            batch_size: Batch size for processing (default: 8)
            
        Returns:
            Depending on format:
            - 'apostrophe': Text with apostrophe after stressed vowel (я')
            - 'synthesis': Text with + before stressed vowel (+я)
            - 'both': Tuple of (apostrophe, synthesis) formats
        """
        single_input = isinstance(texts, str)
        
        if single_input:
            texts = [texts]
        
        results_apostrophe = []
        
        # Process in batches for speed
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Check cache first
            batch_to_process = []
            batch_indices = []
            cached_results = []
            
            for idx, text in enumerate(batch):
                if text in self._cache:
                    cached_results.append((i + idx, self._cache[text]))
                    self._cache_hits += 1
                else:
                    batch_to_process.append(text)
                    batch_indices.append(i + idx)
                    self._cache_misses += 1
            
            # Process uncached texts in batch
            if batch_to_process:
                batch_predictions = self._predict_batch(batch_to_process)
                
                # Cache results
                for text, pred in zip(batch_to_process, batch_predictions):
                    if len(self._cache) < 10000:
                        self._cache[text] = pred
                
                # Combine cached and new results
                all_results = cached_results + list(zip(batch_indices, batch_predictions))
                all_results.sort(key=lambda x: x[0])  # Sort by original index
                results_apostrophe.extend([r[1] for r in all_results])
            else:
                # All cached
                results_apostrophe.extend([r[1] for r in cached_results])
        
        # Convert to synthesis format if needed
        if format == 'synthesis' or format == 'both':
            results_synthesis = [self._apostrophe_to_plus(text) for text in results_apostrophe]
        
        # Return based on format
        if format == 'apostrophe':
            return results_apostrophe[0] if single_input else results_apostrophe
        elif format == 'synthesis':
            return results_synthesis[0] if single_input else results_synthesis
        elif format == 'both':
            if single_input:
                return (results_apostrophe[0], results_synthesis[0])
            else:
                return list(zip(results_apostrophe, results_synthesis))
        else:
            raise ValueError(f"Unknown format: {format}. Use 'apostrophe', 'synthesis', or 'both'")
    
    def put_accent(self, texts: Union[str, List[str]], 
                   format: str = 'apostrophe') -> Union[str, List[str]]:
        """
        Two output formats method for accent placement
        
        Args:
            texts: Text or list of texts to process
            format: Output format ('apostrophe' or 'synthesis')
            
        Returns:
            Text(s) with accents
        """
        return self(texts, format=format)
    
    def use_model_inference(self, enable: bool = True):
        """
        Toggle between simulation and actual model inference
        
        Note: You need to implement the actual model connection here
        """
        if enable:
            print("⚠️  Model inference not yet implemented. Using simulation.")
        # Add your model inference logic here when ready


# ============================================================================
# Accentor Load Function
# ============================================================================

def load_accentor(
    model_path: str = 'model/acc_model.pt',
    vocab_path: str = 'model/vocab.json',
    device: str = None,
    quantize: bool = False
) -> Accentor:
    """
    Load accentor (with two output formats)
    
    Args:
        model_path: Path to model file
        vocab_path: Path to vocabulary file
        device: Device to use ('cuda', 'mps', 'cpu', or None for auto)
        quantize: Use quantization for faster CPU inference
        
    Returns:
        Accentor instance
    """
    print("=" * 60)
    print("Loading Custom Accentor")
    print("=" * 60)
    
    accentor = Accentor(
        model_path=model_path,
        vocab_path=vocab_path,
        device=device,
        quantize=quantize
    )
    
    return accentor


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Test if files exist
    if os.path.exists('acc_model.pt') and os.path.exists('vocab.json'):
        print("Testing accentor...")
        
        # Load accentor
        accentor = load_accentor()
        
        # Test examples
        test_texts = [
            "Я устало иду во главе своей армии, наступая на цветы.",
            "Привет, мир! Как дела, всё уже хорошо?",
            "Хотел я зайти в замок, но ключ в замок не подошёл."
        ]
        
        for text in test_texts:
            print(f"\nInput:     {text}")
            apostrophe = accentor(text, format='apostrophe')
            synthesis = accentor(text, format='synthesis')
            print(f"Apostrophe: {apostrophe}")
            print(f"Synthesis:  {synthesis}")
        
        # Show cache size
        print(f"\nCache size: {len(accentor._cache)} items")
        
        # Clear cache
        accentor.clear_cache()
        
    else:
        print("⚠️  Model files not found in current directory.")
        print("\nRequired files:")
        print("  - acc_model.pt    (model weights)")
        print("  - vocab.json      (vocabulary)")
        print("\nPlace these files in the same directory as accentor.py")