"""
train_stress_model.py
Train stress accent model on sentence_pairs.csv dataset

Works with CSV format: source,target,length,stress_count,word_count
Uses character-level vocabulary built automatically from the data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import csv
import json
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

from model import StressAccentTransformer


class CSVStressDataset(Dataset):
    """Dataset for sentence pairs from CSV file"""
    
    def __init__(
        self,
        csv_path: str,
        vocab: Dict[str, int],
        max_len: int = 512,
        skip_header: bool = True
    ):
        self.pairs = []
        self.vocab = vocab
        self.max_len = max_len
        
        # Load CSV
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            if skip_header:
                next(reader)  # Skip header row
            
            for row in reader:
                if len(row) >= 2:  # At least source and target
                    source = row[0]
                    target = row[1]
                    self.pairs.append((source, target))
        
        print(f"Loaded {len(self.pairs)} pairs from {csv_path}")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        source, target = self.pairs[idx]
        
        # Encode source
        src_indices = [self.vocab.get(c, self.vocab['<unk>']) for c in source]
        src_indices = [self.vocab['<s>']] + src_indices + [self.vocab['</s>']]
        
        # Encode target (same vocab)
        tgt_indices = [self.vocab.get(c, self.vocab['<unk>']) for c in target]
        tgt_indices = [self.vocab['<s>']] + tgt_indices + [self.vocab['</s>']]
        
        # Padding
        src_indices = self._pad_sequence(src_indices, self.vocab['<pad>'])
        tgt_indices = self._pad_sequence(tgt_indices, self.vocab['<pad>'])
        
        return (
            torch.tensor(src_indices, dtype=torch.long),
            torch.tensor(tgt_indices, dtype=torch.long)
        )
    
    def _pad_sequence(self, seq: List[int], pad_idx: int) -> List[int]:
        if len(seq) < self.max_len:
            seq += [pad_idx] * (self.max_len - len(seq))
        else:
            seq = seq[:self.max_len]
        return seq


def build_vocab_from_csv(csv_path: str, min_freq: int = 1) -> Dict[str, int]:
    """
    Build character-level vocabulary from CSV file
    
    Args:
        csv_path: Path to CSV file with sentence pairs
        min_freq: Minimum frequency for character to be included (default 1 = all chars)
    
    Returns:
        vocab: Character to index mapping
    """
    from collections import Counter
    
    char_counts = Counter()
    
    # Read CSV and count characters
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in reader:
            if len(row) >= 2:
                # Count from both source and target
                char_counts.update(row[0])
                char_counts.update(row[1])
    
    # Build vocab with special tokens
    vocab = {
        '<pad>': 0,
        '<s>': 1,
        '</s>': 2,
        '<unk>': 3,
    }
    
    # Add characters above frequency threshold
    idx = len(vocab)
    for char, count in char_counts.most_common():
        if count >= min_freq:
            if char not in vocab:
                vocab[char] = idx
                idx += 1
    
    return vocab


def save_vocab(vocab: Dict[str, int], path: str):
    """Save vocabulary to JSON file"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Vocab saved to {path}")


def load_vocab(path: str) -> Dict[str, int]:
    """Load vocabulary from JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    print(f"Vocab loaded from {path}")
    return vocab


class Trainer:
    """Training class for stress accent model"""
    
    def __init__(
        self,
        model: StressAccentTransformer,
        train_dataset: CSVStressDataset,
        val_dataset: CSVStressDataset,
        vocab: Dict[str, int],
        device: str = 'mps',
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 1000,
        save_dir: str = './checkpoints',
        log_dir: str = './logs',
        early_stopping_patience: int = 3
    ):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.vocab = vocab
        
        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01
        )
        
        # Scheduler with warmup
        self.scheduler = self._create_scheduler(warmup_steps)
        
        # Loss function (ignore padding)
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
        
        # Directories
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.early_stopping_patience = early_stopping_patience
    
    def _create_scheduler(self, warmup_steps: int):
        """Create learning rate scheduler with warmup and cosine decay"""
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return max(0.1, 0.5 * (1 + math.cos(math.pi * step / 
                                                 (len(self.train_loader) * self.num_epochs))))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train(self):
        """Main training loop"""
        print("=" * 60)
        print("🚀 TRAINING START")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Vocab size: {len(self.vocab)}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Epochs: {self.num_epochs}")
        print("=" * 60)
        
        for epoch in range(self.num_epochs):
            print(f"\n📊 Epoch {epoch + 1}/{self.num_epochs}")
            
            train_loss = self._train_epoch(epoch)
            val_loss = self._validate_epoch(epoch)
            
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            
            # Check for overfitting
            gap = val_loss - train_loss
            if gap > 0.5:
                print(f"   ⚠️  WARNING: Possible overfitting (gap: {gap:.4f})")
            elif gap > 0.2:
                print(f"   ⚡ Generalization gap: {gap:.4f}")
            
            # Save checkpoint
            self._save_checkpoint(epoch, val_loss)
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_best_model()
                print(f"   ✅ New best model! Val Loss: {val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1
                print(f"   📊 No improvement for {self.epochs_without_improvement} epoch(s)")
                
                # Early stopping
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\n⏹️  Early stopping triggered after {epoch + 1} epochs")
                    print(f"   No improvement for {self.early_stopping_patience} epochs")
                    break
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Total epochs: {epoch + 1}/{self.num_epochs}")
        print("=" * 60)
    
    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (src, tgt) in enumerate(pbar):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            # Decoder input/output (shift by 1)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Create causal mask for decoder
            tgt_mask = self.model.generate_square_subsequent_mask(
                tgt_input.size(1)
            ).to(self.device)
            
            # Don't use padding masks on MPS - causes compatibility issues
            # The model will handle padding tokens through the loss function
            src_padding_mask = None
            tgt_padding_mask = None
            
            # Forward pass
            logits = self.model(
                src, tgt_input,
                tgt_mask=tgt_mask,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask
            )
            
            # Calculate loss
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )
            
            # Backward pass with gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Log to tensorboard
                if self.global_step % 100 == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                    self.writer.add_scalar('train/lr', lr, self.global_step)
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            pbar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        num_batches = len(self.val_loader)
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for src, tgt in pbar:
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            tgt_mask = self.model.generate_square_subsequent_mask(
                tgt_input.size(1)
            ).to(self.device)
            
            # Don't use padding masks on MPS - causes compatibility issues
            src_padding_mask = None
            tgt_padding_mask = None
            
            logits = self.model(
                src, tgt_input,
                tgt_mask=tgt_mask,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask
            )
            
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )
            
            # Calculate accuracy (excluding padding)
            predictions = logits.argmax(dim=-1)
            mask = tgt_output != self.vocab['<pad>']
            correct = (predictions == tgt_output) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            
            total_loss += loss.item()
            acc = total_correct / total_tokens if total_tokens > 0 else 0
            pbar.set_postfix({'loss': loss.item(), 'acc': f'{acc:.3f}'})
        
        avg_loss = total_loss / num_batches
        avg_acc = total_correct / total_tokens if total_tokens > 0 else 0
        
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/accuracy', avg_acc, epoch)
        
        print(f"   Val Accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
        
        return avg_loss
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save training checkpoint (keep only last 3)"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'global_step': self.global_step,
            'vocab_size': len(self.vocab)
        }
        
        path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        
        # Remove old checkpoints (keep only last 3)
        checkpoints = sorted(self.save_dir.glob('checkpoint_epoch_*.pt'))
        if len(checkpoints) > 3:
            for old_ckpt in checkpoints[:-3]:
                old_ckpt.unlink()
                print(f"   🗑️  Removed old checkpoint: {old_ckpt.name}")
    
    def _save_best_model(self):
        """Save best model weights"""
        path = self.save_dir / 'best_model.pt'
        torch.save(self.model.state_dict(), path)


def split_csv_dataset(
    csv_path: str,
    train_ratio: float = 0.9,
    output_dir: str = './data'
) -> Tuple[str, str]:
    """
    Split CSV dataset into train and validation sets
    
    Args:
        csv_path: Path to full CSV file
        train_ratio: Ratio for training set (default 0.9 = 90% train, 10% val)
        output_dir: Directory to save split files
    
    Returns:
        (train_csv_path, val_csv_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Read all data
    pairs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        pairs = list(reader)
    
    # Shuffle
    import random
    random.seed(42)
    random.shuffle(pairs)
    
    # Split
    split_idx = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    # Save train
    train_path = output_dir / 'train.csv'
    with open(train_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(train_pairs)
    
    # Save val
    val_path = output_dir / 'val.csv'
    with open(val_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(val_pairs)
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_pairs)} samples -> {train_path}")
    print(f"  Val: {len(val_pairs)} samples -> {val_path}")
    
    return str(train_path), str(val_path)


def main():
    """Main training function"""
    print("=" * 60)
    print("🎓 STRESS ACCENT MODEL TRAINING (CSV Dataset)")
    print("=" * 60)
    
    # Check for MPS (Apple Silicon GPU)
    if torch.backends.mps.is_available():
        device = 'mps'
        print("✅ MPS (Metal) available - using M4 GPU")
    else:
        device = 'cpu'
        print("⚠️ MPS not available - using CPU")
    
    # Paths
    CSV_PATH = Path('./sentence_pairs.csv')
    DATA_DIR = Path('./data')
    DATA_DIR.mkdir(exist_ok=True)
    
    # Check if CSV exists
    if not CSV_PATH.exists():
        print(f"❌ Error: {CSV_PATH} not found!")
        print("   Run pdata.py first to generate sentence_pairs.csv")
        return
    
    # Build or load vocabulary
    vocab_path = DATA_DIR / 'vocab.json'
    if vocab_path.exists():
        print(f"\n📚 Loading existing vocab from {vocab_path}")
        vocab = load_vocab(vocab_path)
    else:
        print(f"\n📚 Building vocabulary from {CSV_PATH}")
        vocab = build_vocab_from_csv(CSV_PATH, min_freq=1)
        save_vocab(vocab, vocab_path)
    
    print(f"   Vocab size: {len(vocab)} characters")
    print(f"   Includes: apostrophe ('), ё, and all other characters")
    
    # Split dataset if needed
    train_csv = DATA_DIR / 'train.csv'
    val_csv = DATA_DIR / 'val.csv'
    
    if not train_csv.exists() or not val_csv.exists():
        print(f"\n📊 Splitting dataset (90% train, 10% val)")
        train_csv, val_csv = split_csv_dataset(CSV_PATH, train_ratio=0.9, output_dir=DATA_DIR)
    else:
        print(f"\n📊 Using existing split:")
        print(f"   Train: {train_csv}")
        print(f"   Val: {val_csv}")
    
    # Create datasets
    print("\n📊 Loading datasets...")
    train_dataset = CSVStressDataset(
        train_csv,
        vocab,
        max_len=256
    )
    
    val_dataset = CSVStressDataset(
        val_csv,
        vocab,
        max_len=256
    )
    
    # Create model
    print("\n🏗️ Creating model...")
    model = StressAccentTransformer(
        vocab_size=len(vocab),
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_len=256
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        vocab=vocab,
        device=device,
        batch_size=16,
        learning_rate=5e-4,
        num_epochs=10,
        gradient_accumulation_steps=4,
        warmup_steps=500,
        save_dir='./checkpoints',
        log_dir='./logs',
        early_stopping_patience=3  # Stop if no improvement for 3 epochs
    )
    
    # Train
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    
    print(f"\n⏱️ Time: {elapsed/60:.1f} minutes")
    print(f"💾 Model: ./checkpoints/best_model.pt")
    print(f"📚 Vocab: ./data/vocab.json")
    print(f"📊 Logs: ./logs/ (tensorboard --logdir=./logs)")


if __name__ == "__main__":
    main()