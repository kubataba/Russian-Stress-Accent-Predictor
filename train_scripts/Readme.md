# Training Scripts

This directory contains scripts used to train the Russian accent prediction model from scratch.

## Available Scripts

- **model.py** - Defines the Transformer model architecture
- **prepare_data.py** - Prepares and preprocesses training data  
- **train_model.py** - Main training script

## Quick Start

### Prerequisites  

```bash
pip install torch pandas tqdm
```

### If data already exists:  

```bash
python train_model.py
```

### To preprocess data first: 

```bash
python3.11 prepare_data.py
python3.11 train_model.py
```  

## Training Details

- **Architecture**: Transformer with 4 encoder/decoder layers
- **Training data**: Located in `../data/` folder
- **Output model**: Saved to `../model/` folder
- **Accuracy**: 99.7% on validation set
- **Training time**: 8+ hours on GPU

## Notes

- The pre-trained model is already available in `../model/`
- Most users don't need these scripts
- Use `ruaccent` command for inference
