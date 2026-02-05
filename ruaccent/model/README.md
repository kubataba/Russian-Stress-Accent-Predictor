# Model Files

This directory contains the trained model for Russian stress accent prediction.

## Files

### acc_model.pt
- **Size**: ~30 MB
- **Description**: Trained Transformer model weights
- **Format**: PyTorch state dict
- **Status**: ✅ Available via Git LFS

### vocab.json
- **Size**: ~3 KB  
- **Description**: Character-level vocabulary mapping
- **Format**: JSON dictionary
- **Status**: ✅ Included in repository

## Getting the Model

The model is stored in this repository using Git LFS. To download:  

```bash
# Clone the repository
git clone https://github.com/kubataba/Russian-Stress-Accent-Predictor.git

# Pull large files (model weights)
git lfs pull
```  

Alternatively, you can download individual files directly from GitHub:  

1. Navigate to the `model/` folder
2. Click on `acc_model.pt` or `vocab.json`
3. Click "Download". 

## Model Specifications  

- **Architecture**: Transformer (Encoder-Decoder)
- **Parameters**: ~12.5 million
- **Embedding dimension**: 256
- **Attention heads**: 8
- **Layers**: 4 encoder, 4 decoder
- **Vocabulary size**: 224 characters
- **Max sequence length**: 256

## Performance

- **Validation accuracy**: 99.7%
- **Inference speed**: ~2.5 sentences/sec (Mac Mini M4)
- **Supported formats**: Apostrophe (я') and Synthesis (+я) notation

## Quick Verification

```bash
# Check if files are downloaded
ls -lh model/

# Expected output:
# -rw-r--r--  30M acc_model.pt
# -rw-r--r--  3K  vocab.json
```

## Usage Example

```python
from accentor import load_accentor

# Load the model
model = load_accentor(
    model_path='model/acc_model.pt',
    vocab_path='model/vocab.json'
)

# Use for prediction
accented = model.predict("мама мыла раму")
print(accented)  # ма́ма мы́ла ра́му
```

## File Integrity  

Both files are tracked by Git LFS. The actual model file is stored on GitHub's LFS servers and automatically downloaded when you clone/pull the repository.
