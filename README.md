# Russian Stress Accent Predictor (Accentor) - ruaccent-predictor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Automatic stress accent placement in Russian text using a character-level Transformer model. Available on PyPI as `ruaccent-predictor`.

## 📋 Description

This project is a deep learning model for automatic stress accent placement in Russian text. The model is trained on a dataset of over 224,000 sentence pairs from literary works and achieves **99.7% accuracy** on the validation dataset.

### Key Features

- ✅ 99.7% accuracy on validation dataset
- 🚀 Two output formats: apostrophe (я́) and synthesis (+я)
- ⚡ Batch processing support for speed optimization
- 💾 Built-in result caching
- 🔧 Support for CPU, CUDA, and Apple MPS (Metal)
- 📦 Easy pip installation: `pip install ruaccent-predictor`

### Technical Details

**Character-Level Model**: The model operates at the character level with an automatically extracted vocabulary of 224 characters from the training dataset. This approach allows for high accuracy while maintaining a compact model size (~12.5M parameters).

**Vocabulary**: Automatically extracted from the training corpus, includes:  

- Cyrillic letters (uppercase and lowercase)
- Basic punctuation
- Latin letters
- Special tokens

### Output Formats

**Apostrophe format**: Stress mark is placed **after** the stressed vowel  
`Example: В лесу' родила'сь ёлочка` (Optimal for reading with stress marks during learning)

**Synthesis format**: Plus sign is placed **before** the stressed vowel  
`Example: В лес+у родил+ась ёлочка` (Optimal for speech synthesis, e.g., Silero TTS)

## ⚠️ Model Limitations

The model has the following known limitations:

1. **Does not restore missing letter "ё"**: The model works with the input text as-is and does not replace "е" with "ё"
2. **Does not mark stress on "ё"**: Since "ё" is always stressed in Russian, the model does not place additional stress marks on it
3. **Single-vowel words**: Words with only one vowel are not marked as they are inherently stressed
4. **No grammatical analysis**: The model operates purely on character sequences without morphological or syntactic analysis
5. **Training data limitations**: Accuracy may vary for texts outside the literary domain of the training data

## 📦 PyPI Installation

The package is available on PyPI as `ruaccent-predictor`:  

```bash
pip install ruaccent-predictor
```

### Usage as Python Package  

```python
from ruaccent import load_accentor

# Load the model  

accentor = load_accentor()

# Predict stress accents  

text = "привет мир"
result = accentor(text)
print(result)  # приве'т мир
```

### Usage as CLI Tool

After installation, use the `ruaccent` command:  

```bash
# Process single text
ruaccent "привет как дела"

# Process file
ruaccent --input-file input.txt --output-file output.txt

# Synthesis format
ruaccent "привет" --format synthesis

# Both formats
ruaccent "текст" --format both

# Pipe input
echo "мама мыла раму" | ruaccent 
```

### Available Options:  

- `--format`: Output format (apostrophe, synthesis, both)
- `--batch-size`: Batch size for processing (default: 8)
- `--device`: Device for inference (auto, cpu, cuda, mps)
- `--input-file`, `-i`: Input text file
- `--output-file`, `-o`: Output file

## 🎯 Performance

### Benchmarks  
- **Accuracy**: 99.7% on validation set (22,000 sentences)
- **Speed**: ~10 sentences /sec with batch_size=8 on Mac Mini M4
- **Model size**: ~12.5M parameters
- **Vocabulary**: 224 characters (Cyrillic, punctuation, Latin)

### Optimal Settings  

```python
# For maximum performance
accentor = load_accentor()
results = accentor(texts, batch_size=8, format='apostrophe')
```

## 📁 Project Structure

```
Russian-Stress-Accent-Predictor/
├── ruaccent/                    # Main package (PyPI)
│   ├── __init__.py
│   ├── accentor.py             # Main module with model
│   └── cli.py                  # CLI interface
├── model/                      # Trained model
│   ├── README.md              # Model documentation
│   ├── acc_model.pt           # Model weights (30MB, Git LFS)
│   └── vocab.json             # Character vocabulary
├── data/                       # Datasets
│   ├── train.csv              # Training set (115MB, Git LFS)
│   └── val.csv                # Validation set (13MB)
├── examples/                   # Usage examples
│   ├── basic_usage.py         # Basic examples
│   └── batch_processing.py    # Batch processing and tests
├── train_scripts/              # Model training scripts
│   ├── model.py               # Transformer architecture
│   ├── prepare_data.py        # Data preparation
│   ├── train_model.py         # Model training
│   └── README.md              # Training instructions
├── .gitattributes             # Git LFS configuration
├── .gitignore                 # Ignored files
├── LICENSE                    # MIT license
├── MANIFEST.in                # Included files for PyPI
├── pyproject.toml             # Package configuration
├── README.md                  # This documentation
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
└── run_training.sh            # Training launch script
```

## 🧪 Usage Examples

### Basic Example (examples/basic_usage.py)  

```python
from ruaccent import load_accentor

accentor = load_accentor()
texts = ["привет мир", "мама мыла раму", "солнце светит ярко"]

# Apostrophe format
results = accentor(texts, format='apostrophe')
for original, accented in zip(texts, results):
    print(f"{original} → {accented}")
```

### Batch Processing and Tests (examples/batch_processing.py)  

```bash
python examples/batch_processing.py
```
Tests performance with different batch sizes, shows cache statistics and optimal settings.

## 🏗️ Training Scripts

For developers and researchers in the `train_scripts/` folder:

### Training Scripts  

- `model.py` - Transformer architecture definition
- `prepare_data.py` - Data preprocessing and preparation
- `train_model.py` - Main training script

### Training from Scratch  

```bash
# Install dependencies
pip install torch pandas tqdm

# Start training
cd train_scripts
python train_model.py
```

**Note**: Training requires significant resources (GPU recommended) and takes several hours.

## 🔤 Output Formats

### 1. Apostrophe Format (я').  
 
Apostrophe is placed **after** the stressed vowel:  

- Input: `привет`
- Output: `приве'т`
- Use case: Text display, reading

### 2. Synthesis Format (+я). 

Plus sign is placed **before** the stressed vowel:  

- Input: `привет`
- Output: `прив+ет`
- Use case: Speech synthesis, TTS systems

## 🚀 Quick Start

### After pip installation:  

```bash
# Verify installation
ruaccent "тестовая фраза"

# Run examples  
python examples/basic_usage.py  
```

### From Source Code:  

```bash
# Clone repository  
git clone https://github.com/kubataba/Russian-Stress-Accent-Predictor.git
cd Russian-Stress-Accent-Predictor

# Install in development mode  

pip install -e .

# Use as usual  

ruaccent "ваш текст"
```

## 📊 Performance and Caching

The model uses intelligent caching:  

- **Cache hits**: ~0.0000s per text
- **Cache misses**: ~0.5s for first call
- **Optimal batch size**: 8 (10 sentences /sec on MPS)
- **Cache size**: Up to 10,000 items  

```python
# View cache statistics  
cache_info = accentor.cache_info()
print(f"Cache hits: {cache_info['hits']}, misses: {cache_info['misses']}")

# Clear cache  

accentor.clear_cache()
```

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

The project is distributed under the MIT license. See the `LICENSE` file for details.

The dataset is also distributed under the MIT license:
- **Source**: [nevmenandr/accentual-syllabic-verse-in-russian-prose](https://huggingface.co/datasets/nevmenandr/accentual-syllabic-verse-in-russian-prose)
- **License**: MIT

## 🙏 Acknowledgments

- Dataset provided by [nevmenandr](https://huggingface.co/nevmenandr)
- Project uses the Transformer architecture from PyTorch
- Inspired by natural language processing tasks for Russian language

## 🔗 Useful Links

- **PyPI package**: `ruaccent-predictor`
- **Repository**: https://github.com/kubataba/Russian-Stress-Accent-Predictor
- **Dataset**: https://huggingface.co/datasets/nevmenandr/accentual-syllabic-verse-in-russian-prose
- **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html

---

**Package Version**: 1.1.0  
**Package Name**: ruaccent-predictor  
**Last Updated**: February 2026