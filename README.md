# ECG Classification with Parameter-Efficient Fine-Tuning (PEFT)

A comprehensive framework for binary ECG signal classification using parameter-efficient fine-tuning methods (LoRA and Adapters) on three pre-trained audio models.

## Overview

This project implements and evaluates 6 model configurations for ECG (electrocardiogram) classification:

| Model | PEFT Method | Details |
|-------|-------------|---------|
| **Wav2Vec2** | Adapter | Speech foundation model fine-tuned with bottleneck adapter |
| **Wav2Vec2** | LoRA | Speech foundation model with Low-Rank Adaptation |
| **HuBERT** | Adapter | Self-supervised speech model with adapter layers |
| **HuBERT** | LoRA | Self-supervised speech model with LoRA modules |
| **ECG-FM** | Adapter | ECG-specific foundation model with adapter |
| **ECG-FM** | LoRA | ECG-specific foundation model with LoRA |

All configurations use **frozen pre-trained backbones** with minimal trainable parameters, enabling efficient fine-tuning on downstream ECG classification tasks.

## Key Features

- ✅ **Parameter-Efficient Fine-Tuning**: LoRA and Adapter-based approaches reduce trainable parameters by 95%+
- ✅ **Multi-Model Comparison**: Evaluate speech models (Wav2Vec2, HuBERT) vs. domain-specific models (ECG-FM)
- ✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrices
- ✅ **Hyperparameter Tuning**: Optional grid search over batch sizes, learning rates, weight decay, LoRA ranks
- ✅ **Training Visualization**: Learning curves, comparison plots, confusion matrices
- ✅ **Stratified Data Splits**: 70% train / 15% validation / 15% test with label stratification
- ✅ **GPU/MPS Support**: Automatic device detection (CUDA, Apple Silicon, CPU)

## Requirements

### Python Packages
```
torch>=1.13.0
transformers>=4.25.0
peft>=0.4.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.5.0
```

### Hardware
- **Recommended**: GPU (NVIDIA CUDA or Apple Silicon MPS)
- **Minimum**: CPU (slower, but functional)
- **Memory**: 8GB+ RAM for batch processing

### Data
- **Input**:
  - Supports both `.csv` and `.csv.zip` formats
  - Expected format: 600 numerical features per row, last column is binary label (0/1)

## Installation

### 1. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install torch transformers peft scikit-learn pandas numpy matplotlib
```

## Usage

### Basic Usage
```bash
python bnair_lab3-1.py
```

### Configuration Options

Edit the hyperparameter dictionary in the script:

```python
HP_CONFIG = {
    'batch_sizes': [32, 64],              # Batch sizes to try
    'learning_rates_adapter': [5e-5, 1e-4],  # LR for adapter models
    'learning_rates_lora': [2.5e-5, 5e-5],   # LR for LoRA models
    'epochs': [10],                       # Number of training epochs
    'weight_decays': [1e-4, 1e-3],        # L2 regularization
    'patience': [3],                      # Early stopping patience
    'lora_ranks': [8, 16],                # LoRA rank values
    'adapter_hiddens': [256, 512]         # Adapter hidden dimensions
}
```

### Enable Hyperparameter Tuning

```python
TUNING_MODE = True  # Set to True to run grid search before main training
```

### Custom Data Path
## Project Structure

```
├── bnair_lab3-1.py          # Main training script
├── README.md                # This file
│
├── data/                    # Input data directory
│
├── plots/                   # Output visualizations
│   ├── training_curves/     # Per-model loss/accuracy curves
│   ├── confusion_matrices/  # Test set confusion matrices
│   └── comparison_curves/   # Multi-model comparisons
│
├── metrics/                 # CSV files with evaluation metrics
│   └── *.csv               # Per-model performance summaries
│
├── histories/              # Training history logs
│   └── *.csv              # Per-epoch metrics (loss, acc, f1, auc)
│
└── results.csv            # Final summary table across all models
    detailed.csv           # Detailed metrics (train/val/test splits)
```

## Model Architecture Details

### Wav2Vec2 Models
- **Backbone**: `facebook/wav2vec2-base` (95M parameters, pre-trained on 960h speech)
- **Input Processing**: Signals resampled to 1000 samples, then processed as audio at 16kHz
- **Classification Head**:
  - **Adapter**: Bottleneck module (256/512 hidden dims)
  - **LoRA**: Applied to Q, K, V, Out projections (rank 8-16)

### HuBERT Models
- **Backbone**: `facebook/hubert-base` (95M parameters, self-supervised)
- **Input Processing**: Same signal resampling strategy
- **Classification Head**: Same adapter/LoRA configurations

### ECG-FM Models
- **Backbone**: Domain-specific ECG foundation model
- **Advantage**: Pre-trained on ECG-specific objectives
- **Classification Head**: Adapter or LoRA on ECG representation

## Data Preprocessing

1. **Signal Normalization**: Row-wise z-score normalization (per ECG signal)
2. **Resampling**: Linear interpolation to 1000 samples (target length)
3. **Batch Processing**: Per-batch z-score + upsampling to 16kHz equivalent
4. **Padding**: Processor handles padding to max sequence length in batch

## Training Strategy

### Training Loop
- **Optimizer**: AdamW with weight decay
- **Loss Function**: Binary Cross-Entropy with Logits
- **Early Stopping**: Patience=3 (stop if val_loss doesn't improve for 3 epochs)
- **Learning Rates**:
  - Adapter models: 5e-5
  - LoRA models: 2.5e-5
  - ECG-FM models: Same rates as Wav2Vec2/HuBERT

### Evaluation Metrics
| Metric | Description |
|--------|-------------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1-Score** | 2 * (Precision * Recall) / (Precision + Recall) |
| **AUC-ROC** | Area under Receiver Operating Characteristic curve |
| **Confusion Matrix** | TP, TN, FP, FN counts and percentages |

## Output Files

### Summary Results
- **`results.csv`**: Model comparison table (Val Acc, Val F1, Val AUC, Test Acc)
- **`detailed.csv`**: Full metrics for all models/splits (Precision, Recall, F1, AUC)

### Visualizations (in `plots/`)
- **Training Curves**: Loss and accuracy per epoch (6 plots)
- **Comparison Curves**: Stacked comparison across all 6 models
- **Confusion Matrices**: Per-model validation set confusion matrices

### Training Logs (in `histories/`)
- **CSV Format**: One file per model with per-epoch metrics
  - Columns: `epoch`, `train_loss`, `train_acc`, `val_loss`, `val_acc`, `val_f1`, `val_auc`

## Key Implementation Details

### Parameter Efficiency
```
Wav2Vec2 (95M params)
├── Frozen backbone: ~94.8M params
└── Adapter module: 100K-500K trainable params (~0.1%)

LoRA Configuration (rank=8):
├── Query/Key/V/Out projections: 8×768 = 6,144 params per head
└── Total LoRA params: ~50K-200K (~0.2%)
```

### Device Handling
- Automatic CUDA detection
- Falls back to Apple Silicon (MPS) if available
- Falls back to CPU if neither GPU available

### Reproducibility
- Fixed seed: `SEED = 42` (NumPy, PyTorch, Random)
- Stratified train/val/test splits
- Deterministic operations on CUDA (when applicable)

## Expected Runtime

| Model | Device | Time (10 epochs) |
|-------|--------|------------------|
| Wav2Vec2/HuBERT + Adapter | NVIDIA GPU | 5-10 min |
| Wav2Vec2/HuBERT + LoRA | NVIDIA GPU | 5-10 min |
| ECG-FM + Adapter/LoRA | NVIDIA GPU | 3-8 min |
| All models | CPU | 20-40 min |
| Hyperparameter tuning (TUNING_MODE=True) | GPU | +30-60 min |

## Typical Results

Expected performance ranges (varies by data distribution):

| Model | Val Accuracy | Test Accuracy | Val F1 |
|-------|-------------|---------------|--------|
| Wav2Vec2 + Adapter | 0.78-0.85 | 0.76-0.83 | 0.75-0.83 |
| Wav2Vec2 + LoRA | 0.79-0.86 | 0.77-0.84 | 0.76-0.84 |
| HuBERT + Adapter | 0.80-0.87 | 0.78-0.85 | 0.78-0.85 |
| HuBERT + LoRA | 0.81-0.88 | 0.79-0.86 | 0.79-0.86 |
| ECG-FM + Adapter | 0.82-0.89 | 0.80-0.87 | 0.80-0.87 |
| ECG-FM + LoRA | 0.83-0.90 | 0.81-0.88 | 0.81-0.88 |

## Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size
batch_size = 16  # Instead of 32/64

# Or reduce model size by using smaller adapters
adapter_hiddens = [128]  # Instead of [256, 512]
```

### CUDA/GPU Issues
```python
# Force CPU
dev = torch.device("cpu")

# Or disable CUDA
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```

### Data Not Found
```python
# Check file path
csv_path = "/absolute/path/to/df_segment2.csv"

# Or use current directory
csv_path = "./df_segment2.csv"
```

### Poor Performance
1. Verify data preprocessing (check for NaN/inf values)
2. Increase training epochs (currently 10)
3. Try different learning rates
4. Ensure label distribution is reasonable (not severely imbalanced)

## Citation & References

**PEFT Methods:**
- LoRA: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- Adapters: [Houlsby et al., 2019](https://arxiv.org/abs/1902.00751)

**Foundation Models:**
- Wav2Vec2: [Baevski et al., 2020](https://arxiv.org/abs/2006.11477)
- HuBERT: [Hsu et al., 2021](https://arxiv.org/abs/2106.07447)

## License

This project is provided as-is for educational and research purposes.

## Contact & Support

For questions, issues, or suggestions, please refer to the original Colab notebook or reach out to the project maintainers.

---

**Last Updated**: June 2026  
**Python Version**: 3.8+  
**Framework Versions**: PyTorch 1.13+, Transformers 4.25+, PEFT 0.4+
