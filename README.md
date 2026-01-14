# CNN Architecture Comparison with MNIST

Perbandingan arsitektur CNN (LeNet-5, AlexNet, GoogleNet, ResNet) pada dataset MNIST.

## 🚀 Quick Start

1. Buka [Google Colab](https://colab.research.google.com/)
2. Upload file `CNN_Comparison_MNIST.ipynb`
3. Pastikan Runtime > Change runtime type > **GPU**
4. Run All (Ctrl+F9)

## 📁 File Structure

```
├── CNN_Comparison_MNIST.ipynb  # Notebook untuk Google Colab
├── CNN_Comparison_MNIST.py     # Python script version
└── results/                    # Output directory
    ├── comparison_plots.png
    └── confusion_matrices.png
```

## 🧠 Arsitektur yang Dibandingkan

| Model | Paper | Tahun |
|-------|-------|-------|
| LeNet-5 | LeCun et al. | 1998 |
| AlexNet | Krizhevsky et al. | 2012 |
| GoogleNet | Szegedy et al. | 2014 |
| ResNet | He et al. | 2015 |

## ⚙️ Configuration

```python
CONFIG = {
    'batch_size': 64,
    'epochs': 15,
    'learning_rate': 0.001,
    'image_size': 32,
    'train_split': 0.8,
    'patience': 5  # Early stopping
}
```

## 📊 Features

- ✅ Fair comparison (same preprocessing, hyperparameters)
- ✅ Early stopping & LR scheduler
- ✅ Comprehensive metrics (Accuracy, F1, Precision, Recall)
- ✅ Visualizations (Training curves, Confusion matrices)
- ✅ Per-class analysis
- ✅ Trade-off analysis (Accuracy vs Parameters)
