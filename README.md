# ResNet-18 with Channel Attention for Malaria Detection

A PyTorch implementation comparing ResNet-18 with and without channel attention mechanisms for malaria cell classification. This project demonstrates how attention mechanisms can improve feature learning in medical image classification tasks.

## Overview

This project implements a channel attention mechanism integrated into ResNet-18 for binary classification of malaria-infected vs. healthy blood cells. The implementation compares the performance of:

- **ResNet-18 with Channel Attention**: Enhanced with custom attention gates
- **Plain ResNet-18**: Standard ResNet-18 architecture

## Features

- Custom Channel Attention module with GLU activation
- Medical image classification (malaria detection)
- Comprehensive performance comparison
- Transfer learning with progressive unfreezing
- Multiple evaluation metrics (Accuracy, F1-score, Loss)

## Architecture

### Channel Attention Module

The attention mechanism uses:
- Adaptive average pooling for global context
- Feature dimension reduction (16x by default)
- Gated Linear Unit (GLU) activation
- Dropout for regularization
- Sigmoid gating for channel-wise attention weights

```python
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16, dropout=0.1):
        # Attention mechanism implementation
```

### ResNet Integration

Attention gates are strategically placed after the second convolution in each ResNet block, allowing the network to focus on the most relevant feature channels for malaria detection.

## Dataset

Uses the **Cell Images for Detecting Malaria** dataset from Kaggle:
- **Classes**: Parasitized (malaria-infected) vs. Uninfected cells
- **Split**: 70% training, 15% validation, 15% testing
- **Preprocessing**: Resize to 224×224, normalization, data augmentation

## Installation

```bash
# Install required packages
pip install torch torchvision kagglehub scikit-learn tqdm
```

## Usage

### Quick Start

```python
# Run the complete analysis
python attention_in_resnet_18.py
```

### Custom Configuration

```python
# Analyze with different unfrozen layers and epochs
analyse(train_loader, val_loader, test_loader, 
        unfreeze_layers=2, epochs=5)
```

### Progressive Training Strategy

The project implements a progressive unfreezing approach:

1. **Frozen backbone** (0 layers): Only train classifier and attention
2. **Unfreeze layer4** (1 layer): Fine-tune top layer
3. **Unfreeze layer3-4** (2 layers): Fine-tune top two layers
4. **Unfreeze layer2-4** (3 layers): Fine-tune top three layers
5. **Full unfreezing** (4 layers): Fine-tune entire network

## Experiments

The code automatically runs experiments with different unfreezing strategies:

```python
# Progressive unfreezing experiments
for layers in [0, 1, 2, 3, 4]:
    analyse(train_loader, val_loader, test_loader, 
            unfreeze_layers=layers, epochs=3)
```

## Model Comparison

| Model | Features |
|-------|----------|
| **ResNet-18 + Attention** | • Channel attention gates<br>• GLU activation<br>• Dropout regularization |
| **Plain ResNet-18** | • Standard architecture<br>• Transfer learning<br>• Baseline comparison |

## Key Components

### Training Pipeline
- Cross-entropy loss optimization
- Adam optimizer with 1e-4 learning rate
- Comprehensive metric tracking

### Data Augmentation
- Random horizontal flips
- Random rotation (±10°)
- Standard ImageNet normalization

### Evaluation Metrics
- **Accuracy**: Overall classification performance
- **F1-Score**: Balanced precision-recall metric
- **Loss**: Cross-entropy loss tracking

## Results

The project provides detailed comparisons showing:
- Training/validation curves for both models
- Final test performance metrics
- Impact of progressive unfreezing on performance

## Technical Details

### Hardware Requirements
- CUDA-compatible GPU (recommended)
- Automatic CPU fallback available

### Memory Considerations
- Batch size: 64 (adjustable based on GPU memory)
- Image resolution: 224×224×3
- Model parameters: ~11M (ResNet-18 base)

## Contributing

Feel free to contribute by:
- Experimenting with different attention mechanisms
- Adding more evaluation metrics
- Implementing other backbone architectures
- Improving data augmentation strategies

## License

This project is open source. The dataset is provided by Kaggle under their respective terms.

## Citation

If you use this code in your research, please consider citing:

```bibtex
@misc{resnet_attention_malaria,
  title={ResNet-18 with Channel Attention for Malaria Detection},
  year={2024},
  howpublished={\url{https://github.com/your-repo/resnet-attention-malaria}}
}
```

---

**Note**: This implementation demonstrates the effectiveness of attention mechanisms in medical image classification. The channel attention module helps the network focus on discriminative features crucial for malaria detection.
