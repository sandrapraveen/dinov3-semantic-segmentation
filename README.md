# DINOv3 Semantic Segmentation

PyTorch project exploring semantic segmentation using DINOv3 features and custom dense prediction architectures.

## Features

- DINOv3 transformer backbone
- Custom decoder architectures
- decoder experiments
- Boundary-aware loss experiments
- Semantic segmentation training pipeline
- Inference and visualization utilities

## Project Status

🚧 Work in Progress

Currently experimenting with:
- Decoder architectures
- Fine-tuning strategies
- Boundary-aware losses
- Multi-scale feature fusion

## Project Structure

```bash
src/
├── decoders/
├── losses/
├── dataset.py
├── model.py
├── engine.py
└── utils.py
```

## Installation

```bash
git clone https://github.com/yourusername/dinov3-semantic-segmentation.git
cd dinov3-semantic-segmentation

pip install -r requirements.txt
```

## Dataset

Using the LoveDA dataset.

Place dataset in:

```bash
loveda/
```

## Notes

This repository does not include:
- datasets
- pretrained weights
- checkpoints

## References

- DINOv3
- DPT (Dense Prediction Transformers)
- LoveDA Dataset
