# Fine-tuning-Large-Language-Models-for-Task-Specific-Performance

# Large Model Fine-tuning Framework

This repository provides implementations for fine-tuning large language models using three different methods: **LoRA, RoSA, and GaLore**. We apply these methods to two benchmark tasks: **GSM8K (Mathematical Reasoning)** and **News Summary (Text Summarization)**.

The framework supports both **training and evaluation**, making it easy to experiment with different fine-tuning strategies.

## Features

- Supports three fine-tuning methods:
  - [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation)
  - [RoSA](https://arxiv.org/abs/2401.04679) (Robust Adaptation)
  - [GaLore](https://arxiv.org/abs/2401.04679) (Gradient Low-Rank Adaptation)
- Fine-tuning for two tasks:
  - GSM8K (Grade School Math 8K dataset)
  - News Summarization
- Configurable hyperparameters for easy experimentation
- Supports evaluation and inference with trained models

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (for GPU training)
- Transformers 4.39.0 (Hugging Face)
- Datasets (Hugging Face)

### Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
