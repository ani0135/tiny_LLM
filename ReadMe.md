# Tiny LLM - Language Model Training and Inference

Welcome to the Tiny LLM project! This repository provides a simple yet powerful framework for training and using a character-level language model (LLM) using Shakespeare's vocabulary. The project includes everything you need to get started with training your own model, fine-tuning hyperparameters, and generating text through inference.

## Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
- [Training the Model](#training-the-model)
- [Configuring Hyperparameters](#configuring-hyperparameters)
- [Inference](#inference)

## Overview

The model is trained on **Shakespeare's vocabulary**, and the dataset is located in the `data/` directory. A **Character-level Tokenizer** is used for tokenizing the text, allowing the model to understand characters rather than whole words.

## Getting Started

### Prerequisites

To get started with this project, you'll need to have the following installed:

- Python 3.7+ (Python 3.8+ recommended)
- Pip
- Virtual environment (optional, but recommended)

### Install Dependencies

1. Clone the repository:

   ```bash
   git clone https://github.com/ani0135/tiny_LLM.git
   cd tiny_LLM
2. Install the required Python packages:
    ``` bash
    pip install -r requirements.txt

## Training the Model

To train the language model, use the train.py script. This script will load the Shakespeare dataset, tokenize it, and train a character-level model.

### Running the Training

1. Make sure the data is in the data/ folder. It should include the text data of Shakespeare's works.

2. Run the training script:
    ``` bash
    python train.py

This will start the training process. Make sure you have enough resources (like GPU) for optimal training speed.

## Configuring Hyperparameters

The training process and model architecture can be customized by editing the `config.py` file. This file contains various hyperparameters for the model, such as:

- **Batch Size**: Number of samples per batch.
- **Learning Rate**: The learning rate for the optimizer.
- **Epochs**: Number of training epochs.
- **Model Architecture**: Adjust the number of layers, hidden units, etc.

Modify the `config.py` as per your requirements to fine-tune the model's performance.

---

## Inference

Once the model is trained, you can use the `inference.py` script to generate text.

### Running Inference

To generate text using the trained model, use:

```bash

python inference.py
