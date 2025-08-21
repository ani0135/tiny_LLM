# Tiny LLM - Language Model Training and Inference

Welcome to the Tiny LLM project! This repository provides a simple yet powerful framework for training and using a character-level language model (LLM) using Shakespeare's vocabulary. The project includes everything you need to get started with training your own model, fine-tuning hyperparameters, and generating text through inference.

## Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
- [Training the Model](#training-the-model)
- [Configuring Hyperparameters](#configuring-hyperparameters)
- [Inference](#inference)
- [Troubleshooting](#troubleshooting)
- [License](#license)

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
