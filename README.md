"""
# AI Blackjack Player

An AI agent that learns how to play Blackjack using PyTorch and deep reinforcement learning. This includes a realalistic simulated blackjack training evniornment, card dataset builder for consistant input data, and benchmarking modules to evaluate AI performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Dataset Builder](#dataset-builder)
  - [Training](#training)
  - [Multithreaded Training](#multithreaded-training)
  - [Testing / Benchmarking](#testing--benchmarking)
- [Ruleset](#ruleset)
- [Constants](#constants)
- [Requirements](#requirements)
- [Note](#note)

## Project Overview
This project implements an AI agent that learns to play Blackjack using deep reinforcement learning with PyTorch. The AI is trained using datasets generated via Monte Carlo simulations, which provide a dataset of cards that can be benchmarked for comparing models performance.

This project is **experimental** and not intended to be reproducible or production-ready. The goal is to explore reinforcement learning on Blackjack, with my best training result being 1% loss vs the 0.5% theoretical optimal loss for the given ruleset.

## Features
- Generate datasets of Blackjack games for training
- Train reinforcement learning models using PyTorch, (with GPU acceleration via Nvidia CUDA capable card)
- Support for multithreaded training to create multiple models efficiently (potential improvement with better GPU utalization)
- Benchmark trained models against test datasets

## Installation
Clone the repository and install the dependencies:

```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
```


## Usage

### Dataset Builder
`datasetBuilder.py` generates Blackjack game data for training the AI model, ensuring training and tests between models are comparable.

```bash
python agents/datasetBuilder.py
```

### Training
`fixedTraining.py` trains the AI model using a dataset created by the builder and saves the trained model.

```bash
python agents/fixedTraining.py
```

### Multithreaded Training
Optionally, train multiple models concurrently using:

```bash
python agents/multithreadedTraining.py
```

### Testing / Benchmarking
`NNTest.py` benchmarks a saved model against a saved dataset. To avoid overfitting, create a different dataset for testing.

```bash
python agents/NNTest.py
```
## Ruleset: 
PNG IMAGE HERE

## Requirements
- Python 3.10+
- torch >= 2.5.0
- numpy >= 1.26
- matplotlib >= 3.9
- tqdm



## Note
This project is **experimental and for personal exploration**. It is not intended to be fully reproducible or production-ready. Expect rough edges and hardcoded paths. The trained AI achieves approximately 1% loss compared to the 0.5% theoretical optimal loss in Blackjack with the simulated rule set.


## License
MIT License
