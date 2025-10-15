"""
# AI Blackjack Player

An AI agent that learns how to play Blackjack using PyTorch and deep reinforcement learning. This project includes simulation, training, and benchmarking modules to evaluate AI performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Dataset Builder](#dataset-builder)
  - [Training](#training)
  - [Multithreaded Training](#multithreaded-training)
  - [Testing / Benchmarking](#testing--benchmarking)
- [Constants](#constants)
- [Requirements](#requirements)
- [File Structure](#file-structure)
- [Note](#note)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project implements an AI agent that learns to play Blackjack using deep reinforcement learning with PyTorch. The AI is trained using datasets generated via Monte Carlo simulations and can be benchmarked against saved datasets or models.

This project is **experimental** and not intended to be reproducible or production-ready. The goal is to explore reinforcement learning on Blackjack, with a reported result of around 1% loss vs the 0.5% theoretical optimal loss.

## Features
- Generate datasets of Blackjack games for training
- Train reinforcement learning models using PyTorch
- Support for multithreaded training to create multiple models efficiently
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
`datasetBuilder.py` generates Blackjack game data for training the AI model.

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
`NNTest.py` benchmarks a saved model against a saved dataset.

```bash
python agents/NNTest.py
```

## Constants

### datasetBuilder.py
```python
PERFORMANCE_DEBUG = False
OUTPUT_FILEPATH = "../data/card_dataset"
NUM_HANDS = 1000000
```

### fixedTraining.py
```python
PERFORMANCE_DEBUG = False
INPUT_HANDS = "../data/card_dataset.json"
AGENT_OUTPUT = "../outputs/saved_nns/model.pth"
```

### multithreadedTraining.py
```python
INPUT_HANDS = "../data/card_dataset.json"
LOG_OUTPUT = "../outputs/logs"
AGENT_OUTPUT = "../outputs/saved_nns"
```

### NNTest.py
```python
INPUT_HANDS = "../data/test_dataset.json"
NN_PATH = "../outputs/saved_nns/model_4.pth"
```

## Requirements
- Python 3.10+
- torch >= 2.5.0
- numpy >= 1.26
- matplotlib >= 3.9
- tqdm



## Note
This project is **experimental and for personal exploration**. It is not intended to be fully reproducible or production-ready. Expect rough edges and hardcoded paths. The trained AI achieves approximately 1% loss compared to the 0.5% theoretical optimal loss in Blackjack.


## License
MIT License
