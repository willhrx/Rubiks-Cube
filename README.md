# Rubik's Cube ML Solver

A machine learning project for solving Rubik's Cube using reinforcement learning.

## Project Overview

This project implements a reinforcement learning approach to solving the Rubik's Cube puzzle. It includes:

- A representation of the Rubik's Cube state
- Visualization of the cube in 3D
- A reinforcement learning model
- Training pipeline
- Evaluation metrics
- Demo script for showing solutions

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rubiks-cube-ml.git
cd rubiks-cube-ml

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python -m rubiks_cube_ml.train
```

### Visualizing a Solution

```bash
python -m rubiks_cube_ml.demo
```

## Project Structure

- `rubiks_cube_ml/` - Main package
  - `cube/` - Cube representation and operations
  - `visualization/` - 3D visualization of cube
  - `model/` - Reinforcement learning model
  - `training/` - Training pipeline
  - `evaluation/` - Metrics and evaluation
  - `demo.py` - Demonstration script
  