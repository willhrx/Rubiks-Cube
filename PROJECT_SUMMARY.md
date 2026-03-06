# Rubik's Cube ML Project Summary

This project implements a machine learning approach to solving the Rubik's Cube puzzle using reinforcement learning. The project is structured as a comprehensive Python package with the following components:

## Core Components

1. **Cube Representation** (`rubiks_cube_ml/cube/`)
   - `RubiksCube` class for representing the 3x3x3 Rubik's Cube state
   - `Move` class for representing cube rotations (R, L, U, D, F, B and their variants)
   - Support for scrambling, checking if solved, and applying moves

2. **Visualization** (`rubiks_cube_ml/visualization/`)
   - 3D visualization of the cube using Matplotlib
   - Animation support for visualizing solution sequences
   - Solution path visualization showing step-by-step progress

3. **Reinforcement Learning Model** (`rubiks_cube_ml/model/`)
   - Custom Gym environment for Rubik's Cube
   - Neural network policy model with separate policy and value heads
   - Greedy policy for inference after training

4. **Training Pipeline** (`rubiks_cube_ml/training/`)
   - Base trainer class with common functionality
   - PPO (Proximal Policy Optimization) implementation
   - Checkpointing and TensorBoard logging

5. **Evaluation** (`rubiks_cube_ml/evaluation/`)
   - Evaluation metrics for solve rate at different difficulty levels
   - Visualization of evaluation results
   - Solution demonstration

## Key Features

- **Full 3D Representation**: The cube is represented as a 6×3×3 numpy array for accurate state tracking.
- **Reinforcement Learning**: Uses PPO to train a policy that learns to solve cubes of increasing difficulty.
- **Visualization**: Includes 3D rendering of the cube and solution animations.
- **Comprehensive Testing**: Includes unit tests for the core cube operations.
- **Installable Package**: Structured as a proper Python package with setup.py.

## Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rubiks-cube-ml.git
cd rubiks-cube-ml

# Install the package
pip install -e .
```

### Training a Model

```bash
python -m rubiks_cube_ml train --num_iterations 1000 --steps_per_iteration 2048
```

### Demonstrating a Solution

```bash
python -m rubiks_cube_ml demo --model checkpoints/model_best.pt --scramble_steps 10
```

### Evaluating a Model

```bash
python -m rubiks_cube_ml evaluate --model checkpoints/model_best.pt --difficulty 20
```

## Project Structure

```
rubiks_cube_ml/
├── __init__.py
├── cube/
│   ├── __init__.py
│   ├── cube.py         # Cube representation
│   └── moves.py        # Move definitions
├── visualization/
│   ├── __init__.py
│   └── visualizer.py   # 3D visualization
├── model/
│   ├── __init__.py
│   ├── environment.py  # Gym environment
│   └── policy.py       # Neural network policy
├── training/
│   ├── __init__.py
│   ├── trainer.py      # Base trainer
│   └── ppo_trainer.py  # PPO implementation
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py    # Evaluation metrics
├── demo.py             # Demo script
└── train.py            # Training script
```

## Notes on Implementation

- The environment rewards solving the cube with +10 and penalizes each step with -0.1 to encourage efficient solutions.
- The neural network policy uses a 1024-dimensional hidden layer and outputs both action probabilities and a value estimate.
- The training process uses Generalized Advantage Estimation (GAE) for more stable learning.
- The 3D visualization uses Matplotlib's 3D plotting capabilities to render the cube.

## Future Improvements

- Implement more advanced RL algorithms like MuZero
- Add support for different cube sizes (2×2×2, 4×4×4, etc.)
- Optimize the code for faster training and inference
- Add a web interface for interactive demonstrations
- Implement a pattern-based approach for more efficient solutions