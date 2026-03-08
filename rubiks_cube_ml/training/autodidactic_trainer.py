"""Autodidactic Iteration trainer for Rubik's Cube (DeepCubeA style)."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
from tqdm import tqdm

from ..cube.cube import RubiksCube
from ..cube.moves import MOVES, Move, MoveType, get_inverse_move
from ..model.policy import CubePolicy


class AutodidacticTrainer:
    """
    Implements Autodidactic Iteration from the DeepCubeA paper.

    Key insight: Instead of learning from random exploration, generate
    training data by scrambling FROM the solved state. This guarantees
    we always know the optimal distance to the goal.

    Training data generation:
    1. Start from solved cube
    2. Apply K random moves (K sampled from 1 to max_depth)
    3. Value target = -K (negative cost-to-go)
    4. Policy target = inverse of last move (optimal action)
    """

    def __init__(self,
                 policy: CubePolicy,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 max_scramble_depth: int = 30,
                 batch_size: int = 1024,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 value_weight: float = 1.0,
                 policy_weight: float = 1.0,
                 checkpoint_dir: str = 'checkpoints',
                 log_dir: str = 'logs'):
        """
        Initialize the Autodidactic trainer.

        Args:
            policy: Policy network to train
            device: Device to use for training
            max_scramble_depth: Maximum scramble depth for training data
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
            value_weight: Weight for value loss
            policy_weight: Weight for policy loss
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
        """
        self.policy = policy.to(device)
        self.device = device
        self.max_scramble_depth = max_scramble_depth
        self.batch_size = batch_size
        self.value_weight = value_weight
        self.policy_weight = policy_weight
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100000, gamma=0.5
        )

        # Build action mappings
        self._build_action_mappings()

        # Training stats
        self.total_iterations = 0
        self.best_solve_rate = 0.0

    def _build_action_mappings(self) -> None:
        """Build mappings between actions and moves."""
        # Use first 12 moves (excluding double moves for simpler action space)
        move_list = list(MOVES.values())

        # Filter to only clockwise and counterclockwise (no double)
        self.action_to_move: List[Move] = []
        for move in move_list:
            if move.move_type != MoveType.DOUBLE:
                self.action_to_move.append(move)
            if len(self.action_to_move) >= 12:
                break

        # Build inverse mapping
        self.move_to_action: Dict[str, int] = {}
        for idx, move in enumerate(self.action_to_move):
            key = f"{move.name}_{move.move_type.name}"
            self.move_to_action[key] = idx

    def _get_action_for_move(self, move: Move) -> int:
        """Get action index for a move."""
        key = f"{move.name}_{move.move_type.name}"
        return self.move_to_action.get(key, 0)

    def _get_inverse_action(self, action: int) -> int:
        """Get the action index for the inverse of the given action."""
        move = self.action_to_move[action]
        inverse_move = get_inverse_move(move)
        return self._get_action_for_move(inverse_move)

    def generate_training_batch(self) -> Dict[str, torch.Tensor]:
        """
        Generate a batch of training examples.

        For each example:
        - Scramble a solved cube with K random moves
        - State = scrambled cube
        - Value target = -K (cost to solve)
        - Policy target = inverse of last move (one step closer)

        Returns:
            Dictionary with 'states', 'value_targets', 'policy_targets'
        """
        states = []
        value_targets = []
        policy_targets = []
        weights = []

        for _ in range(self.batch_size):
            # Random depth from 1 to max_scramble_depth
            depth = np.random.randint(1, self.max_scramble_depth + 1)

            # Start from solved cube
            cube = RubiksCube()

            # Apply random moves
            last_action = 0
            for _ in range(depth):
                action = np.random.randint(0, 12)
                move = self.action_to_move[action]
                cube.apply_move(move)
                last_action = action

            # Get state representation
            state = cube.get_state_representation()

            # Value target: negative depth (cost-to-go)
            # Normalize by max_depth for better learning
            value_target = -depth / self.max_scramble_depth

            # Policy target: inverse of last move
            inverse_action = self._get_inverse_action(last_action)

            # Weight: harder examples (deeper scrambles) get more weight
            weight = 1.0 / depth  # Or could use uniform weights

            states.append(state)
            value_targets.append(value_target)
            policy_targets.append(inverse_action)
            weights.append(weight)

        return {
            'states': torch.FloatTensor(np.array(states)).to(self.device),
            'value_targets': torch.FloatTensor(value_targets).unsqueeze(1).to(self.device),
            'policy_targets': torch.LongTensor(policy_targets).to(self.device),
            'weights': torch.FloatTensor(weights).to(self.device)
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            batch: Training batch from generate_training_batch

        Returns:
            Dictionary of loss values
        """
        self.policy.train()

        # Forward pass
        action_probs, values = self.policy(batch['states'])

        # Value loss (MSE)
        value_loss = F.mse_loss(values, batch['value_targets'])

        # Policy loss (cross entropy with inverse moves as targets)
        # Using log probabilities for numerical stability
        log_probs = torch.log(action_probs + 1e-8)
        policy_loss = F.nll_loss(log_probs, batch['policy_targets'])

        # Combined loss
        total_loss = (self.value_weight * value_loss +
                     self.policy_weight * policy_loss)

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()

        return {
            'total_loss': total_loss.item(),
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }

    def train(self,
              num_iterations: int = 100000,
              eval_interval: int = 1000,
              checkpoint_interval: int = 10000,
              eval_episodes: int = 100,
              eval_scramble_depths: List[int] = None) -> Dict[str, List[float]]:
        """
        Main training loop.

        Args:
            num_iterations: Total training iterations
            eval_interval: How often to evaluate
            checkpoint_interval: How often to save checkpoints
            eval_episodes: Episodes per evaluation
            eval_scramble_depths: Scramble depths to evaluate on

        Returns:
            Dictionary of training history
        """
        if eval_scramble_depths is None:
            eval_scramble_depths = [1, 3, 5, 7, 10, 15, 20]

        history = {
            'total_loss': [],
            'value_loss': [],
            'policy_loss': [],
            'solve_rates': []
        }

        pbar = tqdm(range(num_iterations), desc="Training")

        for iteration in pbar:
            # Generate batch and train
            batch = self.generate_training_batch()
            losses = self.train_step(batch)

            self.total_iterations += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total_loss']:.4f}",
                'v_loss': f"{losses['value_loss']:.4f}",
                'p_loss': f"{losses['policy_loss']:.4f}"
            })

            # Record losses
            history['total_loss'].append(losses['total_loss'])
            history['value_loss'].append(losses['value_loss'])
            history['policy_loss'].append(losses['policy_loss'])

            # Evaluation
            if (iteration + 1) % eval_interval == 0:
                solve_rates = self.evaluate(eval_episodes, eval_scramble_depths)
                avg_rate = np.mean(list(solve_rates.values()))
                history['solve_rates'].append(avg_rate)

                print(f"\nIteration {iteration + 1}:")
                for depth, rate in solve_rates.items():
                    print(f"  Depth {depth}: {rate:.1%}")
                print(f"  Average: {avg_rate:.1%}")

                # Save best model
                if avg_rate > self.best_solve_rate:
                    self.best_solve_rate = avg_rate
                    self.save_checkpoint('best_model.pt')

            # Checkpoint
            if (iteration + 1) % checkpoint_interval == 0:
                self.save_checkpoint(f'checkpoint_{iteration + 1}.pt')

        return history

    def evaluate(self,
                 num_episodes: int = 100,
                 scramble_depths: List[int] = None,
                 max_solve_steps: int = 50) -> Dict[int, float]:
        """
        Evaluate the policy on various scramble depths.

        Args:
            num_episodes: Episodes per depth
            scramble_depths: List of scramble depths to test
            max_solve_steps: Maximum steps to attempt solving

        Returns:
            Dictionary mapping depth -> solve rate
        """
        if scramble_depths is None:
            scramble_depths = [1, 3, 5, 10, 15, 20]

        self.policy.eval()
        results = {}

        for depth in scramble_depths:
            solved_count = 0

            for _ in range(num_episodes):
                # Create and scramble cube
                cube = RubiksCube()
                for _ in range(depth):
                    action = np.random.randint(0, 12)
                    cube.apply_move(self.action_to_move[action])

                # Try to solve
                for _ in range(max_solve_steps):
                    if cube.is_solved():
                        solved_count += 1
                        break

                    # Get action from policy
                    state = cube.get_state_representation()
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        action_probs, _ = self.policy(state_tensor)
                        action = torch.argmax(action_probs, dim=1).item()

                    # Apply action
                    cube.apply_move(self.action_to_move[action])

            results[depth] = solved_count / num_episodes

        self.policy.train()
        return results

    def save_checkpoint(self, filename: str) -> None:
        """Save a training checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'total_iterations': self.total_iterations,
            'best_solve_rate': self.best_solve_rate,
        }, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, filename: str) -> None:
        """Load a training checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)

        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.total_iterations = checkpoint['total_iterations']
        self.best_solve_rate = checkpoint.get('best_solve_rate', 0.0)

        print(f"Loaded checkpoint from {path}")
        print(f"  Iterations: {self.total_iterations}")
        print(f"  Best solve rate: {self.best_solve_rate:.1%}")
