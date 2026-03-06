"""Base trainer for Rubik's Cube."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
import os
from tqdm import tqdm

from ..model.policy import CubePolicy
from ..model.environment import RubiksCubeEnv
from ..cube.cube import RubiksCube


class CubeTrainer:
    """Base class for training Rubik's Cube solvers."""
    
    def __init__(self, 
                 policy: CubePolicy,
                 env: RubiksCubeEnv,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 1e-4,
                 log_dir: str = 'logs',
                 checkpoint_dir: str = 'checkpoints'):
        """
        Initialize the trainer.
        
        Args:
            policy: Policy network to train
            env: Environment to train in
            device: Device to use for training
            learning_rate: Learning rate for the optimizer
            log_dir: Directory to save TensorBoard logs
            checkpoint_dir: Directory to save model checkpoints
        """
        self.policy = policy
        self.env = env
        self.device = device
        self.policy.to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
        
        # Directories
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Training stats
        self.episodes = 0
        self.steps = 0
        self.best_reward = -float('inf')
    
    def train(self, num_episodes: int, max_steps_per_episode: int) -> Dict[str, Any]:
        """
        Train the policy.
        
        Args:
            num_episodes: Number of episodes to train for
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            Dictionary of training statistics
        """
        raise NotImplementedError("Subclasses must implement train")
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the current policy.
        
        Args:
            num_episodes: Number of episodes to evaluate on
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.policy.eval()
        
        total_reward = 0.0
        solved_count = 0
        steps_to_solve = []
        
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0.0
            step_count = 0
            
            while not done:
                # Convert observation to tensor
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                # Get action probabilities
                with torch.no_grad():
                    action_probs, _ = self.policy(obs_tensor)
                    action = torch.argmax(action_probs, dim=1).item()
                
                # Take a step
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                step_count += 1
                
                if info['solved']:
                    solved_count += 1
                    steps_to_solve.append(step_count)
            
            total_reward += episode_reward
        
        # Calculate evaluation metrics
        avg_reward = total_reward / num_episodes
        solve_rate = solved_count / num_episodes
        
        if steps_to_solve:
            avg_steps_to_solve = sum(steps_to_solve) / len(steps_to_solve)
        else:
            avg_steps_to_solve = float('inf')
        
        # Log evaluation metrics
        self.writer.add_scalar('Eval/AvgReward', avg_reward, self.episodes)
        self.writer.add_scalar('Eval/SolveRate', solve_rate, self.episodes)
        
        if steps_to_solve:
            self.writer.add_scalar('Eval/AvgStepsToSolve', avg_steps_to_solve, self.episodes)
        
        self.policy.train()
        
        return {
            'avg_reward': avg_reward,
            'solve_rate': solve_rate,
            'avg_steps_to_solve': avg_steps_to_solve if steps_to_solve else None
        }
    
    def save_checkpoint(self, is_best: bool = False) -> str:
        """
        Save a model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
            
        Returns:
            Path to the saved checkpoint
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f'model_{self.episodes}.pt')
        
        # Save the model
        torch.save({
            'episodes': self.episodes,
            'steps': self.steps,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_reward': self.best_reward
        }, checkpoint_path)
        
        # If this is the best model, save a copy
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pt')
            torch.save({
                'episodes': self.episodes,
                'steps': self.steps,
                'policy_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_reward': self.best_reward
            }, best_path)
        
        return checkpoint_path
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load a model checkpoint.
        
        Args:
            path: Path to the checkpoint
            
        Returns:
            Dictionary of loaded checkpoint data
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episodes = checkpoint['episodes']
        self.steps = checkpoint['steps']
        self.best_reward = checkpoint['best_reward']
        
        return checkpoint