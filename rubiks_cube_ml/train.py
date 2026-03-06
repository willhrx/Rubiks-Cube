"""
Training script for Rubik's Cube solver.

This script trains a reinforcement learning model to solve the Rubik's Cube.

Usage:
    python -m rubiks_cube_ml.train --num_iterations <n> --steps_per_iteration <n>
"""

import argparse
import torch
import numpy as np
import os
import time

from .model.policy import CubePolicy
from .model.environment import RubiksCubeEnv
from .training.ppo_trainer import PPOTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training script for Rubik's Cube solver")
    
    # Training parameters
    parser.add_argument("--num_iterations", type=int, default=1000,
                       help="Number of training iterations")
    parser.add_argument("--steps_per_iteration", type=int, default=2048,
                       help="Number of environment steps per iteration")
    parser.add_argument("--eval_interval", type=int, default=10,
                       help="Evaluation interval (in iterations)")
    
    # Environment parameters
    parser.add_argument("--max_steps", type=int, default=50,
                       help="Maximum number of steps per episode")
    parser.add_argument("--scramble_steps", type=int, default=10,
                       help="Number of random moves to scramble the cube")
    
    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=1024,
                       help="Hidden dimension of the policy network")
    
    # PPO parameters
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate for the optimizer")
    parser.add_argument("--clip_ratio", type=float, default=0.2,
                       help="PPO clip ratio")
    parser.add_argument("--num_epochs", type=int, default=4,
                       help="Number of epochs to train on each batch of data")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    
    # Directories
    parser.add_argument("--log_dir", type=str, default="logs",
                       help="Directory to save TensorBoard logs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Directory to save model checkpoints")
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to train on")
    
    return parser.parse_args()


def main():
    """Run the training script."""
    # Parse arguments
    args = parse_args()
    
    # Print training configuration
    print("Training configuration:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create environment
    env = RubiksCubeEnv(max_steps=args.max_steps, scramble_steps=args.scramble_steps)
    
    # Create policy model
    state_dim = 6 * 3 * 3 * 6  # 6 faces × 3×3 positions × 6 colors
    policy = CubePolicy(state_dim=state_dim, hidden_dim=args.hidden_dim)
    
    # Create trainer
    trainer = PPOTrainer(
        policy=policy,
        env=env,
        device=args.device,
        learning_rate=args.learning_rate,
        clip_ratio=args.clip_ratio,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gamma=args.gamma,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Resume training if requested
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
        print(f"Loaded checkpoint at episode {trainer.episodes}")
    
    # Train the policy
    print("\nStarting training...")
    start_time = time.time()
    
    train_stats = trainer.train(
        num_iterations=args.num_iterations,
        steps_per_iteration=args.steps_per_iteration,
        eval_interval=args.eval_interval
    )
    
    # Print training summary
    training_time = time.time() - start_time
    print("\nTraining completed!")
    print(f"Total episodes: {train_stats['episodes']}")
    print(f"Total steps: {train_stats['steps']}")
    print(f"Best solve rate: {train_stats['best_solve_rate']:.2f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    # Print checkpoint location
    print(f"\nFinal model saved to: {os.path.join(args.checkpoint_dir, 'model_best.pt')}")


if __name__ == "__main__":
    main()