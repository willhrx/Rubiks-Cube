"""
Training script for Rubik's Cube solver.

This script trains a reinforcement learning model to solve the Rubik's Cube.

Usage:
    # PPO training with curriculum learning
    python -m rubiks_cube_ml.train --mode ppo --curriculum

    # Autodidactic iteration (DeepCubeA style) - recommended
    python -m rubiks_cube_ml.train --mode autodidactic --architecture improved

    # Full setup with all enhancements
    python -m rubiks_cube_ml.train --mode autodidactic --architecture improved --curriculum --shaped_rewards
"""

import argparse
import torch
import numpy as np
import os
import time

from .model.policy import CubePolicy, ImprovedCubePolicy
from .model.environment import RubiksCubeEnv
from .training.ppo_trainer import PPOTrainer
from .training.autodidactic_trainer import AutodidacticTrainer
from .training.curriculum import CurriculumManager
from .cube.state_features import RewardShaper


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Training script for Rubik's Cube solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick start with autodidactic training (recommended)
  python -m rubiks_cube_ml.train --mode autodidactic

  # PPO with curriculum learning
  python -m rubiks_cube_ml.train --mode ppo --curriculum --shaped_rewards

  # Full setup with improved architecture
  python -m rubiks_cube_ml.train --mode autodidactic --architecture improved --num_iterations 100000
        """
    )

    # Training mode
    parser.add_argument("--mode", type=str, default="autodidactic",
                        choices=["ppo", "autodidactic"],
                        help="Training mode: 'ppo' or 'autodidactic' (DeepCubeA style)")

    # Architecture
    parser.add_argument("--architecture", type=str, default="improved",
                        choices=["standard", "improved"],
                        help="Network architecture: 'standard' (MLP) or 'improved' (ResNet)")

    # Training parameters
    parser.add_argument("--num_iterations", type=int, default=100000,
                        help="Number of training iterations")
    parser.add_argument("--steps_per_iteration", type=int, default=2048,
                        help="Number of environment steps per iteration (PPO only)")
    parser.add_argument("--eval_interval", type=int, default=1000,
                        help="Evaluation interval (in iterations)")

    # Environment parameters
    parser.add_argument("--max_steps", type=int, default=50,
                        help="Maximum number of steps per episode")
    parser.add_argument("--scramble_steps", type=int, default=20,
                        help="Number of random moves to scramble the cube")
    parser.add_argument("--max_scramble_depth", type=int, default=30,
                        help="Maximum scramble depth for autodidactic training")

    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=4096,
                        help="Hidden dimension of the policy network")
    parser.add_argument("--num_blocks", type=int, default=4,
                        help="Number of residual blocks (improved architecture)")

    # Curriculum learning
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable curriculum learning (PPO mode)")
    parser.add_argument("--initial_difficulty", type=int, default=1,
                        help="Initial scramble depth for curriculum")
    parser.add_argument("--success_threshold", type=float, default=0.8,
                        help="Success rate threshold to increase difficulty")

    # Reward shaping
    parser.add_argument("--shaped_rewards", action="store_true",
                        help="Enable reward shaping (PPO mode)")

    # PPO parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for the optimizer")
    parser.add_argument("--clip_ratio", type=float, default=0.2,
                        help="PPO clip ratio")
    parser.add_argument("--num_epochs", type=int, default=4,
                        help="Number of epochs to train on each batch of data")
    parser.add_argument("--batch_size", type=int, default=1024,
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
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on")

    return parser.parse_args()


def create_policy(args):
    """Create policy network based on arguments."""
    state_dim = 6 * 3 * 3 * 6  # 6 faces × 3×3 positions × 6 colors

    if args.architecture == "improved":
        return ImprovedCubePolicy(
            state_dim=state_dim,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks
        )
    else:
        return CubePolicy(
            state_dim=state_dim,
            hidden_dim=args.hidden_dim
        )


def train_autodidactic(args, policy):
    """Train using Autodidactic Iteration (DeepCubeA style)."""
    print("\n" + "=" * 50)
    print("AUTODIDACTIC ITERATION TRAINING")
    print("=" * 50)

    trainer = AutodidacticTrainer(
        policy=policy,
        device=args.device,
        max_scramble_depth=args.max_scramble_depth,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )

    # Resume if requested
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    history = trainer.train(
        num_iterations=args.num_iterations,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.eval_interval * 10
    )

    return history


def train_ppo(args, policy):
    """Train using PPO with optional curriculum and reward shaping."""
    print("\n" + "=" * 50)
    print("PPO TRAINING")
    print("=" * 50)

    # Setup curriculum manager if enabled
    curriculum_manager = None
    if args.curriculum:
        curriculum_manager = CurriculumManager(
            initial_difficulty=args.initial_difficulty,
            max_difficulty=args.scramble_steps,
            success_threshold=args.success_threshold
        )
        print(f"Curriculum learning enabled: {args.initial_difficulty} -> {args.scramble_steps}")

    # Setup reward shaper if enabled
    reward_shaper = None
    if args.shaped_rewards:
        reward_shaper = RewardShaper(gamma=args.gamma)
        print("Reward shaping enabled")

    # Create environment with enhancements
    env = RubiksCubeEnv(
        max_steps=args.max_steps,
        scramble_steps=args.scramble_steps,
        curriculum_manager=curriculum_manager,
        reward_shaper=reward_shaper,
        use_shaped_rewards=args.shaped_rewards
    )

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

    # Resume if requested
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
        print(f"Loaded checkpoint at episode {trainer.episodes}")

    # Train
    train_stats = trainer.train(
        num_iterations=args.num_iterations,
        steps_per_iteration=args.steps_per_iteration,
        eval_interval=args.eval_interval
    )

    return train_stats


def main():
    """Run the training script."""
    # Parse arguments
    args = parse_args()

    # Print training configuration
    print("\n" + "=" * 50)
    print("RUBIK'S CUBE SOLVER TRAINING")
    print("=" * 50)
    print("\nConfiguration:")
    print(f"  Mode: {args.mode}")
    print(f"  Architecture: {args.architecture}")
    print(f"  Device: {args.device}")
    print(f"  Iterations: {args.num_iterations}")
    print(f"  Hidden dim: {args.hidden_dim}")
    if args.mode == "ppo":
        print(f"  Curriculum: {args.curriculum}")
        print(f"  Shaped rewards: {args.shaped_rewards}")

    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Create policy model
    policy = create_policy(args)
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"\nModel parameters: {num_params:,}")

    # Train
    start_time = time.time()

    if args.mode == "autodidactic":
        train_autodidactic(args, policy)
    else:
        train_ppo(args, policy)

    # Print training summary
    training_time = time.time() - start_time
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)
    print(f"Training time: {training_time / 60:.1f} minutes")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()