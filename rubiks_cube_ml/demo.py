"""
Demo script for Rubik's Cube solver.

This script demonstrates the Rubik's Cube solver by:
1. Loading a trained model
2. Creating a scrambled cube
3. Solving the cube with the trained model
4. Visualizing the solution

Usage:
    python -m rubiks_cube_ml.demo --model <path_to_model> --scramble_steps <num_steps>
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from .cube.cube import RubiksCube
from .model.policy import CubePolicy, GreedyPolicy
from .visualization.visualizer import CubeVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Demo for Rubik's Cube solver")
    
    parser.add_argument("--model", type=str, default="checkpoints/model_best.pt",
                       help="Path to the trained model checkpoint")
    parser.add_argument("--scramble_steps", type=int, default=10,
                       help="Number of random moves to scramble the cube")
    parser.add_argument("--max_solve_steps", type=int, default=50,
                       help="Maximum number of steps to try solving")
    parser.add_argument("--save_vis", type=str, default=None,
                       help="Path to save the solution visualization")
    parser.add_argument("--save_animation", type=str, default=None,
                       help="Path to save the solution animation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run the model on")
    
    return parser.parse_args()


def main():
    """Run the demo."""
    # Parse arguments
    args = parse_args()
    
    # Create the policy model
    print(f"Loading model from {args.model}")
    state_dim = 6 * 3 * 3 * 6  # 6 faces × 3×3 positions × 6 colors
    policy = CubePolicy(state_dim=state_dim)
    
    # Load the model if it exists
    if os.path.exists(args.model):
        checkpoint = torch.load(args.model, map_location=args.device)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        print("Model loaded successfully!")
    else:
        print(f"Warning: Model {args.model} not found. Using untrained model.")
    
    policy.to(args.device)
    policy.eval()
    
    # Create a greedy policy for evaluation
    greedy_policy = GreedyPolicy(policy, args.device)
    
    # Create a cube and scramble it
    print(f"Creating a cube and scrambling with {args.scramble_steps} moves...")
    cube = RubiksCube()
    cube.scramble(args.scramble_steps)
    
    # Print the scrambled cube
    print("\nScrambled Cube:")
    print(cube)
    
    # Create a deep copy for solving
    cube_to_solve = cube.copy()
    
    # Solve the cube
    print(f"\nSolving the cube (max {args.max_solve_steps} steps)...")
    start_time = time.time()
    solution = greedy_policy.solve(cube_to_solve, max_steps=args.max_solve_steps)
    solve_time = time.time() - start_time
    
    # Check if solved
    is_solved = cube_to_solve.is_solved()
    
    # Print solution details
    print("\nSolution:")
    print(f"Steps: {len(solution)}")
    print(f"Time: {solve_time:.2f} seconds")
    print(f"Solved: {is_solved}")
    
    # Print the solution moves
    if solution:
        print("\nMoves applied:")
        for i, move in enumerate(solution):
            print(f"{i+1}: {move}")
    
    # Create visualizer
    visualizer = CubeVisualizer()
    
    # Visualize the solution path
    print("\nVisualizing the solution...")
    if args.save_vis:
        visualizer.plot_solution_path(cube, solution, save_path=args.save_vis)
        print(f"Saved solution visualization to {args.save_vis}")
    else:
        visualizer.plot_solution_path(cube, solution)
    
    # Create an animation if requested
    if args.save_animation:
        print("\nCreating animation...")
        visualizer.animate_moves(cube, solution, interval=500, save_path=args.save_animation)
        print(f"Saved animation to {args.save_animation}")
    
    # Show the visualizations
    plt.show()


if __name__ == "__main__":
    main()