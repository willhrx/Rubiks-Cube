"""Evaluation for Rubik's Cube solver."""

from __future__ import annotations
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import time
import os

from ..model.policy import CubePolicy, GreedyPolicy
from ..model.environment import RubiksCubeEnv
from ..cube.cube import RubiksCube
from ..visualization.visualizer import CubeVisualizer


class CubeEvaluator:
    """
    Evaluator for Rubik's Cube solvers.
    
    This class provides methods to evaluate the performance of a
    trained Rubik's Cube solver policy.
    """
    
    def __init__(self, 
                 policy: CubePolicy,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the evaluator.
        
        Args:
            policy: Trained policy model
            device: Device to run the model on
        """
        self.policy = policy
        self.device = device
        self.policy.to(device)
        self.policy.eval()
        
        # Create a greedy policy for evaluation
        self.greedy_policy = GreedyPolicy(policy, device)
        
    def evaluate_solve_rate(self, 
                          num_episodes: int = 100, 
                          scramble_steps: int = 20,
                          max_solve_steps: int = 50) -> Dict[str, Any]:
        """
        Evaluate the solve rate of the policy.
        
        Args:
            num_episodes: Number of episodes to evaluate on
            scramble_steps: Number of random moves to scramble the cube
            max_solve_steps: Maximum number of steps to try solving
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Create environment with specified scramble steps
        env = RubiksCubeEnv(max_steps=max_solve_steps, scramble_steps=scramble_steps)
        
        solved_count = 0
        steps_to_solve = []
        total_steps = 0
        
        for episode in range(num_episodes):
            # Reset the environment
            obs = env.reset()
            cube = env.get_cube_state()
            
            # Try to solve
            solution = self.greedy_policy.solve(cube, max_steps=max_solve_steps)
            
            # Check if solved
            is_solved = cube.is_solved()
            
            if is_solved:
                solved_count += 1
                steps_to_solve.append(len(solution))
            
            total_steps += len(solution)
            
            # Print progress every 10 episodes
            if (episode + 1) % 10 == 0:
                print(f"Evaluated {episode+1}/{num_episodes} episodes, "
                      f"Solve rate: {solved_count/(episode+1):.2f}")
        
        # Calculate metrics
        solve_rate = solved_count / num_episodes
        avg_steps_overall = total_steps / num_episodes
        
        if steps_to_solve:
            avg_steps_when_solved = sum(steps_to_solve) / len(steps_to_solve)
        else:
            avg_steps_when_solved = float('inf')
        
        return {
            'solve_rate': solve_rate,
            'avg_steps_overall': avg_steps_overall,
            'avg_steps_when_solved': avg_steps_when_solved,
            'num_solved': solved_count,
            'num_episodes': num_episodes,
            'scramble_steps': scramble_steps
        }
    
    def evaluate_by_difficulty(self, 
                             max_difficulty: int = 25, 
                             episodes_per_level: int = 20,
                             max_solve_steps: int = 100) -> Dict[int, Dict[str, Any]]:
        """
        Evaluate the policy at different scramble difficulty levels.
        
        Args:
            max_difficulty: Maximum number of scramble steps
            episodes_per_level: Number of episodes per difficulty level
            max_solve_steps: Maximum steps to try solving
            
        Returns:
            Dictionary mapping difficulty level to metrics
        """
        results = {}
        
        # Evaluate at each difficulty level
        for difficulty in range(1, max_difficulty + 1, 2):
            print(f"\nEvaluating at difficulty level {difficulty} scramble steps...")
            
            # Create environment with current difficulty
            env = RubiksCubeEnv(max_steps=max_solve_steps, scramble_steps=difficulty)
            
            solved_count = 0
            steps_to_solve = []
            
            for episode in range(episodes_per_level):
                # Reset the environment
                obs = env.reset()
                cube = env.get_cube_state()
                
                # Try to solve
                solution = self.greedy_policy.solve(cube, max_steps=max_solve_steps)
                
                # Check if solved
                is_solved = cube.is_solved()
                
                if is_solved:
                    solved_count += 1
                    steps_to_solve.append(len(solution))
            
            # Calculate metrics
            solve_rate = solved_count / episodes_per_level
            
            if steps_to_solve:
                avg_steps = sum(steps_to_solve) / len(steps_to_solve)
            else:
                avg_steps = float('inf')
            
            results[difficulty] = {
                'solve_rate': solve_rate,
                'avg_steps': avg_steps,
                'num_solved': solved_count,
                'num_episodes': episodes_per_level
            }
            
            print(f"Difficulty {difficulty}: Solve rate = {solve_rate:.2f}, "
                  f"Avg steps = {avg_steps if avg_steps != float('inf') else 'N/A'}")
        
        return results
    
    def plot_difficulty_results(self, 
                              results: Dict[int, Dict[str, Any]], 
                              save_path: Optional[str] = None):
        """
        Plot the results of the difficulty evaluation.
        
        Args:
            results: Results from evaluate_by_difficulty
            save_path: If provided, save the plot to this path
        
        Returns:
            The created figure
        """
        # Extract data
        difficulties = list(results.keys())
        solve_rates = [results[d]['solve_rate'] for d in difficulties]
        
        avg_steps = []
        for d in difficulties:
            if results[d]['avg_steps'] == float('inf'):
                avg_steps.append(np.nan)
            else:
                avg_steps.append(results[d]['avg_steps'])
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot solve rates
        sns.lineplot(x=difficulties, y=solve_rates, ax=ax1, marker='o')
        ax1.set_title('Solve Rate vs. Scramble Difficulty')
        ax1.set_xlabel('Scramble Steps')
        ax1.set_ylabel('Solve Rate')
        ax1.set_ylim(0, 1.05)
        ax1.grid(True)
        
        # Plot average steps
        sns.lineplot(x=difficulties, y=avg_steps, ax=ax2, marker='o')
        ax2.set_title('Average Solution Steps vs. Scramble Difficulty')
        ax2.set_xlabel('Scramble Steps')
        ax2.set_ylabel('Average Steps to Solve')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def demonstrate_solution(self, 
                           scramble_steps: int = 10,
                           max_solve_steps: int = 50,
                           save_path: Optional[str] = None) -> Tuple[RubiksCube, List[str]]:
        """
        Demonstrate a solution found by the policy.
        
        Args:
            scramble_steps: Number of random moves to scramble the cube
            max_solve_steps: Maximum number of steps to try solving
            save_path: If provided, save a visualization of the solution
            
        Returns:
            Tuple of (final cube state, solution moves)
        """
        # Create a fresh cube
        cube = RubiksCube()
        
        # Scramble the cube
        print(f"Scrambling the cube with {scramble_steps} random moves...")
        cube.scramble(scramble_steps)
        
        # Try to solve
        print("Attempting to solve...")
        start_time = time.time()
        solution = self.greedy_policy.solve(cube.copy(), max_steps=max_solve_steps)
        solve_time = time.time() - start_time
        
        # Check if solved
        is_solved = cube.is_solved()
        
        if is_solved:
            print(f"Solved in {len(solution)} steps! Time: {solve_time:.2f}s")
        else:
            print(f"Failed to solve after {len(solution)} steps. Time: {solve_time:.2f}s")
        
        # Visualize the solution
        visualizer = CubeVisualizer()
        
        if save_path:
            print(f"Saving solution visualization to {save_path}")
            visualizer.plot_solution_path(cube, solution, save_path=save_path)
        else:
            visualizer.plot_solution_path(cube, solution)
            plt.show()
        
        return cube, solution