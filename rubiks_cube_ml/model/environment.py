"""Gymnasium environment for Rubik's Cube."""

from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, List, Any, Optional

from ..cube.cube import RubiksCube
from ..cube.moves import MOVES


class RubiksCubeEnv(gym.Env):
    """
    Reinforcement learning environment for the Rubik's Cube.
    
    This environment follows the OpenAI Gym interface, allowing
    reinforcement learning algorithms to train on solving the Rubik's Cube.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, max_steps: int = 50, scramble_steps: int = 20, render_mode: Optional[str] = None):
        """
        Initialize the environment.

        Args:
            max_steps: Maximum number of steps per episode
            scramble_steps: Number of random moves to scramble the cube
            render_mode: The render mode ('human' or 'rgb_array')
        """
        super(RubiksCubeEnv, self).__init__()
        self.render_mode = render_mode
        
        self.cube = RubiksCube()
        self.max_steps = max_steps
        self.scramble_steps = scramble_steps
        self.step_count = 0
        
        # Define action space (12 moves: R, R', L, L', U, U', D, D', F, F', B, B')
        # Note: Double moves (R2, L2, etc.) are excluded as they can be achieved
        # by applying single moves twice, reducing action space complexity for RL
        self.action_space = spaces.Discrete(12)
        
        # Define observation space
        # 6 faces × 3×3 positions × 6 colors (one-hot)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6 * 3 * 3 * 6,), dtype=np.float32
        )
        
        # Mapping from action indices to moves
        self.action_to_move = list(MOVES.values())[:12]  # Take the first 12 moves
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.

        Resets the cube to a solved state and then scrambles it.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            Tuple of (initial observation, info dict)
        """
        super().reset(seed=seed)

        self.cube = RubiksCube()
        self.step_count = 0

        # Scramble the cube
        self.cube.scramble(self.scramble_steps)

        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: Action to take (index into self.action_to_move)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if action < 0 or action >= len(self.action_to_move):
            raise ValueError(f"Invalid action {action}, must be in range [0, {len(self.action_to_move) - 1}]")

        # Apply the move
        move = self.action_to_move[action]
        self.cube.apply_move(move)

        # Increment step counter
        self.step_count += 1

        # Check if the cube is solved
        solved = self.cube.is_solved()

        # terminated = episode ended due to task completion (solved)
        # truncated = episode ended due to time limit
        terminated = solved
        truncated = not solved and self.step_count >= self.max_steps

        # Calculate reward
        if solved:
            reward = 10.0  # High reward for solving the cube
        else:
            # Negative reward for each step to encourage shorter solutions
            reward = -0.1

        info = {"solved": solved, "steps": self.step_count}

        return self._get_observation(), reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        Uses the render_mode set during initialization.

        Returns:
            If render_mode is 'rgb_array', returns the rendered image as a numpy array
        """
        if self.render_mode is None:
            return None

        if self.render_mode == "human":
            print(self.cube)
            return None
        elif self.render_mode == "rgb_array":
            # Use the visualizer to render the cube
            from ..visualization.visualizer import CubeVisualizer
            visualizer = CubeVisualizer()
            fig = visualizer.visualize(self.cube)

            # Convert the matplotlib figure to a numpy array
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            import matplotlib.pyplot as plt
            plt.close(fig)

            return img
        else:
            raise ValueError(f"Invalid render mode: {self.render_mode}")
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation."""
        return self.cube.get_state_representation()
    
    def get_cube_state(self) -> RubiksCube:
        """Get the current cube state (for external use)."""
        return self.cube