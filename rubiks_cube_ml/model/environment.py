"""Gymnasium environment for Rubik's Cube."""

from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, List, Any, Optional, TYPE_CHECKING

from ..cube.cube import RubiksCube
from ..cube.moves import MOVES

if TYPE_CHECKING:
    from ..training.curriculum import CurriculumManager
    from ..cube.state_features import RewardShaper


class RubiksCubeEnv(gym.Env):
    """
    Reinforcement learning environment for the Rubik's Cube.

    This environment follows the OpenAI Gym interface, allowing
    reinforcement learning algorithms to train on solving the Rubik's Cube.

    Supports optional curriculum learning and reward shaping.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self,
                 max_steps: int = 50,
                 scramble_steps: int = 20,
                 render_mode: Optional[str] = None,
                 curriculum_manager: Optional['CurriculumManager'] = None,
                 reward_shaper: Optional['RewardShaper'] = None,
                 use_shaped_rewards: bool = False):
        """
        Initialize the environment.

        Args:
            max_steps: Maximum number of steps per episode
            scramble_steps: Number of random moves to scramble the cube
                           (ignored if curriculum_manager is provided)
            render_mode: The render mode ('human' or 'rgb_array')
            curriculum_manager: Optional curriculum manager for adaptive difficulty
            reward_shaper: Optional reward shaper for dense rewards
            use_shaped_rewards: Whether to use shaped rewards
        """
        super(RubiksCubeEnv, self).__init__()
        self.render_mode = render_mode

        self.cube = RubiksCube()
        self.max_steps = max_steps
        self.scramble_steps = scramble_steps
        self.step_count = 0

        # Curriculum learning
        self.curriculum_manager = curriculum_manager

        # Reward shaping
        self.reward_shaper = reward_shaper
        self.use_shaped_rewards = use_shaped_rewards
        
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
            options: Additional options:
                - scramble_steps: Override the number of scramble steps

        Returns:
            Tuple of (initial observation, info dict)
        """
        super().reset(seed=seed)

        self.cube = RubiksCube()
        self.step_count = 0

        # Determine scramble depth
        if options and 'scramble_steps' in options:
            scramble_depth = options['scramble_steps']
        elif self.curriculum_manager is not None:
            scramble_depth = self.curriculum_manager.get_scramble_depth()
        else:
            scramble_depth = self.scramble_steps

        # Scramble the cube
        self.cube.scramble(scramble_depth)

        # Initialize reward shaper if present
        if self.reward_shaper is not None:
            self.reward_shaper.reset(self.cube)

        info = {'scramble_depth': scramble_depth}
        return self._get_observation(), info
    
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

        # Calculate base reward
        if solved:
            base_reward = 10.0  # High reward for solving the cube
        else:
            # Negative reward for each step to encourage shorter solutions
            base_reward = -0.1

        # Apply reward shaping if enabled
        if self.use_shaped_rewards and self.reward_shaper is not None:
            reward = self.reward_shaper.compute_shaped_reward(
                self.cube, base_reward, terminated or truncated
            )
        else:
            reward = base_reward

        # Record episode result for curriculum
        if (terminated or truncated) and self.curriculum_manager is not None:
            self.curriculum_manager.record_episode(solved)

        info = {
            "solved": solved,
            "steps": self.step_count,
            "base_reward": base_reward
        }

        # Add state features to info if reward shaper is available
        if self.reward_shaper is not None:
            info["features"] = self.reward_shaper.get_features(self.cube)

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