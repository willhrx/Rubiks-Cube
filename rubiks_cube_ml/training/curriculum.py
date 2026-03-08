"""Curriculum learning manager for Rubik's Cube training."""

from __future__ import annotations
from typing import List, Optional
from collections import deque


class CurriculumManager:
    """
    Manages curriculum progression based on success rate.

    Starts with easy scrambles (few moves) and increases difficulty
    as the agent achieves high solve rates.
    """

    def __init__(self,
                 initial_difficulty: int = 1,
                 max_difficulty: int = 20,
                 success_threshold: float = 0.8,
                 window_size: int = 100,
                 increase_step: int = 1):
        """
        Initialize the curriculum manager.

        Args:
            initial_difficulty: Starting scramble depth (number of moves)
            max_difficulty: Maximum scramble depth
            success_threshold: Solve rate required to increase difficulty
            window_size: Number of episodes to track for success rate
            increase_step: How much to increase difficulty each time
        """
        self.initial_difficulty = initial_difficulty
        self.current_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.success_threshold = success_threshold
        self.window_size = window_size
        self.increase_step = increase_step

        # Track success history using deque for efficiency
        self.success_history: deque = deque(maxlen=window_size)

        # Track total episodes and difficulty changes
        self.total_episodes = 0
        self.difficulty_history: List[tuple] = [(0, initial_difficulty)]

    def record_episode(self, solved: bool) -> None:
        """
        Record the outcome of an episode.

        Args:
            solved: Whether the cube was solved in this episode
        """
        self.success_history.append(1 if solved else 0)
        self.total_episodes += 1

        # Check if we should increase difficulty
        if self.should_increase_difficulty():
            self.increase_difficulty()

    def get_success_rate(self) -> float:
        """Get the current success rate over the window."""
        if len(self.success_history) == 0:
            return 0.0
        return sum(self.success_history) / len(self.success_history)

    def should_increase_difficulty(self) -> bool:
        """Check if difficulty should be increased."""
        if self.current_difficulty >= self.max_difficulty:
            return False
        if len(self.success_history) < self.window_size:
            return False
        return self.get_success_rate() >= self.success_threshold

    def increase_difficulty(self) -> None:
        """Increase the difficulty level."""
        if self.current_difficulty < self.max_difficulty:
            self.current_difficulty = min(
                self.current_difficulty + self.increase_step,
                self.max_difficulty
            )
            # Clear history when difficulty changes
            self.success_history.clear()
            # Record the difficulty change
            self.difficulty_history.append((self.total_episodes, self.current_difficulty))

    def reset(self) -> None:
        """Reset curriculum to initial state."""
        self.current_difficulty = self.initial_difficulty
        self.success_history.clear()
        self.total_episodes = 0
        self.difficulty_history = [(0, self.initial_difficulty)]

    def get_scramble_depth(self) -> int:
        """Get the current scramble depth for training."""
        return self.current_difficulty

    def get_stats(self) -> dict:
        """Get curriculum statistics."""
        return {
            'current_difficulty': self.current_difficulty,
            'success_rate': self.get_success_rate(),
            'total_episodes': self.total_episodes,
            'window_size': len(self.success_history),
            'difficulty_history': self.difficulty_history.copy()
        }

    def __repr__(self) -> str:
        return (f"CurriculumManager(difficulty={self.current_difficulty}, "
                f"success_rate={self.get_success_rate():.2%})")
