"""State features for reward shaping in Rubik's Cube training."""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .cube import RubiksCube


class CubeStateFeatures:
    """
    Extract features from cube state for reward shaping.

    These features provide intermediate signals that help guide
    the learning process beyond sparse solved/unsolved rewards.
    """

    # Corner positions: (face, row, col) for each of the 3 stickers of each corner
    # 8 corners total
    CORNER_POSITIONS = [
        # URF corner (Up-Right-Front)
        [(0, 2, 2), (4, 0, 0), (2, 0, 2)],
        # UFL corner (Up-Front-Left)
        [(0, 2, 0), (2, 0, 0), (5, 0, 2)],
        # ULB corner (Up-Left-Back)
        [(0, 0, 0), (5, 0, 0), (3, 0, 2)],
        # UBR corner (Up-Back-Right)
        [(0, 0, 2), (3, 0, 0), (4, 0, 2)],
        # DFR corner (Down-Front-Right)
        [(1, 0, 2), (2, 2, 2), (4, 2, 0)],
        # DLF corner (Down-Left-Front)
        [(1, 0, 0), (5, 2, 2), (2, 2, 0)],
        # DBL corner (Down-Back-Left)
        [(1, 2, 0), (3, 2, 2), (5, 2, 0)],
        # DRB corner (Down-Right-Back)
        [(1, 2, 2), (4, 2, 2), (3, 2, 0)],
    ]

    # Edge positions: (face, row, col) for each of the 2 stickers of each edge
    # 12 edges total
    EDGE_POSITIONS = [
        # Top layer edges
        [(0, 1, 2), (4, 0, 1)],   # UR
        [(0, 2, 1), (2, 0, 1)],   # UF
        [(0, 1, 0), (5, 0, 1)],   # UL
        [(0, 0, 1), (3, 0, 1)],   # UB
        # Middle layer edges
        [(2, 1, 2), (4, 1, 0)],   # FR
        [(2, 1, 0), (5, 1, 2)],   # FL
        [(3, 1, 0), (4, 1, 2)],   # BR
        [(3, 1, 2), (5, 1, 0)],   # BL
        # Bottom layer edges
        [(1, 1, 2), (4, 2, 1)],   # DR
        [(1, 0, 1), (2, 2, 1)],   # DF
        [(1, 1, 0), (5, 2, 1)],   # DL
        [(1, 2, 1), (3, 2, 1)],   # DB
    ]

    # Solved colors for each face
    SOLVED_COLORS = {
        0: 0,  # UP -> WHITE
        1: 1,  # DOWN -> YELLOW
        2: 2,  # FRONT -> RED
        3: 3,  # BACK -> ORANGE
        4: 4,  # RIGHT -> BLUE
        5: 5,  # LEFT -> GREEN
    }

    @staticmethod
    def count_correct_facelets(cube: 'RubiksCube') -> int:
        """
        Count the number of facelets (stickers) in correct positions.

        Args:
            cube: The cube to analyze

        Returns:
            Number of correct facelets (0-54)
        """
        correct = 0
        for face in range(6):
            expected_color = CubeStateFeatures.SOLVED_COLORS[face]
            correct += np.sum(cube.state[face] == expected_color)
        return int(correct)

    @staticmethod
    def count_correct_centers(cube: 'RubiksCube') -> int:
        """
        Count centers in correct position (always 6 for standard cube).

        Centers don't move, so this is always 6.
        """
        correct = 0
        for face in range(6):
            if cube.state[face, 1, 1] == CubeStateFeatures.SOLVED_COLORS[face]:
                correct += 1
        return correct

    @staticmethod
    def count_correct_edges(cube: 'RubiksCube') -> int:
        """
        Count edge pieces in correct position AND orientation.

        Args:
            cube: The cube to analyze

        Returns:
            Number of correctly placed edges (0-12)
        """
        correct = 0
        for edge_stickers in CubeStateFeatures.EDGE_POSITIONS:
            all_correct = True
            for face, row, col in edge_stickers:
                expected = CubeStateFeatures.SOLVED_COLORS[face]
                if cube.state[face, row, col] != expected:
                    all_correct = False
                    break
            if all_correct:
                correct += 1
        return correct

    @staticmethod
    def count_correct_corners(cube: 'RubiksCube') -> int:
        """
        Count corner pieces in correct position AND orientation.

        Args:
            cube: The cube to analyze

        Returns:
            Number of correctly placed corners (0-8)
        """
        correct = 0
        for corner_stickers in CubeStateFeatures.CORNER_POSITIONS:
            all_correct = True
            for face, row, col in corner_stickers:
                expected = CubeStateFeatures.SOLVED_COLORS[face]
                if cube.state[face, row, col] != expected:
                    all_correct = False
                    break
            if all_correct:
                correct += 1
        return correct

    @staticmethod
    def count_complete_faces(cube: 'RubiksCube') -> int:
        """
        Count faces that are completely solved (all same color).

        Args:
            cube: The cube to analyze

        Returns:
            Number of complete faces (0-6)
        """
        complete = 0
        for face in range(6):
            center_color = cube.state[face, 1, 1]
            if np.all(cube.state[face] == center_color):
                complete += 1
        return complete

    @staticmethod
    def get_all_features(cube: 'RubiksCube') -> Dict[str, int]:
        """
        Get all state features for a cube.

        Args:
            cube: The cube to analyze

        Returns:
            Dictionary of feature name -> value
        """
        return {
            'correct_facelets': CubeStateFeatures.count_correct_facelets(cube),
            'correct_edges': CubeStateFeatures.count_correct_edges(cube),
            'correct_corners': CubeStateFeatures.count_correct_corners(cube),
            'complete_faces': CubeStateFeatures.count_complete_faces(cube),
        }

    @staticmethod
    def compute_potential(cube: 'RubiksCube') -> float:
        """
        Compute a potential value for potential-based reward shaping.

        Higher potential = closer to solved.
        Uses weighted combination of features.

        Args:
            cube: The cube to analyze

        Returns:
            Potential value (normalized to roughly 0-1 range)
        """
        # Get raw counts
        correct_facelets = CubeStateFeatures.count_correct_facelets(cube)
        correct_edges = CubeStateFeatures.count_correct_edges(cube)
        correct_corners = CubeStateFeatures.count_correct_corners(cube)

        # Normalize each component
        # Facelets: 54 total (but 6 centers always correct)
        facelet_score = correct_facelets / 54.0

        # Edges: 12 total, worth more than individual facelets
        edge_score = correct_edges / 12.0

        # Corners: 8 total, worth the most
        corner_score = correct_corners / 8.0

        # Weighted combination (pieces matter more than facelets)
        potential = 0.2 * facelet_score + 0.4 * edge_score + 0.4 * corner_score

        return potential


class RewardShaper:
    """
    Computes shaped rewards using potential-based shaping.

    Potential-based shaping preserves optimal policies while providing
    denser reward signals during training.
    """

    def __init__(self, gamma: float = 0.99, scale: float = 1.0):
        """
        Initialize the reward shaper.

        Args:
            gamma: Discount factor (must match training gamma)
            scale: Scaling factor for shaped rewards
        """
        self.gamma = gamma
        self.scale = scale
        self._prev_potential: float = 1.0  # Start at solved potential

    def reset(self, cube: 'RubiksCube') -> None:
        """Reset for a new episode."""
        self._prev_potential = CubeStateFeatures.compute_potential(cube)

    def compute_shaped_reward(self,
                               cube: 'RubiksCube',
                               base_reward: float,
                               done: bool) -> float:
        """
        Compute shaped reward using potential-based shaping.

        F(s, s') = gamma * Phi(s') - Phi(s)

        This preserves optimal policies while providing denser signals.

        Args:
            cube: Current cube state
            base_reward: Original environment reward
            done: Whether episode is done

        Returns:
            Shaped reward
        """
        current_potential = CubeStateFeatures.compute_potential(cube)

        # Potential-based shaping: F = gamma * Phi(s') - Phi(s)
        shaping_reward = self.gamma * current_potential - self._prev_potential

        # Update previous potential
        self._prev_potential = current_potential

        # Combined reward
        return base_reward + self.scale * shaping_reward

    def get_features(self, cube: 'RubiksCube') -> Dict[str, int]:
        """Get current state features."""
        return CubeStateFeatures.get_all_features(cube)
