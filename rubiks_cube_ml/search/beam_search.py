"""Beam search solver for Rubik's Cube."""

from __future__ import annotations
import torch
import numpy as np
from typing import List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
import heapq

if TYPE_CHECKING:
    from ..model.policy import CubePolicy, ImprovedCubePolicy

from ..cube.cube import RubiksCube
from ..cube.moves import MOVES, Move, MoveType


@dataclass
class BeamNode:
    """Node in the beam search tree."""
    cube: RubiksCube
    moves: List[str]
    value: float
    depth: int

    def __lt__(self, other: 'BeamNode') -> bool:
        """Compare nodes by value (higher is better)."""
        return self.value > other.value  # Reversed for max-heap behavior with heapq


class BeamSearch:
    """
    Beam search solver using learned value function.

    Maintains a beam of the most promising states and expands them
    using the policy network's value estimates.
    """

    def __init__(self,
                 policy: 'CubePolicy',
                 beam_width: int = 1000,
                 max_depth: int = 30,
                 device: str = 'cpu'):
        """
        Initialize beam search.

        Args:
            policy: Trained policy network
            beam_width: Number of states to keep at each level
            max_depth: Maximum search depth
            device: Device for neural network inference
        """
        self.policy = policy
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.device = device

        self.policy.to(device)
        self.policy.eval()

        # Build action to move mapping (12 moves, no double moves)
        self._build_action_mapping()

    def _build_action_mapping(self) -> None:
        """Build mapping from action indices to moves."""
        self.action_to_move: List[Move] = []
        self.move_to_string: dict = {}

        for name, move in MOVES.items():
            if move.move_type != MoveType.DOUBLE:
                self.action_to_move.append(move)
                self.move_to_string[id(move)] = name
            if len(self.action_to_move) >= 12:
                break

    def _evaluate_states(self, cubes: List[RubiksCube]) -> np.ndarray:
        """Evaluate a batch of cube states using the value network."""
        states = np.array([cube.get_state_representation() for cube in cubes])
        state_tensor = torch.FloatTensor(states).to(self.device)

        with torch.no_grad():
            _, values = self.policy(state_tensor)

        return values.cpu().numpy().flatten()

    def _get_move_string(self, move: Move) -> str:
        """Get string representation of a move."""
        if move.move_type == MoveType.CLOCKWISE:
            return move.name
        elif move.move_type == MoveType.COUNTERCLOCKWISE:
            return f"{move.name}'"
        else:
            return f"{move.name}2"

    def search(self, cube: RubiksCube) -> Optional[List[str]]:
        """
        Search for a solution using beam search.

        Args:
            cube: Scrambled cube to solve

        Returns:
            List of move strings if solution found, None otherwise
        """
        # Check if already solved
        if cube.is_solved():
            return []

        # Initialize beam with starting state
        initial_value = self._evaluate_states([cube])[0]
        beam = [BeamNode(cube.copy(), [], initial_value, 0)]

        for depth in range(self.max_depth):
            # Expand all nodes in beam
            candidates: List[BeamNode] = []

            for node in beam:
                # Skip if already solved
                if node.cube.is_solved():
                    return node.moves

                # Try all 12 moves
                child_cubes = []
                child_moves_list = []

                for action_idx, move in enumerate(self.action_to_move):
                    child_cube = node.cube.copy()
                    child_cube.apply_move(move)

                    # Check if solved
                    if child_cube.is_solved():
                        return node.moves + [self._get_move_string(move)]

                    child_cubes.append(child_cube)
                    child_moves_list.append(node.moves + [self._get_move_string(move)])

                # Batch evaluate children
                if child_cubes:
                    values = self._evaluate_states(child_cubes)

                    for child_cube, child_moves, value in zip(child_cubes, child_moves_list, values):
                        candidates.append(BeamNode(
                            cube=child_cube,
                            moves=child_moves,
                            value=float(value),
                            depth=depth + 1
                        ))

            # Keep top beam_width candidates
            candidates.sort(key=lambda n: n.value, reverse=True)
            beam = candidates[:self.beam_width]

            # Early termination if beam is empty
            if not beam:
                break

        # No solution found within max_depth
        return None

    def search_batch(self,
                     cubes: List[RubiksCube],
                     return_stats: bool = False) -> List[Optional[List[str]]]:
        """
        Search for solutions for multiple cubes.

        Args:
            cubes: List of scrambled cubes
            return_stats: Whether to return search statistics

        Returns:
            List of solutions (or None for unsolved cubes)
        """
        return [self.search(cube) for cube in cubes]


class BatchBeamSearch:
    """
    Optimized beam search that processes multiple cubes in parallel.

    More efficient than individual searches when solving many cubes.
    """

    def __init__(self,
                 policy: 'CubePolicy',
                 beam_width: int = 100,
                 max_depth: int = 30,
                 batch_size: int = 1024,
                 device: str = 'cpu'):
        """
        Initialize batch beam search.

        Args:
            policy: Trained policy network
            beam_width: Beam width per cube
            max_depth: Maximum search depth
            batch_size: Batch size for neural network
            device: Device for inference
        """
        self.policy = policy
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.batch_size = batch_size
        self.device = device

        self.policy.to(device)
        self.policy.eval()

        # Build action mapping
        self.action_to_move: List[Move] = []
        for move in MOVES.values():
            if move.move_type != MoveType.DOUBLE:
                self.action_to_move.append(move)
            if len(self.action_to_move) >= 12:
                break

    def _evaluate_batch(self, states: np.ndarray) -> np.ndarray:
        """Evaluate a batch of states."""
        # Process in chunks if batch is too large
        all_values = []

        for i in range(0, len(states), self.batch_size):
            chunk = states[i:i + self.batch_size]
            state_tensor = torch.FloatTensor(chunk).to(self.device)

            with torch.no_grad():
                _, values = self.policy(state_tensor)

            all_values.append(values.cpu().numpy())

        return np.concatenate(all_values).flatten()

    def search(self, cube: RubiksCube) -> Optional[List[str]]:
        """Search for a single cube (delegates to simple BeamSearch)."""
        simple_search = BeamSearch(
            self.policy,
            beam_width=self.beam_width,
            max_depth=self.max_depth,
            device=self.device
        )
        return simple_search.search(cube)
