"""Monte Carlo Tree Search solver for Rubik's Cube."""

from __future__ import annotations
import torch
import numpy as np
from typing import List, Optional, Dict, TYPE_CHECKING
from dataclasses import dataclass, field
import math

if TYPE_CHECKING:
    from ..model.policy import CubePolicy, ImprovedCubePolicy

from ..cube.cube import RubiksCube
from ..cube.moves import MOVES, Move, MoveType


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    cube: RubiksCube
    parent: Optional['MCTSNode'] = None
    action: Optional[int] = None  # Action that led to this node
    children: Dict[int, 'MCTSNode'] = field(default_factory=dict)

    # MCTS statistics
    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 0.0

    # Cached properties
    is_solved: bool = field(init=False)
    is_expanded: bool = False

    def __post_init__(self):
        self.is_solved = self.cube.is_solved()

    @property
    def q_value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, c_puct: float = 1.0) -> float:
        """
        Upper Confidence Bound score for selection.

        UCB = Q + c_puct * P * sqrt(N_parent) / (1 + N)

        Args:
            c_puct: Exploration constant

        Returns:
            UCB score
        """
        if self.parent is None:
            return 0.0

        exploration = (c_puct * self.prior *
                      math.sqrt(self.parent.visit_count) /
                      (1 + self.visit_count))

        return self.q_value + exploration

    def best_child(self, c_puct: float = 1.0) -> Optional['MCTSNode']:
        """Get the child with highest UCB score."""
        if not self.children:
            return None

        return max(self.children.values(), key=lambda n: n.ucb_score(c_puct))

    def most_visited_child(self) -> Optional['MCTSNode']:
        """Get the child with most visits."""
        if not self.children:
            return None

        return max(self.children.values(), key=lambda n: n.visit_count)


class MCTS:
    """
    Monte Carlo Tree Search for Rubik's Cube.

    Uses the learned policy for action priors and value network
    for state evaluation.
    """

    def __init__(self,
                 policy: 'CubePolicy',
                 num_simulations: int = 100,
                 c_puct: float = 1.0,
                 max_depth: int = 50,
                 device: str = 'cpu'):
        """
        Initialize MCTS.

        Args:
            policy: Trained policy network
            num_simulations: Number of MCTS simulations per search
            c_puct: Exploration constant for UCB
            max_depth: Maximum depth to search
            device: Device for neural network inference
        """
        self.policy = policy
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.device = device

        self.policy.to(device)
        self.policy.eval()

        # Build action mapping
        self._build_action_mapping()

    def _build_action_mapping(self) -> None:
        """Build mapping from action indices to moves."""
        self.action_to_move: List[Move] = []

        for move in MOVES.values():
            if move.move_type != MoveType.DOUBLE:
                self.action_to_move.append(move)
            if len(self.action_to_move) >= 12:
                break

    def _get_move_string(self, move: Move) -> str:
        """Get string representation of a move."""
        if move.move_type == MoveType.CLOCKWISE:
            return move.name
        elif move.move_type == MoveType.COUNTERCLOCKWISE:
            return f"{move.name}'"
        else:
            return f"{move.name}2"

    def _evaluate(self, cube: RubiksCube) -> tuple:
        """
        Evaluate a cube state using the policy network.

        Returns:
            Tuple of (action_priors, value)
        """
        state = cube.get_state_representation()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs, value = self.policy(state_tensor)

        return action_probs.cpu().numpy()[0], value.item()

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select a leaf node using UCB.

        Traverses the tree from root to leaf, always choosing
        the child with highest UCB score.
        """
        current = node

        while current.is_expanded and current.children and not current.is_solved:
            current = current.best_child(self.c_puct)

        return current

    def _expand(self, node: MCTSNode) -> None:
        """
        Expand a node by creating all children.

        Uses the policy network to set action priors.
        """
        if node.is_solved or node.is_expanded:
            return

        # Get priors from policy network
        priors, _ = self._evaluate(node.cube)

        # Create children for all actions
        for action_idx in range(12):
            child_cube = node.cube.copy()
            child_cube.apply_move(self.action_to_move[action_idx])

            child = MCTSNode(
                cube=child_cube,
                parent=node,
                action=action_idx,
                prior=priors[action_idx]
            )

            node.children[action_idx] = child

        node.is_expanded = True

    def _evaluate_leaf(self, node: MCTSNode) -> float:
        """
        Evaluate a leaf node.

        Returns the value estimate for this node.
        """
        if node.is_solved:
            return 1.0  # Maximum value for solved state

        _, value = self._evaluate(node.cube)
        return value

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """
        Backpropagate the value up the tree.
        """
        current = node

        while current is not None:
            current.visit_count += 1
            current.total_value += value
            current = current.parent

    def search(self, cube: RubiksCube) -> Optional[List[str]]:
        """
        Search for a solution using MCTS.

        Args:
            cube: Scrambled cube to solve

        Returns:
            List of move strings if solution found, None otherwise
        """
        # Check if already solved
        if cube.is_solved():
            return []

        # Create root node
        root = MCTSNode(cube=cube.copy())

        # Run simulations
        for _ in range(self.num_simulations):
            # Selection
            leaf = self._select(root)

            # Check if we found a solution
            if leaf.is_solved:
                return self._extract_path(leaf)

            # Expansion
            self._expand(leaf)

            # Evaluation
            value = self._evaluate_leaf(leaf)

            # Backpropagation
            self._backpropagate(leaf, value)

        # Extract best path after simulations
        return self._extract_best_path(root)

    def _extract_path(self, node: MCTSNode) -> List[str]:
        """Extract the path from root to this node."""
        path = []
        current = node

        while current.parent is not None:
            move = self.action_to_move[current.action]
            path.append(self._get_move_string(move))
            current = current.parent

        return list(reversed(path))

    def _extract_best_path(self, root: MCTSNode) -> Optional[List[str]]:
        """
        Extract the best path from root.

        Follows the most visited children until we reach a leaf or solved state.
        """
        path = []
        current = root

        for _ in range(self.max_depth):
            if current.is_solved:
                return path

            if not current.children:
                break

            # Get most visited child
            best_child = current.most_visited_child()
            if best_child is None:
                break

            move = self.action_to_move[best_child.action]
            path.append(self._get_move_string(move))
            current = best_child

        # Check if final state is solved
        if current.is_solved:
            return path

        return None  # No solution found


class WeightedAStarMCTS(MCTS):
    """
    MCTS variant that uses A*-style weighted search.

    Combines MCTS exploration with A* heuristic guidance.
    """

    def __init__(self,
                 policy: 'CubePolicy',
                 num_simulations: int = 100,
                 c_puct: float = 1.0,
                 weight: float = 0.5,
                 max_depth: int = 50,
                 device: str = 'cpu'):
        """
        Initialize weighted A* MCTS.

        Args:
            policy: Trained policy network
            num_simulations: Number of simulations
            c_puct: Exploration constant
            weight: Weight for depth penalty (0 = pure MCTS, 1 = depth-weighted)
            max_depth: Maximum search depth
            device: Device for inference
        """
        super().__init__(policy, num_simulations, c_puct, max_depth, device)
        self.weight = weight

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select with depth-weighted UCB.

        Penalizes deeper nodes to prefer shorter solutions.
        """
        current = node
        depth = 0

        while current.is_expanded and current.children and not current.is_solved:
            # Modified UCB with depth penalty
            def weighted_ucb(child):
                base_ucb = child.ucb_score(self.c_puct)
                depth_penalty = self.weight * depth / self.max_depth
                return base_ucb - depth_penalty

            current = max(current.children.values(), key=weighted_ucb)
            depth += 1

        return current
