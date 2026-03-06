"""Policy model for solving Rubik's Cube."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, List

from ..cube.cube import RubiksCube
from ..cube.moves import MOVES


class CubePolicy(nn.Module):
    """
    Neural network policy for solving the Rubik's Cube.
    
    This model takes the cube state as input and outputs a policy (action probabilities)
    and value estimate.
    """
    
    def __init__(self, state_dim: int = 6*3*3*6, hidden_dim: int = 1024, num_actions: int = 12):
        """
        Initialize the policy network.
        
        Args:
            state_dim: Dimension of the input state
            hidden_dim: Dimension of the hidden layers
            num_actions: Number of possible actions
        """
        super(CubePolicy, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        # Feature extraction layers
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor of shape (batch_size, state_dim)
            
        Returns:
            Tuple of (action_probs, value)
        """
        features = self.feature_layers(x)
        
        # Policy output (action probabilities)
        policy_logits = self.policy_head(features)
        action_probs = F.softmax(policy_logits, dim=-1)
        
        # Value output (state value estimate)
        value = self.value_head(features)
        
        return action_probs, value
        
    def save(self, path: str):
        """Save the model to a file."""
        torch.save(self.state_dict(), path)
        
    def load(self, path: str, device: str = 'cpu'):
        """Load the model from a file."""
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()


class GreedyPolicy:
    """
    Greedy policy that selects actions based on the trained policy network.
    
    This is used for inference/evaluation after training.
    """
    
    def __init__(self, policy_network: CubePolicy, device: str = 'cpu'):
        """
        Initialize the greedy policy.
        
        Args:
            policy_network: Trained policy network
            device: Device to run the network on
        """
        self.policy_network = policy_network
        self.device = device
        self.policy_network.to(device)
        self.policy_network.eval()
        
    def select_action(self, state: np.ndarray) -> int:
        """
        Select the best action according to the policy.
        
        Args:
            state: Current state representation
            
        Returns:
            Action index
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, _ = self.policy_network(state_tensor)
            
            # Select the action with the highest probability
            action = torch.argmax(action_probs, dim=1).item()
            
        return action
    
    def solve(self, cube: RubiksCube, max_steps: int = 100) -> List[str]:
        """
        Attempt to solve the cube using the policy.
        
        Args:
            cube: Cube to solve
            max_steps: Maximum number of steps to try
            
        Returns:
            List of moves applied
        """
        # Make a copy of the cube to avoid modifying the original
        cube_copy = cube.copy()
        moves_applied = []
        
        # List of action indices for the policy
        action_to_move = list(MOVES.values())[:12]
        
        for _ in range(max_steps):
            if cube_copy.is_solved():
                break
                
            # Get the state representation
            state = cube_copy.get_state_representation()
            
            # Select the action
            action_idx = self.select_action(state)
            move = action_to_move[action_idx]
            
            # Apply the move
            cube_copy.apply_move(move)
            
            # Record the move
            for move_str, move_obj in MOVES.items():
                if move_obj == move:
                    moves_applied.append(move_str)
                    break
        
        return moves_applied