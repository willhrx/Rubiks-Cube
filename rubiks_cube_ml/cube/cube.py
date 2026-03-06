"""Rubik's Cube representation."""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import numpy as np
from enum import Enum
import random

from .moves import Move, MOVES


class Color(Enum):
    """Colors of the Rubik's Cube faces."""

    WHITE = 0
    YELLOW = 1
    RED = 2
    ORANGE = 3
    BLUE = 4
    GREEN = 5


class RubiksCube:
    """
    Representation of a Rubik's Cube.
    
    This class implements a 3x3x3 Rubik's Cube using a 3D representation.
    The cube is represented as a 6x3x3 numpy array, where the first dimension
    represents the face (0=UP, 1=DOWN, 2=FRONT, 3=BACK, 4=RIGHT, 5=LEFT).
    """

    # Face indices
    UP = 0
    DOWN = 1
    FRONT = 2
    BACK = 3
    RIGHT = 4
    LEFT = 5

    # Default colors for solved state
    FACE_COLORS = {
        UP: Color.WHITE,
        DOWN: Color.YELLOW,
        FRONT: Color.RED,
        BACK: Color.ORANGE,
        RIGHT: Color.BLUE,
        LEFT: Color.GREEN,
    }

    def __init__(self):
        """Initialize a solved Rubik's Cube."""
        # 6 faces, each 3x3
        self.state = np.zeros((6, 3, 3), dtype=np.int8)
        
        # Initialize solved state
        for face, color in self.FACE_COLORS.items():
            self.state[face, :, :] = color.value
            
        self.move_history: List[Move] = []

    def copy(self) -> RubiksCube:
        """Create a deep copy of the cube."""
        new_cube = RubiksCube()
        new_cube.state = self.state.copy()
        new_cube.move_history = self.move_history.copy()
        return new_cube

    def is_solved(self) -> bool:
        """Check if the cube is solved."""
        for face in range(6):
            # Each face should have the same color for all 9 pieces
            color = self.state[face, 0, 0]
            if not np.all(self.state[face, :, :] == color):
                return False
        return True

    def apply_move(self, move: Move) -> RubiksCube:
        """Apply a move to the cube."""
        from .moves import MoveType

        # Record the move
        self.move_history.append(move)

        # Get the base move method (e.g., _move_r for R, R', R2)
        base_method = getattr(self, f"_move_{move.name.lower()}")

        if move.move_type == MoveType.CLOCKWISE:
            base_method()
        elif move.move_type == MoveType.COUNTERCLOCKWISE:
            # Apply 3 times (equivalent to prime/inverse)
            base_method()
            base_method()
            base_method()
        elif move.move_type == MoveType.DOUBLE:
            # Apply twice
            base_method()
            base_method()

        return self

    def _move_r(self):
        """Rotate the right face clockwise."""
        # Rotate right face
        self.state[self.RIGHT] = np.rot90(self.state[self.RIGHT], k=3)
        
        # Update adjacent faces
        temp = self.state[self.UP, :, 2].copy()
        self.state[self.UP, :, 2] = self.state[self.BACK, ::-1, 0].copy()
        self.state[self.BACK, ::-1, 0] = self.state[self.DOWN, :, 2].copy()
        self.state[self.DOWN, :, 2] = self.state[self.FRONT, :, 2].copy()
        self.state[self.FRONT, :, 2] = temp

    def _move_l(self):
        """Rotate the left face clockwise."""
        # Rotate left face
        self.state[self.LEFT] = np.rot90(self.state[self.LEFT], k=3)
        
        # Update adjacent faces
        temp = self.state[self.UP, :, 0].copy()
        self.state[self.UP, :, 0] = self.state[self.FRONT, :, 0].copy()
        self.state[self.FRONT, :, 0] = self.state[self.DOWN, :, 0].copy()
        self.state[self.DOWN, :, 0] = self.state[self.BACK, ::-1, 2].copy()
        self.state[self.BACK, ::-1, 2] = temp

    def _move_u(self):
        """Rotate the up face clockwise."""
        # Rotate up face
        self.state[self.UP] = np.rot90(self.state[self.UP], k=3)
        
        # Update adjacent faces
        temp = self.state[self.FRONT, 0, :].copy()
        self.state[self.FRONT, 0, :] = self.state[self.RIGHT, 0, :].copy()
        self.state[self.RIGHT, 0, :] = self.state[self.BACK, 0, :].copy()
        self.state[self.BACK, 0, :] = self.state[self.LEFT, 0, :].copy()
        self.state[self.LEFT, 0, :] = temp

    def _move_d(self):
        """Rotate the down face clockwise."""
        # Rotate down face
        self.state[self.DOWN] = np.rot90(self.state[self.DOWN], k=3)
        
        # Update adjacent faces
        temp = self.state[self.FRONT, 2, :].copy()
        self.state[self.FRONT, 2, :] = self.state[self.LEFT, 2, :].copy()
        self.state[self.LEFT, 2, :] = self.state[self.BACK, 2, :].copy()
        self.state[self.BACK, 2, :] = self.state[self.RIGHT, 2, :].copy()
        self.state[self.RIGHT, 2, :] = temp

    def _move_f(self):
        """Rotate the front face clockwise."""
        # Rotate front face
        self.state[self.FRONT] = np.rot90(self.state[self.FRONT], k=3)
        
        # Update adjacent faces
        temp = self.state[self.UP, 2, :].copy()
        self.state[self.UP, 2, :] = self.state[self.LEFT, ::-1, 2].copy()
        self.state[self.LEFT, :, 2] = self.state[self.DOWN, 0, ::-1].copy()
        self.state[self.DOWN, 0, :] = self.state[self.RIGHT, :, 0].copy()
        self.state[self.RIGHT, ::-1, 0] = temp

    def _move_b(self):
        """Rotate the back face clockwise."""
        # Rotate back face
        self.state[self.BACK] = np.rot90(self.state[self.BACK], k=3)
        
        # Update adjacent faces
        temp = self.state[self.UP, 0, :].copy()
        self.state[self.UP, 0, :] = self.state[self.RIGHT, ::-1, 2].copy()
        self.state[self.RIGHT, :, 2] = self.state[self.DOWN, 2, ::-1].copy()
        self.state[self.DOWN, 2, :] = self.state[self.LEFT, :, 0].copy()
        self.state[self.LEFT, ::-1, 0] = temp

    def scramble(self, num_moves: int = 20) -> RubiksCube:
        """Scramble the cube with random moves."""
        moves = list(MOVES.keys())
        for _ in range(num_moves):
            move = MOVES[random.choice(moves)]
            self.apply_move(move)
        return self

    def get_state_representation(self) -> np.ndarray:
        """
        Get a representation of the cube state suitable for ML models.
        
        Returns a one-hot encoded representation of the cube state.
        """
        # 6 faces × 3×3 positions × 6 colors (one-hot)
        one_hot = np.zeros((6, 3, 3, 6), dtype=np.float32)
        
        for face in range(6):
            for i in range(3):
                for j in range(3):
                    color = self.state[face, i, j]
                    one_hot[face, i, j, color] = 1.0
                    
        # Flatten to 1D array for model input
        return one_hot.reshape(-1)
    
    def __str__(self) -> str:
        """String representation of the cube."""
        result = []
        
        # Color mapping for terminal display
        color_names = ["W", "Y", "R", "O", "B", "G"]
        
        # Add each face
        for face in range(6):
            face_name = ["UP", "DOWN", "FRONT", "BACK", "RIGHT", "LEFT"][face]
            result.append(f"{face_name}:")
            
            for i in range(3):
                row = []
                for j in range(3):
                    color_idx = self.state[face, i, j]
                    row.append(color_names[color_idx])
                result.append(" ".join(row))
            result.append("")
            
        return "\n".join(result)