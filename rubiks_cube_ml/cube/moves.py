"""Moves for the Rubik's Cube."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List


class MoveType(Enum):
    """Type of move on the Rubik's Cube."""
    CLOCKWISE = auto()
    COUNTERCLOCKWISE = auto()
    DOUBLE = auto()


@dataclass
class Move:
    """
    Representation of a move on the Rubik's Cube.
    
    Attributes:
        name: Name of the move (e.g., 'R', 'L', 'U')
        move_type: Type of move (clockwise, counterclockwise, double)
    """
    name: str
    move_type: MoveType
    
    def __str__(self) -> str:
        """String representation of the move."""
        if self.move_type == MoveType.CLOCKWISE:
            return self.name
        elif self.move_type == MoveType.COUNTERCLOCKWISE:
            return f"{self.name}'"
        else:  # DOUBLE
            return f"{self.name}2"


# Define all possible moves
MOVES: Dict[str, Move] = {
    "R": Move("R", MoveType.CLOCKWISE),
    "R'": Move("R", MoveType.COUNTERCLOCKWISE),
    "R2": Move("R", MoveType.DOUBLE),
    "L": Move("L", MoveType.CLOCKWISE),
    "L'": Move("L", MoveType.COUNTERCLOCKWISE),
    "L2": Move("L", MoveType.DOUBLE),
    "U": Move("U", MoveType.CLOCKWISE),
    "U'": Move("U", MoveType.COUNTERCLOCKWISE),
    "U2": Move("U", MoveType.DOUBLE),
    "D": Move("D", MoveType.CLOCKWISE),
    "D'": Move("D", MoveType.COUNTERCLOCKWISE),
    "D2": Move("D", MoveType.DOUBLE),
    "F": Move("F", MoveType.CLOCKWISE),
    "F'": Move("F", MoveType.COUNTERCLOCKWISE),
    "F2": Move("F", MoveType.DOUBLE),
    "B": Move("B", MoveType.CLOCKWISE),
    "B'": Move("B", MoveType.COUNTERCLOCKWISE),
    "B2": Move("B", MoveType.DOUBLE),
}


def get_inverse_move(move: Move) -> Move:
    """Get the inverse of a move."""
    if move.move_type == MoveType.CLOCKWISE:
        return Move(move.name, MoveType.COUNTERCLOCKWISE)
    elif move.move_type == MoveType.COUNTERCLOCKWISE:
        return Move(move.name, MoveType.CLOCKWISE)
    else:  # DOUBLE moves are their own inverse
        return Move(move.name, MoveType.DOUBLE)