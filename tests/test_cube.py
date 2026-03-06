"""Tests for the Rubik's Cube representation."""

import pytest
import numpy as np

from rubiks_cube_ml.cube.cube import RubiksCube
from rubiks_cube_ml.cube.moves import Move, MoveType, MOVES


class TestRubiksCube:
    """Test cases for the RubiksCube class."""
    
    def test_init(self):
        """Test that the cube is initialized in a solved state."""
        cube = RubiksCube()
        
        # Check that each face has the correct color
        for face in range(6):
            # Get the color of the first piece on this face
            color = cube.state[face, 0, 0]
            
            # Check that all pieces on this face have the same color
            assert np.all(cube.state[face, :, :] == color)
    
    def test_is_solved(self):
        """Test the is_solved method."""
        cube = RubiksCube()
        assert cube.is_solved(), "Newly initialized cube should be solved"
        
        # Apply a move and check that the cube is no longer solved
        cube.apply_move(MOVES["R"])
        assert not cube.is_solved(), "Cube should not be solved after a move"
    
    def test_copy(self):
        """Test the copy method."""
        cube = RubiksCube()
        cube_copy = cube.copy()
        
        # Check that the state is the same
        assert np.array_equal(cube.state, cube_copy.state)
        
        # Modify the copy and check that the original is unchanged
        cube_copy.apply_move(MOVES["R"])
        assert not np.array_equal(cube.state, cube_copy.state)
        assert cube.is_solved()
        assert not cube_copy.is_solved()
    
    def test_move_r(self):
        """Test the R move."""
        cube = RubiksCube()
        
        # Store the initial state
        initial_up = cube.state[cube.UP, :, 2].copy()
        initial_front = cube.state[cube.FRONT, :, 2].copy()
        initial_down = cube.state[cube.DOWN, :, 2].copy()
        initial_back = cube.state[cube.BACK, :, 0].copy()
        
        # Apply the R move
        cube.apply_move(MOVES["R"])
        
        # Check that the right face was rotated correctly
        # In a clockwise rotation, the corners should move:
        # top-right -> bottom-right -> bottom-left -> top-left
        
        # Check that the adjacent faces were updated correctly
        # UP -> FRONT -> DOWN -> BACK -> UP
        assert np.array_equal(initial_up, cube.state[cube.BACK, ::-1, 0])
        assert np.array_equal(initial_front, cube.state[cube.UP, :, 2])
        assert np.array_equal(initial_down, cube.state[cube.FRONT, :, 2])
        assert np.array_equal(initial_back[::-1], cube.state[cube.DOWN, :, 2])
        
        # The move should be recorded in the history
        assert cube.move_history == [MOVES["R"]]
    
    def test_scramble_and_solve(self):
        """Test scrambling and then applying the inverse moves."""
        cube = RubiksCube()
        
        # Scramble with a known sequence
        moves = ["R", "U", "F"]
        for move_str in moves:
            cube.apply_move(MOVES[move_str])
        
        # Apply the inverse moves in reverse order
        inverse_moves = ["F'", "U'", "R'"]
        for move_str in inverse_moves:
            cube.apply_move(MOVES[move_str])
        
        # The cube should be solved again
        assert cube.is_solved(), "Cube should be solved after applying inverse moves"
    
    def test_get_state_representation(self):
        """Test the get_state_representation method."""
        cube = RubiksCube()
        
        # Get the state representation
        state_rep = cube.get_state_representation()
        
        # The shape should be (6 * 3 * 3 * 6,)
        assert state_rep.shape == (6 * 3 * 3 * 6,)
        
        # For a solved cube, each face should have 9 pieces of the same color
        # So each face should have 9 ones in the correct color channel
        state_3d = state_rep.reshape(6, 3, 3, 6)
        
        for face in range(6):
            color = cube.state[face, 0, 0]
            
            # Count the number of ones in this face's representation
            ones_count = np.sum(state_3d[face, :, :, color])
            assert ones_count == 9, f"Face {face} should have 9 pieces of color {color}"


class TestDoubleMoves:
    """Test cases for double moves (R2, L2, U2, D2, F2, B2)."""

    def test_r2_equals_r_twice(self):
        """Test that R2 is equivalent to applying R twice."""
        cube1 = RubiksCube()
        cube2 = RubiksCube()

        cube1.apply_move(MOVES["R2"])
        cube2.apply_move(MOVES["R"])
        cube2.apply_move(MOVES["R"])

        assert np.array_equal(cube1.state, cube2.state)

    def test_l2_equals_l_twice(self):
        """Test that L2 is equivalent to applying L twice."""
        cube1 = RubiksCube()
        cube2 = RubiksCube()

        cube1.apply_move(MOVES["L2"])
        cube2.apply_move(MOVES["L"])
        cube2.apply_move(MOVES["L"])

        assert np.array_equal(cube1.state, cube2.state)

    def test_u2_equals_u_twice(self):
        """Test that U2 is equivalent to applying U twice."""
        cube1 = RubiksCube()
        cube2 = RubiksCube()

        cube1.apply_move(MOVES["U2"])
        cube2.apply_move(MOVES["U"])
        cube2.apply_move(MOVES["U"])

        assert np.array_equal(cube1.state, cube2.state)

    def test_d2_equals_d_twice(self):
        """Test that D2 is equivalent to applying D twice."""
        cube1 = RubiksCube()
        cube2 = RubiksCube()

        cube1.apply_move(MOVES["D2"])
        cube2.apply_move(MOVES["D"])
        cube2.apply_move(MOVES["D"])

        assert np.array_equal(cube1.state, cube2.state)

    def test_f2_equals_f_twice(self):
        """Test that F2 is equivalent to applying F twice."""
        cube1 = RubiksCube()
        cube2 = RubiksCube()

        cube1.apply_move(MOVES["F2"])
        cube2.apply_move(MOVES["F"])
        cube2.apply_move(MOVES["F"])

        assert np.array_equal(cube1.state, cube2.state)

    def test_b2_equals_b_twice(self):
        """Test that B2 is equivalent to applying B twice."""
        cube1 = RubiksCube()
        cube2 = RubiksCube()

        cube1.apply_move(MOVES["B2"])
        cube2.apply_move(MOVES["B"])
        cube2.apply_move(MOVES["B"])

        assert np.array_equal(cube1.state, cube2.state)

    def test_double_move_is_self_inverse(self):
        """Test that applying a double move twice returns to original state."""
        for move_name in ["R2", "L2", "U2", "D2", "F2", "B2"]:
            cube = RubiksCube()
            original = cube.state.copy()

            cube.apply_move(MOVES[move_name])
            cube.apply_move(MOVES[move_name])

            assert np.array_equal(cube.state, original), f"{move_name} applied twice should return to original"


class TestPrimeMoves:
    """Test cases for prime/counterclockwise moves."""

    def test_r_prime_equals_r_three_times(self):
        """Test that R' is equivalent to applying R three times."""
        cube1 = RubiksCube()
        cube2 = RubiksCube()

        cube1.apply_move(MOVES["R'"])
        cube2.apply_move(MOVES["R"])
        cube2.apply_move(MOVES["R"])
        cube2.apply_move(MOVES["R"])

        assert np.array_equal(cube1.state, cube2.state)

    def test_move_and_prime_cancel(self):
        """Test that a move followed by its prime returns to original."""
        for face in ["R", "L", "U", "D", "F", "B"]:
            cube = RubiksCube()
            original = cube.state.copy()

            cube.apply_move(MOVES[face])
            cube.apply_move(MOVES[f"{face}'"])

            assert np.array_equal(cube.state, original), f"{face} then {face}' should return to original"