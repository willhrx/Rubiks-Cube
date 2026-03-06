"""
Sample visualization script for the Rubik's Cube.

This script generates a visualization of a Rubik's Cube and a simple solution
sequence for demonstration purposes.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from rubiks_cube_ml.cube.cube import RubiksCube
from rubiks_cube_ml.cube.moves import MOVES
from rubiks_cube_ml.visualization.visualizer import CubeVisualizer


def main():
    """Generate sample visualizations."""
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Create a cube
    cube = RubiksCube()
    
    # Create a visualizer
    visualizer = CubeVisualizer()
    
    # Visualize the solved cube
    fig = visualizer.visualize(cube)
    plt.savefig("output/solved_cube.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Scramble the cube with a simple sequence
    scramble_moves = ["R", "U", "R'", "U'", "F'", "L", "F", "L'"]
    print(f"Scrambling with sequence: {scramble_moves}")
    
    for move_str in scramble_moves:
        cube.apply_move(MOVES[move_str])
    
    # Visualize the scrambled cube
    fig = visualizer.visualize(cube)
    plt.savefig("output/scrambled_cube.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create a simple "solution" (just the inverse of the scramble)
    solution = ["L", "F'", "L'", "F", "U", "R", "U'", "R'"]
    
    # Visualize the solution path
    fig = visualizer.plot_solution_path(cube, solution)
    plt.savefig("output/solution_path.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create an animation of the solution
    visualizer.animate_moves(cube, solution, interval=500, save_path="output/solution_animation.gif")
    
    print("Generated visualizations in the 'output' directory:")
    print("  - solved_cube.png")
    print("  - scrambled_cube.png")
    print("  - solution_path.png")
    print("  - solution_animation.gif")


if __name__ == "__main__":
    main()