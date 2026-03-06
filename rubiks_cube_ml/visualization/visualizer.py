"""3D visualization of the Rubik's Cube."""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from typing import List, Tuple, Optional
import time

from ..cube.cube import RubiksCube, Color


class CubeVisualizer:
    """
    3D visualization of the Rubik's Cube using Matplotlib.
    
    This class provides methods to visualize the Rubik's Cube in 3D
    and animate sequences of moves.
    """
    
    # RGB color mapping
    COLOR_MAP = {
        Color.WHITE.value: (1.0, 1.0, 1.0),  # White
        Color.YELLOW.value: (1.0, 1.0, 0.0),  # Yellow
        Color.RED.value: (1.0, 0.0, 0.0),  # Red
        Color.ORANGE.value: (1.0, 0.5, 0.0),  # Orange
        Color.BLUE.value: (0.0, 0.0, 1.0),  # Blue
        Color.GREEN.value: (0.0, 0.8, 0.0),  # Green
    }
    
    def __init__(self, figsize: Tuple[int, int] = (10, 10)):
        """Initialize the visualizer."""
        self.figsize = figsize
        self.fig = None
        self.ax = None

    def _create_figure(self):
        """Create a new 3D figure."""
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])
        self.ax.set_axis_off()
        
    def _plot_cubie(self, x: int, y: int, z: int, colors: List[Optional[int]]):
        """
        Plot a single cubie at position (x, y, z) with given colors.
        
        Args:
            x, y, z: Position of the cubie (-1, 0, 1)
            colors: List of 6 colors (or None) for [+x, -x, +y, -y, +z, -z] faces
        """
        # Size of the cubie
        s = 0.45
        
        # Vertices of the cubie
        vertices = [
            [x + s, y + s, z + s], [x + s, y + s, z - s],
            [x + s, y - s, z + s], [x + s, y - s, z - s],
            [x - s, y + s, z + s], [x - s, y + s, z - s],
            [x - s, y - s, z + s], [x - s, y - s, z - s]
        ]
        
        # Faces of the cubie (each face is defined by 4 vertices)
        faces = [
            [vertices[0], vertices[1], vertices[3], vertices[2]],  # +x face
            [vertices[4], vertices[5], vertices[7], vertices[6]],  # -x face
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # +y face
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # -y face
            [vertices[0], vertices[2], vertices[6], vertices[4]],  # +z face
            [vertices[1], vertices[3], vertices[7], vertices[5]]   # -z face
        ]
        
        # Plot each face with its color
        for i, (face, color) in enumerate(zip(faces, colors)):
            if color is not None:  # Only plot faces with a color
                face_color = self.COLOR_MAP[color]
                collection = Poly3DCollection([face], alpha=1.0)
                collection.set_facecolor(face_color)
                collection.set_edgecolor('black')
                self.ax.add_collection3d(collection)

    def visualize(self, cube: RubiksCube, view_angles: Tuple[float, float] = (30, 30)):
        """
        Visualize the current state of the cube.
        
        Args:
            cube: RubiksCube object to visualize
            view_angles: (elevation, azimuth) angles for the 3D view
        
        Returns:
            The created figure
        """
        self._create_figure()
        
        # Set viewing angle
        self.ax.view_init(elev=view_angles[0], azim=view_angles[1])
        
        # Map from cube coordinates to 3D space
        # The cube is centered at the origin
        coord_map = {0: -1, 1: 0, 2: 1}
        
        # Plot each cubie
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    # Skip the interior cubie
                    if x == 1 and y == 1 and z == 1:
                        continue
                    
                    # Determine colors for each face of this cubie
                    colors = [None] * 6
                    
                    # +x face (RIGHT)
                    if x == 2:
                        colors[0] = cube.state[RubiksCube.RIGHT, 2-z, y]
                    # -x face (LEFT)
                    if x == 0:
                        colors[1] = cube.state[RubiksCube.LEFT, 2-z, 2-y]
                    # +y face (UP)
                    if y == 2:
                        colors[2] = cube.state[RubiksCube.UP, 2-x, z]
                    # -y face (DOWN)
                    if y == 0:
                        colors[3] = cube.state[RubiksCube.DOWN, x, z]
                    # +z face (FRONT)
                    if z == 2:
                        colors[4] = cube.state[RubiksCube.FRONT, 2-y, x]
                    # -z face (BACK)
                    if z == 0:
                        colors[5] = cube.state[RubiksCube.BACK, 2-y, 2-x]
                    
                    # Plot the cubie
                    self._plot_cubie(coord_map[x], coord_map[y], coord_map[z], colors)
        
        self.ax.set_title("Rubik's Cube")
        plt.tight_layout()
        
        return self.fig
    
    def animate_moves(self, cube: RubiksCube, moves: List[str], 
                     interval: int = 500, save_path: Optional[str] = None):
        """
        Create an animation of applying a sequence of moves to the cube.
        
        Args:
            cube: Initial RubiksCube state
            moves: List of move notations (e.g., ["R", "U'", "F2"])
            interval: Time between frames (ms)
            save_path: If provided, save the animation to this file path
            
        Returns:
            The animation object
        """
        from ..cube.moves import MOVES
        
        # Make a deep copy of the cube to avoid modifying the original
        cube_copy = cube.copy()
        
        # Create the initial figure
        self._create_figure()
        
        # Function to update the plot for each frame
        def update(frame):
            self.ax.clear()
            self.ax.set_xlim([-2, 2])
            self.ax.set_ylim([-2, 2])
            self.ax.set_zlim([-2, 2])
            self.ax.set_axis_off()
            
            # Apply the move for this frame
            if frame > 0 and frame <= len(moves):
                move = MOVES[moves[frame-1]]
                cube_copy.apply_move(move)
                
            # Set title to show the current move
            if frame == 0:
                self.ax.set_title("Initial State")
            else:
                self.ax.set_title(f"Move: {moves[frame-1]}")
            
            # Visualize the current state
            # Map from cube coordinates to 3D space
            coord_map = {0: -1, 1: 0, 2: 1}
            
            # Plot each cubie
            for x in range(3):
                for y in range(3):
                    for z in range(3):
                        # Skip the interior cubie
                        if x == 1 and y == 1 and z == 1:
                            continue
                        
                        # Determine colors for each face of this cubie
                        colors = [None] * 6
                        
                        # +x face (RIGHT)
                        if x == 2:
                            colors[0] = cube_copy.state[RubiksCube.RIGHT, 2-z, y]
                        # -x face (LEFT)
                        if x == 0:
                            colors[1] = cube_copy.state[RubiksCube.LEFT, 2-z, 2-y]
                        # +y face (UP)
                        if y == 2:
                            colors[2] = cube_copy.state[RubiksCube.UP, 2-x, z]
                        # -y face (DOWN)
                        if y == 0:
                            colors[3] = cube_copy.state[RubiksCube.DOWN, x, z]
                        # +z face (FRONT)
                        if z == 2:
                            colors[4] = cube_copy.state[RubiksCube.FRONT, 2-y, x]
                        # -z face (BACK)
                        if z == 0:
                            colors[5] = cube_copy.state[RubiksCube.BACK, 2-y, 2-x]
                        
                        # Plot the cubie
                        self._plot_cubie(coord_map[x], coord_map[y], coord_map[z], colors)
            
            # Set the view angle
            self.ax.view_init(elev=30, azim=30)
        
        # Create the animation
        ani = animation.FuncAnimation(
            self.fig, update, frames=len(moves) + 1, 
            interval=interval, blit=False
        )
        
        # Save the animation if requested
        if save_path:
            ani.save(save_path, writer='pillow', fps=1000/interval)
            
        return ani
    
    def plot_solution_path(self, cube: RubiksCube, solution_moves: List[str], 
                         save_path: Optional[str] = None):
        """
        Plot a grid of images showing the solution path.
        
        Args:
            cube: Initial RubiksCube state
            solution_moves: List of move notations for the solution
            save_path: If provided, save the figure to this file path
            
        Returns:
            The figure object
        """
        from ..cube.moves import MOVES
        
        # Make a deep copy of the cube to avoid modifying the original
        cube_copy = cube.copy()
        
        # Calculate grid dimensions
        n_frames = len(solution_moves) + 1  # Initial state + each move
        cols = min(5, n_frames)
        rows = (n_frames + cols - 1) // cols
        
        # Create a new figure
        fig = plt.figure(figsize=(cols * 4, rows * 4))
        
        # Plot the initial state
        ax = fig.add_subplot(rows, cols, 1, projection='3d')
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        ax.set_axis_off()
        ax.set_title("Initial State")
        
        # Map from cube coordinates to 3D space
        coord_map = {0: -1, 1: 0, 2: 1}
        
        # Visualize the initial state
        self._visualize_cube_on_axis(cube_copy, ax, coord_map)
        
        # Apply moves one by one and plot
        for i, move_str in enumerate(solution_moves):
            move = MOVES[move_str]
            cube_copy.apply_move(move)
            
            ax = fig.add_subplot(rows, cols, i + 2, projection='3d')
            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.set_zlim([-2, 2])
            ax.set_axis_off()
            ax.set_title(f"Move {i+1}: {move_str}")
            
            self._visualize_cube_on_axis(cube_copy, ax, coord_map)
        
        plt.tight_layout()
        
        # Save the figure if requested
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            
        return fig
    
    def _visualize_cube_on_axis(self, cube: RubiksCube, ax, coord_map):
        """Helper to visualize a cube on a specific axis."""
        # Plot each cubie
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    # Skip the interior cubie
                    if x == 1 and y == 1 and z == 1:
                        continue
                    
                    # Determine colors for each face of this cubie
                    colors = [None] * 6
                    
                    # +x face (RIGHT)
                    if x == 2:
                        colors[0] = cube.state[RubiksCube.RIGHT, 2-z, y]
                    # -x face (LEFT)
                    if x == 0:
                        colors[1] = cube.state[RubiksCube.LEFT, 2-z, 2-y]
                    # +y face (UP)
                    if y == 2:
                        colors[2] = cube.state[RubiksCube.UP, 2-x, z]
                    # -y face (DOWN)
                    if y == 0:
                        colors[3] = cube.state[RubiksCube.DOWN, x, z]
                    # +z face (FRONT)
                    if z == 2:
                        colors[4] = cube.state[RubiksCube.FRONT, 2-y, x]
                    # -z face (BACK)
                    if z == 0:
                        colors[5] = cube.state[RubiksCube.BACK, 2-y, 2-x]
                    
                    # Plot the cubie
                    s = 0.45
                    
                    # Vertices of the cubie
                    x_pos, y_pos, z_pos = coord_map[x], coord_map[y], coord_map[z]
                    vertices = [
                        [x_pos + s, y_pos + s, z_pos + s], [x_pos + s, y_pos + s, z_pos - s],
                        [x_pos + s, y_pos - s, z_pos + s], [x_pos + s, y_pos - s, z_pos - s],
                        [x_pos - s, y_pos + s, z_pos + s], [x_pos - s, y_pos + s, z_pos - s],
                        [x_pos - s, y_pos - s, z_pos + s], [x_pos - s, y_pos - s, z_pos - s]
                    ]
                    
                    # Faces of the cubie
                    faces = [
                        [vertices[0], vertices[1], vertices[3], vertices[2]],  # +x face
                        [vertices[4], vertices[5], vertices[7], vertices[6]],  # -x face
                        [vertices[0], vertices[1], vertices[5], vertices[4]],  # +y face
                        [vertices[2], vertices[3], vertices[7], vertices[6]],  # -y face
                        [vertices[0], vertices[2], vertices[6], vertices[4]],  # +z face
                        [vertices[1], vertices[3], vertices[7], vertices[5]]   # -z face
                    ]
                    
                    # Plot each face with its color
                    for i, (face, color) in enumerate(zip(faces, colors)):
                        if color is not None:  # Only plot faces with a color
                            face_color = self.COLOR_MAP[color]
                            collection = Poly3DCollection([face], alpha=1.0)
                            collection.set_facecolor(face_color)
                            collection.set_edgecolor('black')
                            ax.add_collection3d(collection)
        
        # Set the view angle
        ax.view_init(elev=30, azim=30)