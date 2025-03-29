"""
mavsim_python: drawing tools
    - Beard & McLain, PUP, 2012
    - Update history:
        4/15/2019 - BGM
        7/13/2023 - RWB
        1/16/2024 - RWB
"""
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh
import os
from rotations import euler_to_rotation
from pathlib import Path

current_directory = Path(__file__).parent
path_str = str(current_directory)

# # Obtaining the current directory
# cwd = os.getcwd()

class DrawMav:
    def __init__(self, state, ax, scale = 4):
        """
        Draw the MAV.

        The input to this function is a (message) class with properties that define the state.
        The following properties are assumed:
            state.north  # north position
            state.east  # east position
            state.altitude   # altitude
            state.phi  # roll angle
            state.theta  # pitch angle
            state.psi  # yaw angle
        """
        self.unit_length = scale
        self.ax = ax

        mav_position = np.array([[state.north], [state.east], [-state.altitude]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R_bi = euler_to_rotation(state.phi, state.theta, state.psi)
        # R_bi = np.array([[state.r11, state.r12, state.r13],\
        #                  [state.r21, state.r22, state.r23],\
        #                  [state.r31, state.r32, state.r33]])
        # convert North-East Down to East-North-Up for rendering
        # self.R_ned = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        # self.R_ned = np.eye(3)
        # We perform a z rotation by -90 degrees
        self.R_ned = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        # get STL mesh
        stl_mesh = mesh.Mesh.from_file(path_str + '/aircraft1.stl')
        self.mav_points = self.unit_length * stl_mesh.points.reshape(-1, 3)
        self.mav_faces = np.arange(self.mav_points.shape[0]).reshape(-1, 3)

        # Rotate and translate points
        # transformed_points = self.rotate_points(self.mav_points, np.diag([-1, -1, 1]) @ self.R_ned)
        transformed_points = self.rotate_points(self.mav_points, self.R_ned)
        transformed_points = self.rotate_points(transformed_points, R_bi)
        transformed_points = self.translate_points(transformed_points, mav_position)

        # Add object to plot
        self.mav_body = self.add_object(transformed_points, self.mav_faces)

    def update(self, state):
        mav_position = np.array([[state.north], [state.east], [-state.altitude]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R_bi = euler_to_rotation(state.phi, state.theta, state.psi)
        # R_bi = np.array([[state.r11, state.r12, state.r13],\
        #                  [state.r21, state.r22, state.r23],\
        #                  [state.r31, state.r32, state.r33]])

        # Rotate and translate points
        transformed_points = self.rotate_points(self.mav_points, self.R_ned)
        transformed_points = self.rotate_points(transformed_points, R_bi)
        transformed_points = self.translate_points(transformed_points, mav_position)

        # Update object in the plot
        self.update_object(self.mav_body, transformed_points)

    def add_object(self, points, faces):
        """Add the MAV object to the Matplotlib plot."""
        poly3d = Poly3DCollection(points[faces], alpha=0.6, edgecolor='k')
        self.ax.add_collection3d(poly3d)
        return poly3d

    def update_object(self, object, points):
        """Update the MAV object in the Matplotlib plot."""
        # print("Points for updation are ", points)
        object.set_verts(points)
        self.ax.figure.canvas.draw_idle()

    def rotate_points(self, points, R):
        """Rotate points by the rotation matrix R."""
        rotated_points = points @ R.T
        return rotated_points

    def translate_points(self, points, translation):
        """Translate points by the vector translation."""
        translated_points = points + np.dot(np.ones([points.shape[0], 1]), translation.T)
        return translated_points


# ### Example usage
# class State:
#     north = 50
#     east = 0
#     altitude = 0
#     phi = 0
#     theta = 0
#     psi = 0

# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Set axis limits and labels
# ax.set_xlim(-50, 50)
# ax.set_ylim(-50, 50)
# ax.set_zlim(-50, 50)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# state = State()
# mav = DrawMav(state, ax)

# # Show the plot
# plt.show()
