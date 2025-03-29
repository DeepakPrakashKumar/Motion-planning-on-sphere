import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import os
import sys
from msg_state import MsgState
import time
import math
from rotations import euler_to_rotation

plt.rcParams['text.usetex'] = True
# plt.rcParams['figure.dpi'] = 150

# # Including the following command to ensure that python is able to find the relevant files afer changing directory
# sys.path.insert(0, '')
# # Obtaining the current directory
# cwd = os.getcwd()

# # Importing code for the sphere
# rel_path = '\Sphere code'
# os.chdir(cwd + rel_path)
# from Path_generation_sphere import generate_points_sphere

# # Returning to initial directory
# os.chdir(cwd)

from view_manager import ViewManager

# The code for Arrow3D is taken from https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
# The code for the animation is based on https://matplotlib.org/stable/gallery/mplot3d/wire3d_animation_sgskip.html
class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)
    
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)
    return arrow

setattr(Axes3D, 'arrow3D', _arrow3D)

def plot_trajectory(ini_config, fin_config, pos_global, tang_global_path, tang_normal_global_path,\
                     surf_normal_global_path, left_turns_pos, right_turns_pos, r, R, xgrid_size = [-20, 20],\
                     ygrid_size = [-20, 20], zgrid_size = [-20, 20], length_vec_orientation = 5,\
                     scale_aircraft = 3, video_name = False, elev = False, azim = False):
    # In this function, the trajectory is visualized and is animated.

    fig = plt.figure(figsize=(6.4,4.8)) # figsize=(12,10)
    ax = fig.add_subplot(projection='3d')

    plt.ion()

    plt.tight_layout()

    # Setting up an object for visualizing the aircraft
    if video_name != False:
        viewers = ViewManager(ax, animation=True, video=True, scale_aircraft=scale_aircraft, video_name = video_name)
    else:
        viewers = ViewManager(ax, animation=True, video=False, scale_aircraft=scale_aircraft, video_name = 'trajectory_aircraft.mp4')

    # We define the length of the arrow for representing the orientation
    length = length_vec_orientation

    # We also plot the spheres at the initial configuration
    xini_inner, yini_inner, zini_inner = generate_points_sphere([0, 0, 0], R)
    ax.plot_surface(xini_inner, yini_inner, zini_inner, color = 'grey', alpha = 0.4)

    # Draw latitude lines
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    for i in np.linspace(0, np.pi, 10):
        x_lat = R * np.cos(u) * np.sin(i)
        y_lat = R * np.sin(u) * np.sin(i)
        z_lat = R * np.ones(np.size(u)) * np.cos(i)
        ax.plot(x_lat, y_lat, z_lat, color='black', linestyle = ':', linewidth=0.5)

    # Draw longitude lines
    for i in np.linspace(0, 2 * np.pi, 20):
        x_lon = R * np.cos(i) * np.sin(v)
        y_lon = R * np.sin(i) * np.sin(v)
        z_lon = R * np.cos(v)
        ax.plot(x_lon, y_lon, z_lon, color='black', linestyle = ':', linewidth=0.5)

    ax.scatter(ini_config[0, 0], ini_config[1, 0], ini_config[2, 0], marker = 'o', s = 50, linewidth = 1.5,\
            color = 'r') # label = 'Initial point')
    # ax.scatter(fin_config[0, 0], fin_config[1, 0], fin_config[2, 0], marker = 'D', s = 50, linewidth = 1.5,\
    #         color = 'b', label = 'Final point')
    # We plot the orientation of the vehicle as well
    ax.arrow3D(ini_config[0, 0], ini_config[1, 0], ini_config[2, 0], length*ini_config[0, 1],\
                length*ini_config[1, 1], length*ini_config[2, 1], mutation_scale=20, fc='green', label = 'Tangent vector')
    ax.arrow3D(ini_config[0, 0], ini_config[1, 0], ini_config[2, 0], length*ini_config[0, 2],\
                length*ini_config[1, 2], length*ini_config[2, 2], mutation_scale=20, fc='red', label = 'Tangent normal vector')
    ax.arrow3D(ini_config[0, 0], ini_config[1, 0], ini_config[2, 0], length*ini_config[0, 0],\
                length*ini_config[1, 0], length*ini_config[2, 0], mutation_scale=20, fc='blue', label = 'Position vector')
    ax.arrow3D(fin_config[0, 0], fin_config[1, 0], fin_config[2, 0], length*fin_config[0, 1],\
                length*fin_config[1, 1], length*fin_config[2, 1], mutation_scale=20, fc='green')
    ax.arrow3D(fin_config[0, 0], fin_config[1, 0], fin_config[2, 0], length*fin_config[0, 2],\
                length*fin_config[1, 2], length*fin_config[2, 2], mutation_scale=20, fc='red')
    ax.arrow3D(fin_config[0, 0], fin_config[1, 0], fin_config[2, 0], length*fin_config[0, 0],\
                length*fin_config[1, 0], length*fin_config[2, 0], mutation_scale=20, fc='blue')
    
    # We check if left and right turns are to be plotted
    if len(left_turns_pos) != 0:
        ax.plot3D(left_turns_pos[:, 0], left_turns_pos[:, 1], left_turns_pos[:, 2], linewidth = 2.5, color = 'r', linestyle = ':', label = 'Left turn')
    if len(right_turns_pos) != 0:
        ax.plot3D(right_turns_pos[:, 0], right_turns_pos[:, 1], right_turns_pos[:, 2], linewidth = 2.5, color = 'tab:orange', linestyle = ':', label = 'Right turn')

    ax.plot3D(pos_global[:, 0], pos_global[:, 1], pos_global[:, 2], linewidth = 2.5, color = 'k', label = 'Trajectory')
    
    ax.set_xlabel(r'$X$', fontsize = 18)
    ax.set_ylabel(r'$Y$', fontsize = 18)
    ax.set_zlabel(r'$Z$', fontsize = 18)

    ax.set_xlim(xgrid_size[0], xgrid_size[1])
    ax.set_ylim(ygrid_size[0], ygrid_size[1])
    ax.set_zlim(zgrid_size[0], zgrid_size[1])

    ax.tick_params(axis='both', which='major', labelsize = 14)
    ax.tick_params(axis='both', which='minor', labelsize = 14)

    # # Plotting the inflection points if they exist
    # if len(xinf) != 0:
    #     for i in range(len(xinf)):
    #         if i == 0:
    #             ax.scatter(xinf[i], yinf[i], zinf[i], marker = 'o', s = 50, linewidth = 1.5,\
    #                     color = 'magenta', label = 'Inflection point')
    #         else:
    #             ax.scatter(xinf[i], yinf[i], zinf[i], marker = 'o', s = 50, linewidth = 1.5,\
    #                     color = 'magenta')

    # ax.legend(fontsize = 14, loc = 0, ncol = 2)

    if elev != False and azim != False:
        ax.view_init(elev = elev, azim = azim)

    # We now begin plotting the configuration
    locplot = None; tangplot = None; surfplot = None; tangnorm_plot = None
    true_state = MsgState()
    for i in range(1, len(pos_global[:, 0])):

        # If a plot already exists, we remove it
        if locplot:
            locplot.remove()
            # tangplot.remove()
            # tangnorm_plot.remove()
            # surfplot.remove()
        
        # if np.mod(i, 10) == 0:
        # We plot the current configuration
        locplot = ax.scatter(pos_global[i, 0], pos_global[i, 1], pos_global[i, 2], marker = 'o', linewidth = 1.5,\
            color = 'k')
        # tangplot = ax.arrow3D(pos_global[i, 0], pos_global[i, 1], pos_global[i, 2], length*tang_global_path[i, 0],\
        #         length*tang_global_path[i, 1], length*tang_global_path[i, 2], mutation_scale=20, fc='green')
        # tangnorm_plot = ax.arrow3D(pos_global[i, 0], pos_global[i, 1], pos_global[i, 2], length*tang_normal_global_path[i, 0],\
        #             length*tang_normal_global_path[i, 1], length*tang_normal_global_path[i, 2], mutation_scale=20, fc='red')
        # surfplot = ax.arrow3D(pos_global[i, 0], pos_global[i, 1], pos_global[i, 2], length*surf_normal_global_path[i, 0],\
        #             length*surf_normal_global_path[i, 1], length*surf_normal_global_path[i, 2], mutation_scale=20, fc='blue')

        # We update the state of the aircraft
        true_state.north = pos_global[i, 0]
        true_state.east = pos_global[i, 1]
        true_state.altitude = -pos_global[i, 2]

        # Constructing the rotation matrix
        R_mat = np.array([[tang_global_path[i, 0], tang_normal_global_path[i, 0], surf_normal_global_path[i, 0]],\
                        [tang_global_path[i, 1], tang_normal_global_path[i, 1], surf_normal_global_path[i, 1]],\
                        [tang_global_path[i, 2], tang_normal_global_path[i, 2], surf_normal_global_path[i, 2]]])

        # We compute the orientation of the aircraft
        # Calculating the angles. The net rotation matrix is given below. Here, psi is yaw angle, theta is pitch, and phi is roll angle.
        """cψcθ −sψcφ + cψsθsφ sψsφ + cψsθcφ
        sψcθ cψcφ + sψsθsφ −cψsφ + sψsθcφ
        −sθ cθsφ cθcφ"""
        pitch_angle_val = np.nan; yaw_angle_val = np.nan; roll_angle_val = np.nan

        if math.sqrt((tang_global_path[i][0])**2 + (tang_global_path[i][1])**2) <= 10**(-8):
            
            pitch_angle = -np.sign(tang_global_path[i][2])*math.pi/2
            # We set roll angle to be zero
            roll_angle = 0.0
            yaw_angle = math.atan2(-tang_normal_global_path[i][0], tang_normal_global_path[i][1])

            # We check if the desired rotation matrix matches
            R_net = euler_to_rotation(roll_angle, pitch_angle, yaw_angle)

            if abs(max(map(max, R_mat - R_net))) <= 10**(-4) and abs(min(map(min, R_mat - R_net))) <= 10**(-4):
                    
                pitch_angle_val = pitch_angle
                yaw_angle_val = yaw_angle
                roll_angle_val = roll_angle

        else:

            # pitch_angle = math.atan2(-tang_global_path[i][2], math.sqrt((tang_global_path[i][0])**2 + (tang_global_path[i][1])**2))
            # yaw_angle = math.atan2(tang_global_path[i][1], tang_global_path[i][0])
            # roll_angle = math.atan2(surf_normal_global_path[i][2], surf_normal_global_path[i][2])
            pitch_angle = [math.atan2(-tang_global_path[i][2], math.sqrt((tang_global_path[i][0])**2 + (tang_global_path[i][1])**2)),\
                           math.atan2(-tang_global_path[i][2], -math.sqrt((tang_global_path[i][0])**2 + (tang_global_path[i][1])**2))]
            for angle in pitch_angle:

                yaw_angle = math.atan2(tang_global_path[i][1]/math.cos(angle), tang_global_path[i][0]/math.cos(angle))
                roll_angle = math.atan2(tang_normal_global_path[i][2]/math.cos(angle), surf_normal_global_path[i][2]/math.cos(angle))

                # We check if the desired rotation matrix matches
                R_net = euler_to_rotation(roll_angle, angle, yaw_angle)

                if abs(max(map(max, R_mat - R_net))) <= 10**(-4) and abs(min(map(min, R_mat - R_net))) <= 10**(-4):
                    
                    pitch_angle_val = angle
                    yaw_angle_val = yaw_angle
                    roll_angle_val = roll_angle
                    break

        if np.isnan(pitch_angle_val):

            print('R is ', R_mat, ' and R_net is ', R_net)
            raise Exception("Could not find euler angles.")
        # true_state.psi = yaw_angle[i]
        # true_state.theta = pitch_angle[i]
        # true_state.phi = roll_angle[i]
        true_state.psi = yaw_angle_val
        true_state.theta = pitch_angle_val
        true_state.phi = roll_angle_val

        # Plotting the aircraft
        viewers.update(
            fig,
            time.time(),
            true_state = true_state,  # true states
        )
        
        # ax.set_xlim(pos_global[i, 0] - 2, pos_global[i, 0] + 2)
        # ax.set_ylim(pos_global[i, 1] - 1.5, pos_global[i, 1] + 1.5)
        # ax.set_zlim(pos_global[i, 2] - 1.5, pos_global[i, 2] + 1.5)

        # ax.view_init(elev = pitch_angle, azim = yaw_angle, roll=0)
        # Updating the video

        print('Printing the ', i, 'th point')
        plt.pause(0.05)

    # We close the simulation
    viewers.close()

    plt.ioff()  # Turn off interactive mode

    # We show the trajectory plot
    plt.show()

def generate_points_sphere(center, R):
    '''
    This function generates points on a sphere whose center is given by the variable
    "center" and with a radius of R.

    Parameters
    ----------
    center : Numpy 1x3 array
        Contains the coordinates corresponding to the center of the sphere.
    R : Scalar
        Contains the radius of the sphere.

    Returns
    -------
    x_grid : Numpy nd array
        Contains the x-coordinate of the points on the sphere.
    y_grid : Numpy nd array
        Contains the y-coordinate of the points on the sphere.
    z_grid : Numpy nd array
        Contains the z-coordinate of the points on the sphere.

    '''
    
    theta = np.linspace(0, 2*np.pi, 50)
    phi = np.linspace(-np.pi/2, np.pi/2, 50)
    
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    # Finding the coordinates of the points on the sphere in the global frame
    x_grid = center[0] + R*np.cos(theta_grid)*np.cos(phi_grid)
    y_grid = center[1] + R*np.sin(theta_grid)*np.cos(phi_grid)
    z_grid = center[2] + R*np.sin(phi_grid)
    
    return x_grid, y_grid, z_grid