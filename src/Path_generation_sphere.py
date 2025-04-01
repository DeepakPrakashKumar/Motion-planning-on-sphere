# -*- coding: utf-8 -*-
import numpy as np
import math
from math import cos as cos
from math import sin as sin
from math import sqrt as sqrt
import os
import copy
import numpy.polynomial.polynomial as poly
import sys
from CubicEquationSolver import solve

# Importing the functions
from plotting_class import plotting_functions

def operators_segments(ini_config, phi, r, R, seg_type = 'l'):
    '''
    This function defines the operators corresponding to left, , and great
    circle turns on a sphere.

    Parameters
    ----------
    ini_config : Numpy array
        Contains the configuration before the segment. The syntax followed is as
        follows:
            The first column contains the position.
            The second column contains the tangent vector.
            The third column contains the tangent-normal vector.
    phi : Scalar
        Describes the angle of the turn.
    r : Scalar
        Radius of the tight turn.
    R : Scalar
        Radius of the sphere.
    seg_type : Character, optional
        Defines the type of segment considered.

    Returns
    -------
    fin_config : Numpy array
        Contains the configuration after the corresponding segment. The syntax
        followed is the same as the initial configuration.

    '''
        
    # Defining the scaled radius of the turn
    rb = r/R
    
    # Defining the matrix corresponding to each segment type given the angle
    # of the turn
    if seg_type.lower() == 'g':
        
        R = np.array([[math.cos(phi), -(1/R)*math.sin(phi), 0],\
                      [R*math.sin(phi), math.cos(phi), 0],\
                      [0, 0, 1]])
    
    elif seg_type.lower() == 'l':
        
        R = np.array([[1 - (1 - math.cos(phi))*rb**2, -(rb/R)*math.sin(phi),\
                       (1/R)*(1 - math.cos(phi))*rb*math.sqrt(1 - rb**2)],\
                      [r*math.sin(phi), math.cos(phi), -math.sin(phi)*math.sqrt(1 - rb**2)],\
                      [(1 - math.cos(phi))*r*math.sqrt(1 - rb**2),\
                       math.sin(phi)*math.sqrt(1 - rb**2),\
                       math.cos(phi) + (1 - math.cos(phi))*rb**2]])
            
    elif seg_type.lower() == 'r':
        
        R = np.array([[1 - (1 - math.cos(phi))*rb**2, -(rb/R)*math.sin(phi),\
                       -(1/R)*(1 - math.cos(phi))*rb*math.sqrt(1 - rb**2)],\
                      [r*math.sin(phi), math.cos(phi), math.sin(phi)*math.sqrt(1 - rb**2)],\
                      [-(1 - math.cos(phi))*r*math.sqrt(1 - rb**2),\
                       -math.sin(phi)*math.sqrt(1 - rb**2),\
                       math.cos(phi) + (1 - math.cos(phi))*rb**2]])
            
    # Obtaining the final configuration
    fin_config = np.matmul(ini_config, R)
    
    return fin_config

def Seg_pts(ini_config, phi, r, R, seg_type = 'l', dist_disc = 0.025):
    '''
    This function generates points along the segment on a sphere. Moreover, the
    tangent vector at the generated points are also returned.

    Parameters
    ----------
    ini_config : Numpy array
        Contains the configuration before the segment. The syntax followed is as
        follows:
            The first column contains the position.
            The second column contains the tangent vector.
            The third column contains the tangent-normal vector.
    phi : Scalar
        Describes the angle of the turn.
    r : Scalar
        Radius of the tight turn.
    R : Scalar
        Radius of the sphere.
    seg_type : Character, optional
        Defines the type of segment considered.
    dist_disc : Optional
        Describes the distance between points along the segment.

    Returns
    -------
    pos_array : Numpy array
        Contains the coordinates of points along the segment on a sphere.
    tang_array : Numpy array
        Contains the tangent vector at the generated points along the segment on a sphere.

    '''
    
    # Checking if a valid segment type is passed
    seg_types_allowed = ['l', 'r', 'g']
    
    if seg_type.lower() not in seg_types_allowed:
        
        raise Exception('Incorrect path type passed.')
        
    # Discretizing the angle of the turn
    if seg_type.lower() == 'g':
        
        num_disc = int(np.ceil(phi*R/dist_disc))
        
    else:
        
        num_disc = int(np.ceil(phi*r/dist_disc))
        
    # Defining the discretization of the angle
    phi_disc = np.linspace(0, phi, num_disc)
    
    # Declaring the arrays used to store the positions and the tangent vectors
    pos_array = np.empty((num_disc, 3))
    tang_array = np.empty((num_disc, 3))
    
    for i in range(len(phi_disc)):
        
        # Running the function in which the operators are defined
        config_turn = operators_segments(ini_config, phi_disc[i], r, R, seg_type)
        # Extracting the position and tangent vectors from config_turn
        pos_array[i, 0] = config_turn[0, 0]
        pos_array[i, 1] = config_turn[1, 0]
        pos_array[i, 2] = config_turn[2, 0]
        tang_array[i, 0] = config_turn[0, 1]
        tang_array[i, 1] = config_turn[1, 1]
        tang_array[i, 2] = config_turn[2, 1]
    
    return pos_array, tang_array

def points_path(ini_config, r, R, angle_segments, path_type = 'lgl'):
    '''
    This function generates points along a path on the sphere. Moreover, points
    along the circles corresponding to each segment of the path are also returned.

    Parameters
    ----------
    ini_config : Numpy array
        Contains the configuration before the segment. The syntax followed is as
        follows:
            The first column contains the position.
            The second column contains the tangent vector.
            The third column contains the tangent-normal vector.
    r : Scalar
        Radius of the tight turn.
    R : Scalar
        Radius of the sphere.
    angle_segments : Numpy array
        Contains the angle for each segment of the path.
    path_type : String, optional
        Defines the type of path considered. The default is 'lgl'.

    Returns
    -------
    x_coords_path, y_coords_path, z_coords_path : Numpy arrays
        Contains the coordinates of points along the path.
    fin_config_path : Numpy array
        Contains the final configuration obtained after the path.
    x_coords_circles, y_coords_circles, z_coords_circles : Numpy arrays
        Contains the coordinates of points along the circles corresponding 
        to each segment of the path.
    Tx_path, Ty_path, Tz_path : Numpy arrays
        Contains the direction cosines of the tangent vector along the path.

    '''
            
    if len(angle_segments) < len(path_type):
        
        raise Exception('Number of parameters of the path is lesser than the number '\
                        + 'of segments of the path.')
            
    # Declaring the position array to store coordinates of points along the path
    x_coords_path = []
    y_coords_path = []
    z_coords_path = []
    Tx_path = []; Ty_path = []; Tz_path = []
    # Declaring the position array to store coordinates of points along the circles
    # corresponding to the path
    x_coords_circles = []
    y_coords_circles = []
    z_coords_circles = []
    
    # Storing the initial configuration before every segment
    ini_config_seg = ini_config
    
    for i in range(len(path_type)):
        
        points_path_seg, tang_path_seg = Seg_pts(ini_config_seg, angle_segments[i], r, R, path_type[i])
        # Appending the obtained points to the arrays, except the last point (if not the final segment)
        # We ignore the last point because it is the same as the first point of the next segment
        if i < len(path_type) - 1:

            x_coords_path = np.append(x_coords_path, points_path_seg[:-1, 0])
            y_coords_path = np.append(y_coords_path, points_path_seg[:-1, 1])
            z_coords_path = np.append(z_coords_path, points_path_seg[:-1, 2])
            Tx_path = np.append(Tx_path, tang_path_seg[:-1, 0])
            Ty_path = np.append(Ty_path, tang_path_seg[:-1, 1])
            Tz_path = np.append(Tz_path, tang_path_seg[:-1, 2])

        else:

            x_coords_path = np.append(x_coords_path, points_path_seg[:, 0])
            y_coords_path = np.append(y_coords_path, points_path_seg[:, 1])
            z_coords_path = np.append(z_coords_path, points_path_seg[:, 2])
            Tx_path = np.append(Tx_path, tang_path_seg[:, 0])
            Ty_path = np.append(Ty_path, tang_path_seg[:, 1])
            Tz_path = np.append(Tz_path, tang_path_seg[:, 2])
        
        # Updating the initial configuration for the next segment to the final 
        # configuration of the considered segment of the path
        ini_config_seg = operators_segments(ini_config_seg, angle_segments[i], r, R, path_type[i]) 
        
        # Obtaining the points along the circle corresponding to the ith segment of the path
        # NOTE THAT THE POINTS ARE OBTAINED SUCH THAT THEY DO NOT OVERLAP WITH POINTS ON THE PATH
        points_path_seg, _ = Seg_pts(ini_config_seg, 2*math.pi - angle_segments[i],\
                                     r, R, path_type[i])
        x_coords_circles = np.append(x_coords_circles, points_path_seg[:, 0])
        y_coords_circles = np.append(y_coords_circles, points_path_seg[:, 1])
        z_coords_circles = np.append(z_coords_circles, points_path_seg[:, 2])
        
    fin_config_path = ini_config_seg
    
    return x_coords_path, y_coords_path, z_coords_path, fin_config_path, x_coords_circles,\
        y_coords_circles, z_coords_circles, Tx_path, Ty_path, Tz_path
        
def modifying_initial_final_configurations_unit_sphere(ini_config, fin_config, R):
    '''
    In this function, the initial and final configuration is modified for the purpose
    of inverse kinematics such that
        - the sphere is unit
        - the initial configuration is the identity matrix, i.e., the initial
        location is on the x-axis and the initial tangent vector is oriented along
        the y-axis.

    Parameters
    ----------
    ini_config : Numpy array
        Contains the initial configuration. The syntax followed is as
        follows:
            The first column contains the position.
            The second column contains the tangent vector.
            The third column contains the tangent-normal vector.
    fin_config : Numpy array
        Contains the final configuration. The same syntax used for ini_config is
        used here.
    R : Scalar
        Radius of the sphere.

    Returns
    -------
    fin_config_mod : Numpy array
        Contains the final configuration lying on a unit sphere such that the initial
        configuration matrix is the identity matrix.

    '''
    
    # Obtaining the new final configuration after scaling
    fin_config_scaled = np.array([[fin_config[0, 0]/R, fin_config[0, 1], fin_config[0, 2]],\
                                  [fin_config[1, 0]/R, fin_config[1, 1], fin_config[1, 2]],\
                                  [fin_config[2, 0]/R, fin_config[2, 1], fin_config[2, 2]]])
    # Obtaining the scaled initial configuration
    ini_config_scaled = np.array([[ini_config[0, 0]/R, ini_config[0, 1], ini_config[0, 2]],\
                                  [ini_config[1, 0]/R, ini_config[1, 1], ini_config[1, 2]],\
                                  [ini_config[2, 0]/R, ini_config[2, 1], ini_config[2, 2]]])
    # Modifying the final configuration such that the initial configuration coincides
    # with the identity matrix. NOTE: For any path, Rfinal = Rinitial x Rnetrotation. Hence,
    # Rfinal_mod = Rinitial**T x Rfinal
    fin_config_mod = np.matmul(ini_config_scaled.transpose(), fin_config_scaled)
    
    return fin_config_mod

def path_generation_sphere_three_seg(ini_config, fin_config, r, R, path_type = 'lgl', tol_path = 10**(-4)):
    '''
    In this function, the chosen three-segment path to connect the chosen initial and final configurations
    are constructed, if they exist.

    Parameters
    ----------
    ini_config : Numpy array
        Contains the initial configuration. The syntax followed is as
        follows:
            The first column contains the position.
            The second column contains the tangent vector.
            The third column contains the tangent-normal vector.
    fin_config : Numpy array
        Contains the final configuration. The same syntax used for ini_config is
        used here.
    r : Scalar
        Radius of the tight turn.
    R : Scalar
        Radius of the sphere.
    path_type : String, optional
        Defines the type of path considered. The default is 'lgl'.

    Returns
    -------
    path_params : Numpy array
        Contains the parameters associated with the generated path. Note that multiple paths
        of the same type can be possibly generated. For each path, the path length, phi1, phi2,
        phi3 are included as an array, where phii denotes the arc angle of the ith segment.

    '''
    
    # Modifying the configurations and the parameters of the turn
    fin_config_mod = modifying_initial_final_configurations_unit_sphere(ini_config, fin_config, R)
    r_mod = r/R
    
    path_types = ['lgl', 'rgr', 'lgr', 'rgl', 'lrl', 'rlr']
    
    if path_type not in path_types:
        
        raise Exception('Incorrect path type passed.')
    
    # We check if the considered path is RGL or RGR
    # In this case, we modify the configurations so that we construct an LGR and LRL path, respectively,
    # and use its parameters for the RGL and RLR paths
    path_type_ini = path_type

    fin_config_mod_path = copy.deepcopy(fin_config_mod)
    if path_type in ['rgl', 'rlr']:

        # We reflect the final location and tangent vector about the XY plane
        for i in range(3):
            fin_config_mod_path[2, i] = -fin_config_mod[2, i]
        # We reflect the tangent-normal about the XY plane, but also reverse it
        # after to ensure that we have a rotation matrix
        for i in range(3):
            fin_config_mod_path[i, 2] = -fin_config_mod_path[i, 2]
        
        if path_type == 'rgl':
            path_type = 'lgr'
        elif path_type == 'rlr':
            path_type = 'lrl'
        
    # Defining variables corresponding to the final configuration
    alpha11 = fin_config_mod_path[0, 0]; alpha12 = fin_config_mod_path[0, 1]; alpha13 = fin_config_mod_path[0, 2];
    alpha21 = fin_config_mod_path[1, 0]; alpha22 = fin_config_mod_path[1, 1]; alpha23 = fin_config_mod_path[1, 2];
    alpha31 = fin_config_mod_path[2, 0]; alpha32 = fin_config_mod_path[2, 1]; alpha33 = fin_config_mod_path[2, 2];
    
    # Storing the details of the path
    path_params = []
    
    if path_type in path_types:
        
        if path_type == 'lgl':
            
            cphi2 = (alpha11 + r_mod*math.sqrt(1 - r_mod**2)*(alpha13 + alpha31)\
                     + r_mod**2*(alpha33 - alpha11 - 1))/(1 - r_mod**2)
        
        elif path_type == 'lgr':
            
            cphi2 = ((1 - r_mod**2)*alpha11 + r_mod*math.sqrt(1 - r_mod**2)*(alpha31 - alpha13)\
                     + (r_mod**2)*(1 - alpha33))/((1 - r_mod**2))
        
        elif path_type == 'rgr':
        
            cphi2 = (alpha11 - r_mod*math.sqrt(1 - r_mod**2)*(alpha13 + alpha31)\
                     + r_mod**2*(alpha33 - alpha11 - 1))/(1 - r_mod**2)
                
        elif path_type == 'lrl':
            
            cphi2 = ((1 - r_mod**2)*alpha11 + r_mod*sqrt(1 - r_mod**2)*(alpha13 + alpha31)\
                     + r_mod**2*alpha33 - (1 - 2*r_mod**2)**2)/(4*r_mod**2*(1 - r_mod**2))
        
        # Obtaining the two solutions (if they exist) for phi2. Accounting for
        # numerical inaccuracies
        if abs(cphi2) > 1 and abs(cphi2) <= 1 + 10**(-12):
        
            cphi2 = np.sign(cphi2)
            
        # elif abs(cphi2) >= 1 - 10**(-8):
            
        #     cphi2 = np.sign(cphi2)
        
        # We also check if cphi2 is nearly 1 or -1, which means that the middle segment is non-existent
        # or it is exactly pi.
        # This tolerance must be very small since for slightly non-zero G segments, larger tolerance
        # can miss solutions due to the later check on whether the inverse kinematics solution
        # attains the given final configuration
        elif abs(cphi2) >= 1 - 10**(-12):
            
            cphi2 = np.sign(cphi2)
            
        # print('cphi2 value is ', cphi2)

        # Checking if path exists
        if abs(cphi2) > 1: # In this case, the path does not exist
            
            # print(path_type.upper() + ' path does not exist as cphi2 is ' + str(cphi2) + '.')
            path_length = np.NaN; phi1 = np.NaN; phi2 = np.NaN; phi3 = np.NaN;
            path_params.append([path_length, phi1, phi2, phi3])
            
            return path_params
        
        # Checking if the path is a degenerate 'C' if lgl or rgr path
        elif cphi2 == 1 and (path_type in ['lgl', 'rgr', 'lrl', 'rlr']):
            
            # print('Path is of type ' + path_type[0].upper())
            
            # Setting the second and third angles to zero. Third angle is set to zero
            # since the first and third arcs of the same type.
            phi2 = 0
            phi3 = 0
                
            phi1 = np.mod(math.atan2(alpha21, r_mod*alpha22), 2*math.pi)
            # print('Computed phi1 value is ', phi1, 'since final configuration matrix is ', fin_config_mod)
            # We check if phi1 is nearly zero
            if 2*math.pi - phi1 < 10**(-8):
                
                phi1 = 0
                
            # Testing if the final configuration obtained from the C path is the
            # same as the desired final configuration
            # HERE, WE USE THE ORIGINAL PATH TYPE TO CHECK IF WE HAVE ATTAINED THE FINAL CONFIGURATION
            fin_config_path =\
                points_path(np.identity(3), r, 1, [phi1, phi2, phi3], path_type_ini)[3]
                
            # Checking if the minimum and maximum value in the difference in the final
            # configurations is small
            if abs(max(map(max, fin_config_path - fin_config_mod))) <= tol_path\
                and abs(min(map(min, fin_config_path - fin_config_mod))) <= tol_path:
                
                if path_type == 'lgl' or path_type == 'lrl':    
                
                    path_params.append([r*phi1, phi1, phi2, phi3])
                    
                elif path_type == 'rgr' or path_type == 'rlr':
                    
                    path_params.append([r*phi1, phi1, phi2, phi3])                    
               
            else:
                
                # print(path_type.upper() + ' path does not exist.')
                path_params.append([np.NaN, np.NaN, np.NaN, np.NaN])
                               
        else:
            
            # One or two possible solutions exist for phi2
            if abs(cphi2) == 1:
                
                # We check if the considered path is an LRL path. This is because
                # the middle segment cannot be pi for it is to be optimal.
                # The LRpiL path is considered seperately, and is only considered if
                # the turning radius is larger than 1/sqrt(2)
                if path_type == 'lrl':
                    
                    phi2_array = []
                
                else:

                    phi2_array = [math.acos(cphi2)]
                
            else:
                
                if path_type == 'lrl': # We only consider the solution in (pi, 2*pi)
                    
                    phi2_array = [2*math.pi - math.acos(cphi2)]
                
                else:

                    phi2_array = [math.acos(cphi2), 2*math.pi - math.acos(cphi2)]
            
            # print('Solutions for phi2 for path ' + str(path_type) + ' are ' + str(phi2_array))
            
            # Obtaining the possible solutions for phi1 and phi3 corresponding to
            # each phi2
            for phi2 in phi2_array:
                
                # print('phi2 considered is ' + str(phi2))
                # Obtaining possible solutions for phi1 and phi3
                if path_type == 'lgl':
                    
                    phi1RHS = ((alpha33 - alpha11)*r_mod - alpha13*r_mod**2*(1 - r_mod**2)**(-0.5)\
                               + alpha31*math.sqrt(1 - r_mod**2))/math.sqrt(r_mod**2*(1 - cos(phi2))**2 + (sin(phi2))**2)
                        
                    beta = math.atan2(sin(phi2), r_mod*(1 - cos(phi2)))
                    
                    phi3RHS = ((alpha33 - alpha11)*r_mod + alpha13*math.sqrt(1 - r_mod**2)\
                               - alpha31*r_mod**2*(1 - r_mod**2)**(-0.5))/math.sqrt(r_mod**2*(1 - cos(phi2))**2 + (sin(phi2))**2)

                    # print('phi1RHS and phi3RHS are ', phi1RHS, phi3RHS)
                
                elif path_type == 'lgr':
                    
                    if cphi2 == -1: # In this case, wherein the angle of the G
                        # segment is exactly pi, the path can be modified to be a degenerate LG or GR path.
                                                    
                        phi1 = np.mod(math.atan2(alpha12, -r_mod*alpha22), 2*math.pi)
                        # We check if phi1 is nearly zero
                        if 2*math.pi - phi1 <= 10**(-8):
                            phi1 = 0

                        phi3 = 0
                        phi2 = math.pi
                        
                    else:
                        
                        phi1RHS = (r_mod*math.sqrt(1 - r_mod**2)*alpha11 - (1 - r_mod**2)*alpha31\
                                   - (r_mod**2)*alpha13 + r_mod*math.sqrt(1 - r_mod**2)*alpha33)/\
                                  (math.sqrt((r_mod*math.sqrt(1 - r_mod**2)*cos(phi2) + r_mod*math.sqrt(1 - r_mod**2))**2\
                                             + (1 - r_mod**2)*(sin(phi2))**2))
                        
                        # Including a negative sign in addition for beta and gamma, since
                        # beta and gamma are added to the angle corr. to RHS further down,
                        # but analytically, beta and gamma should be subtracted
                        beta = -math.atan2(sin(phi2), (r_mod*(cos(phi2) + 1)))
                        
                        phi3RHS = (r_mod*math.sqrt(1 - r_mod**2)*alpha11 + (1 - r_mod**2)*alpha13\
                                   + (r_mod**2)*alpha31 + r_mod*math.sqrt(1 - r_mod**2)*alpha33)/\
                                  (math.sqrt((r_mod*math.sqrt(1 - r_mod**2)*cos(phi2) + r_mod*math.sqrt(1 - r_mod**2))**2\
                                             + (1 - r_mod**2)*(sin(phi2))**2))
                
                elif path_type == 'rgr':
                
                    phi1RHS = ((alpha33 - alpha11)*r_mod + alpha13*r_mod**2*(1 - r_mod**2)**(-0.5)\
                               - alpha31*math.sqrt(1 - r_mod**2))/math.sqrt(r_mod**2*(1 - cos(phi2))**2 + (sin(phi2))**2)
                        
                    beta = math.atan2(sin(phi2), r_mod*(1 - cos(phi2)))
                    
                    phi3RHS = ((alpha33 - alpha11)*r_mod - alpha13*math.sqrt(1 - r_mod**2)\
                               + alpha31*r_mod**2*(1 - r_mod**2)**(-0.5))/math.sqrt(r_mod**2*(1 - cos(phi2))**2 + (sin(phi2))**2)
                    
                elif path_type == 'lrl':
                    
                    A = (2*r_mod**2 - 1)*(1 - cos(phi2))
                    B = sin(phi2)
                    
                    phi1RHS = ((r_mod**2 - 1)*alpha11 + r_mod*sqrt(1 - r_mod**2)*(alpha31 - alpha13) + r_mod**2*alpha33\
                                - (8*r_mod**6 - 12*r_mod**4 + 6*r_mod**2 - 1 - 4*(2*r_mod**6 - 3*r_mod**4 + r_mod**2)*cos(phi2)))/(4*r_mod**2*(1 - r_mod**2))
                        
                    beta = math.atan2(B, A)
                    
                    phi3RHS = ((r_mod**2 - 1)*alpha11 + r_mod*sqrt(1 - r_mod**2)*(alpha13 - alpha31) + r_mod**2*alpha33\
                                - (8*r_mod**6 - 12*r_mod**4 + 6*r_mod**2 - 1 - 4*(2*r_mod**6 - 3*r_mod**4 + r_mod**2)*cos(phi2)))/(4*r_mod**2*(1 - r_mod**2))
                    
                    phi1RHS /= sqrt(A**2 + B**2)
                    phi3RHS /= sqrt(A**2 + B**2)

                if path_type in ['lgr', 'rgl'] and cphi2 == -1:
                    
                    phi1_array = [phi1]
                    phi3_array = [phi3]
                
                else:
                
                    # Checking if solution for phi1 and phi3 can be obtained withih
                    # a tolerance
                    if abs(phi1RHS) > 1 and abs(phi1RHS) <= 1 + 10**(-8):
            
                        phi1RHS = np.sign(phi1RHS)
                        
                    if abs(phi3RHS) > 1 and abs(phi3RHS) <= 1 + 10**(-8):
            
                        phi3RHS = np.sign(phi3RHS)
                        
                    # Checking condition for phi1 and phi3 cannot be solved for
                    if abs(phi1RHS) > 1:
                        
                        phi1_array = []
                        # phi3_array = []
                        # continue # Skipping the possible solution for phi2 as it is not a viable solution
                        
                    # Checking if one or two solutions exist for phi1
                    elif abs(phi1RHS) == 1:
                        
                        # Only one solution exists for phi1
                        phi1_array = [np.mod(math.acos(phi1RHS) + beta, 2*math.pi)]
                        
                        # Checking if the angle is nearly 2*math.pi
                        if 2*math.pi - phi1_array[0] <= 10**(-8): # or phi1_array[0] <= 10**(-8):
                            
                            phi1_array[0] = 0
                        
                    else:
                        
                        # Obtaining the two possible solutions for phi1
                        phi1_array = [np.mod(math.acos(phi1RHS) + beta, 2*math.pi),\
                                      np.mod(2*math.pi - math.acos(phi1RHS) + beta, 2*math.pi)]
                        
                        # Checking if one of the solutions is nearly 2*math.pi
                        if 2*math.pi - phi1_array[0] <= 10**(-8): # or phi1_array[0] <= 10**(-8):
                            
                            phi1_array[0] = 0
                            
                        if 2*math.pi - phi1_array[1] <= 10**(-8): # or phi1_array[1] <= 10**(-8):
                            
                            phi1_array[1] = 0
                            
                        # # Checking if the two solutions are the same
                        # if abs(phi1_array[0] - phi1_array[1]) <= 10**(-8):
                            
                        #     phi1_array = [np.mod(math.acos(phi1RHS) + beta, 2*math.pi)]
                    
                    # Checking if one or two solutions exist for phi3
                    if abs(phi3RHS) > 1:

                        phi3_array = []

                    elif abs(phi3RHS) == 1:
                        
                        # Only one solution exists for phi1
                        phi3_array = [np.mod(math.acos(phi3RHS) + beta, 2*math.pi)]
                        
                        # # Checking if one of the angles is nearly 2*math.pi
                        if 2*math.pi - phi3_array[0] <= 10**(-8): # or phi3_array[0] <= 10**(-8):
                            
                            phi3_array[0] = 0
                        
                    else:
                        
                        # Obtaining the two possible solutions for phi3
                        phi3_array = [np.mod(math.acos(phi3RHS) + beta, 2*math.pi),\
                                      np.mod(2*math.pi - math.acos(phi3RHS) + beta, 2*math.pi)]
                            
                        # Checking if one of the solutions is nearly 2*math.pi
                        if 2*math.pi - phi3_array[0] <= 10**(-8): # or phi3_array[0] <= 10**(-8):
                            
                            phi3_array[0] = 0
                            
                        if 2*math.pi - phi3_array[1] <= 10**(-8): # or phi3_array[1] <= 10**(-8):
                            
                            phi3_array[1] = 0
                            
                        # # Checking if the two solutions are the same
                        # if abs(phi3_array[0] - phi3_array[1]) <= 10**(-8):
                            
                        #     phi3_array = [np.mod(math.acos(phi3RHS) + gamma, 2*math.pi)]
                
                # print('Solutions for phi 1 are ' + str(phi1_array))
                # print('Solutions for phi 3 are ' + str(phi3_array))
                
                # From all possible solutions for phi1 and phi3 for the chosen phi2,
                # identifying those solutions that connect to the chosen final configuration
                for phi1 in phi1_array:
                    
                    for phi3 in phi3_array:
                        
                        # Obtaining the final configuration of the path
                        # HERE, WE USE THE INITIAL PATH TYPE AND THE ACTUAL FINAL CONFIGURATION WITHOUT REFLECTION TO CHECK
                        fin_config_path =\
                            points_path(np.identity(3), r_mod, 1, [phi1, phi2, phi3], path_type_ini)[3]
                        
                        # print(fin_config_path)
                            
                        # Checking if the minimum and maximum value in the difference in the final
                        # configurations is small
                        if abs(max(map(max, fin_config_path - fin_config_mod))) <= tol_path\
                            and abs(min(map(min, fin_config_path - fin_config_mod))) <= tol_path:
                            
                            if path_type == 'lgl':    
                            
                                # Appending the path length and path parameters
                                path_length = r*(phi1 + phi3) + R*phi2
                                
                            elif path_type == 'lgr':
                                
                                # Appending the path length and path parameters
                                path_length = r*(phi1 + phi3) + R*phi2
                                
                            elif path_type == 'rgr':
                                
                                # Appending the path length and path parameters
                                path_length = r*(phi1 + phi3) + R*phi2
                                
                            elif path_type == 'lrl':
                                
                                # Appending the path length and path parameters
                                path_length = r*(phi1 + phi3 + phi2)
                                
                            path_params.append([path_length, phi1, phi2, phi3])
                            
            # print('Parameters of the ' + str(path_type) + ' path.')
            # print(path_params)
                            # print('Solution for phi2 number ' + str(phi2_array.index(phi2)) +\
                            #       ', solution for phi1 number ' + str(phi1_array.index(phi1)) +\
                            #       ', solution for phi3 number ' + str(phi3_array.index(phi3)))
                            
            # Checking if no solution was obtained for the path
            if len(path_params) == 0:
                
                # print(path_type.upper() + ' path does not exist.')
                path_params.append([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
                
    return path_params

def path_generation_C_Cpi_C(ini_config, fin_config, r, R, path_type = 'lrl', tol_path = 10**(-4)):
    '''
    In this function, an LRpiL path or RLpiR path is constructed to connect the chosen
    initial and final configurations, if it exists.

    Parameters
    ----------
    ini_config : Numpy array
        Contains the initial configuration. The syntax followed is as
        follows:
            The first column contains the position.
            The second column contains the tangent vector.
            The third column contains the tangent-normal vector.
    fin_config : Numpy array
        Contains the final configuration. The same syntax used for ini_config is
        used here.
    r : Scalar
        Radius of the tight turn.
    R : Scalar
        Radius of the sphere.
    path_type : String, optional
        Defines the type of path considered. The default is 'lrl'.

    Returns
    -------
    path_params : Numpy array
        Contains the parameters associated with the generated path. Note that multiple paths
        of the same type can be possibly generated. For each path, the path length, phi1, phi2,
        phi3 are included as an array, where phii denotes the arc angle of the ith segment.

    '''

    # Modifying the configurations and the parameters of the turn
    fin_config_mod = modifying_initial_final_configurations_unit_sphere(ini_config, fin_config, R)
    r_mod = r/R
    
    path_types = ['lrl', 'rlr']
    
    if path_type not in path_types:
        
        raise Exception('Incorrect path type passed.')
    
    # If the path considered is an RLR path, we reflect the final configuration about the XY plane
    # and construct an LRL path
    if path_type == 'rlr':

        fin_config_mod_path = copy.deepcopy(fin_config_mod)
        # We reflect the final location and tangent vector about the XY plane
        for i in range(3):
            fin_config_mod_path[2, i] = -fin_config_mod[2, i]
        # We reflect the tangent-normal about the XY plane, but also reverse it
        # after to ensure that we have a rotation matrix
        for i in range(3):
            fin_config_mod_path[i, 2] = -fin_config_mod_path[i, 2]

    else:

        fin_config_mod_path = fin_config_mod
        
    # We now construct an LRL path since we have switched the configurations for the RLR path  
    # Defining variables corresponding to the final configuration
    alpha11 = fin_config_mod_path[0, 0]; alpha12 = fin_config_mod_path[0, 1]; alpha13 = fin_config_mod_path[0, 2];
    alpha21 = fin_config_mod_path[1, 0]; alpha22 = fin_config_mod_path[1, 1]; alpha23 = fin_config_mod_path[1, 2];
    alpha31 = fin_config_mod_path[2, 0]; alpha32 = fin_config_mod_path[2, 1]; alpha33 = fin_config_mod_path[2, 2];
    
    # Storing the details of the path
    path_params = []

    # We now obtain the solutions for phi1 and phi3
    if abs(r_mod - 1/sqrt(2)) > 10**(-6): # Tolerance adjusted based on unit tests

        # In this case, we can solve for phi1 and phi3
        phi1RHS = (1/(8*(r_mod**2 - 1)*r_mod**2))*(1 - 8*r_mod**2 + 8*r_mod**4\
                                                    + (1/(1 - 2*r_mod**2))*(alpha11*(r_mod**2 - 1)\
                                                                             + r_mod*((alpha31 - alpha13)*sqrt(1 - r_mod**2) + alpha33*r_mod)))
        phi3RHS = (1/(8*(r_mod**2 - 1)*r_mod**2))*(1 - 8*r_mod**2 + 8*r_mod**4\
                                                    + (1/(1 - 2*r_mod**2))*(alpha11*(r_mod**2 - 1)\
                                                                             + r_mod*((alpha13 - alpha31)*sqrt(1 - r_mod**2) + alpha33*r_mod)))
        
        # Checking if solution for phi1 and phi3 can be obtained withih
        # a tolerance
        if abs(phi1RHS) > 1 and abs(phi1RHS) <= 1 + 10**(-8):

            phi1RHS = np.sign(phi1RHS)
            
        if abs(phi3RHS) > 1 and abs(phi3RHS) <= 1 + 10**(-8):

            phi3RHS = np.sign(phi3RHS)
        
        # print('RHS for phi1 solution is ', phi1RHS)

        # Checking condition for phi1 and phi3 cannot be solved for
        if abs(phi1RHS) > 1:
            
            phi1_array = []
            
        # Checking if one or two solutions exist for phi1
        elif abs(phi1RHS) == 1:
            
            # Only one solution exists for phi1
            phi1_array = [np.mod(math.acos(phi1RHS), 2*math.pi)]
            
            # Checking if the angle is nearly 2*math.pi
            if 2*math.pi - phi1_array[0] <= 10**(-8):
                
                phi1_array[0] = 0
            
        else:
            
            # Obtaining the two possible solutions for phi1
            phi1_array = [np.mod(math.acos(phi1RHS), 2*math.pi),\
                            np.mod(2*math.pi - math.acos(phi1RHS), 2*math.pi)]
            
            # Checking if one of the solutions is nearly 2*math.pi
            if 2*math.pi - phi1_array[0] <= 10**(-8):
                
                phi1_array[0] = 0
                
            if 2*math.pi - phi1_array[1] <= 10**(-8):
                
                phi1_array[1] = 0
        
        # Checking if one or two solutions exist for phi3
        if abs(phi3RHS) > 1:

            phi3_array = []

        elif abs(phi3RHS) == 1:
            
            # Only one solution exists for phi1
            phi3_array = [np.mod(math.acos(phi3RHS), 2*math.pi)]
            
            # # Checking if one of the angles is nearly 2*math.pi
            if 2*math.pi - phi3_array[0] <= 10**(-8):
                
                phi3_array[0] = 0
            
        else:
            
            # Obtaining the two possible solutions for phi3
            phi3_array = [np.mod(math.acos(phi3RHS), 2*math.pi),\
                            np.mod(2*math.pi - math.acos(phi3RHS), 2*math.pi)]
                
            # Checking if one of the solutions is nearly 2*math.pi
            if 2*math.pi - phi3_array[0] <= 10**(-8):
                
                phi3_array[0] = 0
                
            if 2*math.pi - phi3_array[1] <= 10**(-8):
                
                phi3_array[1] = 0

        # print('Solutions for phi 1 are ' + str(phi1_array))
        # print('Solutions for phi 3 are ' + str(phi3_array))

        # We now run through the obtained solutions for phi1 and phi3
        for phi1 in phi1_array:
                    
            for phi3 in phi3_array:
                
                # Obtaining the final configuration of the path
                fin_config_path_construct =\
                    points_path(np.identity(3), r_mod, 1, [phi1, math.pi, phi3], path_type)[3]
                
                # print(fin_config_path_construct)
                    
                # Checking if the minimum and maximum value in the difference in the final
                # configurations is small
                # WE CHECK WITH THE UNCHANGED FINAL CONFIGURATION IN THIS CASE
                if abs(max(map(max, fin_config_path_construct - fin_config_mod))) <= tol_path\
                    and abs(min(map(min, fin_config_path_construct - fin_config_mod))) <= tol_path:
                    
                    path_params.append([r*(phi1 + math.pi + phi3), phi1, math.pi, phi3])
                            
        # print('Parameters of the ' + str(path_type) + ' path.')
        # print(path_params)
                        # print('Solution for phi2 number ' + str(phi2_array.index(phi2)) +\
                        #       ', solution for phi1 number ' + str(phi1_array.index(phi1)) +\
                        #       ', solution for phi3 number ' + str(phi3_array.index(phi3)))
                        
        # Checking if no solution was obtained for the path
        if len(path_params) == 0:
            
            # print(path_type.upper() + ' path does not exist.')
            path_params.append([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])

    else:

        # In this case, the turning radius is 1/sqrt(2), which is a special case.
        # We first obtain the expression for phi1 - phi3
        phi1phi3_diff = math.atan2(sqrt(2)*alpha21, -alpha22)

        if phi1phi3_diff < 0:

            phi1 = 0; phi3 = np.mod(-phi1phi3_diff, 2*math.pi)

        else:

            phi1 = np.mod(phi1phi3_diff, 2*math.pi); phi3 = 0

        # We check if the final configuration is attained
        fin_config_path_construct =\
                points_path(np.identity(3), r_mod, 1, [phi1, math.pi, phi3], path_type)[3]
            
        # print(fin_config_path_construct)
            
        # Checking if the minimum and maximum value in the difference in the final
        # configurations is small
        if abs(max(map(max, fin_config_path_construct - fin_config_mod))) <= tol_path\
            and abs(min(map(min, fin_config_path_construct - fin_config_mod))) <= tol_path:
                
            path_params.append([r*(phi1 + math.pi + phi3), phi1, math.pi, phi3])

        else:

            path_params.append([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])

    return path_params

def path_generation_CCCC(ini_config, fin_config, r, R, path_type = 'lrlr', tol_path = 10**(-4)):
    '''
    In this function, the chosen path of type LRLR or RLRL to connect the chosen initial and final configurations
    are constructed, if they exist. Note that such paths can be optimal for r/R > 1/2.
    
    Parameters
    ----------
    ini_config : Numpy array
        Contains the initial configuration. The syntax followed is as
        follows:
            The first column contains the position.
            The second column contains the tangent vector.
            The third column contains the tangent-normal vector.
    fin_config : Numpy array
        Contains the final configuration. The same syntax used for ini_config is
        used here.
    r : Scalar
        Radius of the tight turn.
    R : Scalar
        Radius of the sphere.
    path_type : String, optional
        Defines the type of path considered. The default is 'lrlr'.

    Returns
    -------
    path_params : Numpy array
        Contains the parameters associated with the generated path. Note that multiple paths
        of the same type can be possibly generated. For each path, the path length, phi1, phi2, phi2,
        phi3 are included as an array, where phii denotes the arc angle of the ith segment.
        Note that in this case, the angle of the middle two segments are the same and equal to phi2.

    '''

    # Modifying the configurations and the parameters of the turn
    fin_config_mod = modifying_initial_final_configurations_unit_sphere(ini_config, fin_config, R)
    r_mod = r/R
    
    path_types = ['lrlr', 'rlrl']
    
    if path_type not in path_types:
        
        raise Exception('Incorrect path type passed.')
    
    # If the path considered is an RLRL path, we reflect the final configuration about the XY plane
    # and construct an LRLR path
    if path_type == 'rlrl':

        fin_config_mod_path = copy.deepcopy(fin_config_mod)
        # We reflect the final location and tangent vector about the XY plane
        for i in range(3):
            fin_config_mod_path[2, i] = -fin_config_mod[2, i]
        # We reflect the tangent-normal about the XY plane, but also reverse it
        # after to ensure that we have a rotation matrix
        for i in range(3):
            fin_config_mod_path[i, 2] = -fin_config_mod_path[i, 2]

    else:

        fin_config_mod_path = fin_config_mod

    # We now construct an LRL path since we have switched the configurations for the RLR path  
    # Defining variables corresponding to the final configuration
    alpha11 = fin_config_mod_path[0, 0]; alpha12 = fin_config_mod_path[0, 1]; alpha13 = fin_config_mod_path[0, 2];
    alpha21 = fin_config_mod_path[1, 0]; alpha22 = fin_config_mod_path[1, 1]; alpha23 = fin_config_mod_path[1, 2];
    alpha31 = fin_config_mod_path[2, 0]; alpha32 = fin_config_mod_path[2, 1]; alpha33 = fin_config_mod_path[2, 2];
    
    # Storing the details of the path
    path_params = []

    # Now, we construct an LRLR path, wherein the angle of the middle segments are equal and greater than pi.
    # First, we obtain the solutions for phi2.
    # To this end, we solve a quadratic equation
    a = 8*r_mod**4*(r_mod**2 - 1); b = -8*(r_mod**2 - 3*r_mod**4 + 2*r_mod**6)
    c = -1 + 10*r_mod**2 - 16*r_mod**4 + 8*r_mod**6 - (alpha11*(r_mod**2 - 1) + r_mod*(sqrt(1 - r_mod**2)*(alpha13 - alpha31) + alpha33*r_mod))

    disc = b**2 - 4*a*c
    if -10**(12) <= disc <= 0: disc = 0.0

    if disc > 0:

        cphi2soln = [(-b - sqrt(disc))/(2*a), (-b + sqrt(disc))/(2*a)]

    elif disc == 0:

        cphi2soln = [-b/(2*a)]

    else: # We cannot obtain a solution for phi2

        return np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
    
    # We run through each solution
    phi2_arr = []
    for cphi2 in cphi2soln:

        if abs(cphi2) > 1 and abs(cphi2) <= 1 + 10**(-12):
        
            cphi2 = np.sign(cphi2)

        elif abs(cphi2) >= 1 - 10**(-12):
            
            cphi2 = np.sign(cphi2)

        # We check if the path exists
        if abs(cphi2) < 1: # We do not consider the case of cphi2 = +-1.

            # We pick the solution such that it is greater than pi
            phi2_arr.append(2*math.pi - math.acos(cphi2))

    # We run through all possible solutions for phi2
    for phi2 in phi2_arr:

        # We check if we can solve for phi1 and phi3
        if abs(math.cos(phi2) - (1 - 1/(2*r_mod**2))) <= 10**(-6): # Check the tolerance

            # In this case, we set phi3 to zero and solve for phi
            cosphi1 = (1/r_mod**2)*((sqrt(4*r_mod**2 - 1)/(2*r_mod**2))*alpha12 - ((2*r_mod**2 - 1)/(2*r_mod))*alpha22)
            sinphi1 = (1/r_mod**2)*(((2*r_mod**2 - 1)/(2*r_mod**2))*alpha12 + (sqrt(4*r_mod**2 - 1)/(2*r_mod))*alpha22)

            # We check that sinphi1**2 + cosphi1**2 = 1
            if abs(sinphi1**2 + cosphi1**2 - 1) <= 10**(-12):

                # We now solve for phi1
                phi1 = np.mod(math.atan2(sinphi1, cosphi1), 2*math.pi)
                phi3 = 0

                # We check if the final configuration is attained
                fin_config_path_construct = points_path(np.identity(3), r_mod, 1, [phi1, phi2, phi2, phi3], path_type)[3]

                if abs(max(map(max, fin_config_path_construct - fin_config_mod))) <= tol_path\
                    and abs(min(map(min, fin_config_path_construct - fin_config_mod))) <= tol_path:
                        
                    path_params.append([r*(phi1 + 2*phi2 + phi3), phi1, phi2, phi2, phi3])

        else: # In this case, we obtain the solutions for phi1 and phi3

            A = 4*r_mod**2*(1 - r_mod**2)*(2*r_mod**2*cos(phi2) - 2*r_mod**2 + 1)*((2*r_mod**2 - 1)*cos(phi2) - 2*r_mod**2 + 2)
            B = 4*r_mod**2*(1 - r_mod**2)*(2*r_mod**2*cos(phi2) - 2*r_mod**2 + 1)*(-sin(phi2))
            C = (2*r_mod**2 - 1)*(12*r_mod**6 - 20*r_mod**4 + 10*r_mod**2 + 4*(r_mod**2 - 1)*r_mod**4*cos(2*phi2) - 8*(2*r_mod**6 - 3*r_mod**4 + r_mod**2)*cos(phi2) - 1)

            phi1RHS = (alpha11*(1 - r_mod**2) + r_mod*(-sqrt(1 - r_mod**2)*(alpha13 + alpha31) + alpha33*r_mod) - C)/sqrt(A**2 + B**2)
            phi3RHS = (alpha11*(1 - r_mod**2) + r_mod*(sqrt(1 - r_mod**2)*(alpha13 + alpha31) + alpha33*r_mod) - C)/sqrt(A**2 + B**2)

            beta = math.atan2(B, A)

            # Checking if solution for phi1 and phi3 can be obtained withih
            # a tolerance
            if abs(phi1RHS) > 1 and abs(phi1RHS) <= 1 + 10**(-8):

                phi1RHS = np.sign(phi1RHS)
                
            if abs(phi3RHS) > 1 and abs(phi3RHS) <= 1 + 10**(-8):

                phi3RHS = np.sign(phi3RHS)
            
            # print('RHS for phi1 solution is ', phi1RHS)

            # Checking condition for phi1 and phi3 cannot be solved for
            if abs(phi1RHS) > 1:
                
                phi1_array = []
                
            # Checking if one or two solutions exist for phi1
            elif abs(phi1RHS) == 1:
                
                # Only one solution exists for phi1
                phi1_array = [np.mod(math.acos(phi1RHS) + beta, 2*math.pi)]
                
                # Checking if the angle is nearly 2*math.pi
                if 2*math.pi - phi1_array[0] <= 10**(-8):
                    
                    phi1_array[0] = 0
                
            else:
                
                # Obtaining the two possible solutions for phi1
                phi1_array = [np.mod(math.acos(phi1RHS) + beta, 2*math.pi),\
                                np.mod(2*math.pi - math.acos(phi1RHS) + beta, 2*math.pi)]
                
                # Checking if one of the solutions is nearly 2*math.pi
                if 2*math.pi - phi1_array[0] <= 10**(-8):
                    
                    phi1_array[0] = 0
                    
                if 2*math.pi - phi1_array[1] <= 10**(-8):
                    
                    phi1_array[1] = 0
            
            # Checking if one or two solutions exist for phi3
            if abs(phi3RHS) > 1:

                phi3_array = []

            elif abs(phi3RHS) == 1:
                
                # Only one solution exists for phi1
                phi3_array = [np.mod(math.acos(phi3RHS) + beta, 2*math.pi)]
                
                # # Checking if one of the angles is nearly 2*math.pi
                if 2*math.pi - phi3_array[0] <= 10**(-8):
                    
                    phi3_array[0] = 0
                
            else:
                
                # Obtaining the two possible solutions for phi3
                phi3_array = [np.mod(math.acos(phi3RHS) + beta, 2*math.pi),\
                                np.mod(2*math.pi - math.acos(phi3RHS) + beta, 2*math.pi)]
                    
                # Checking if one of the solutions is nearly 2*math.pi
                if 2*math.pi - phi3_array[0] <= 10**(-8):
                    
                    phi3_array[0] = 0
                    
                if 2*math.pi - phi3_array[1] <= 10**(-8):
                    
                    phi3_array[1] = 0

            # print('Solutions for phi 1 are ' + str(phi1_array))
            # print('Solutions for phi 3 are ' + str(phi3_array))

            # We now run through the obtained solutions for phi1 and phi3
            for phi1 in phi1_array:
                        
                for phi3 in phi3_array:

                    # We check if the final configuration is attained
                    fin_config_path_construct = points_path(np.identity(3), r_mod, 1, [phi1, phi2, phi2, phi3], path_type)[3]

                    if abs(max(map(max, fin_config_path_construct - fin_config_mod))) <= tol_path\
                        and abs(min(map(min, fin_config_path_construct - fin_config_mod))) <= tol_path:
                            
                        path_params.append([r*(phi1 + 2*phi2 + phi3), phi1, phi2, phi2, phi3])

    # We check if a solution was obtained
    if len(path_params) == 0:
            
        # print(path_type.upper() + ' path does not exist.')
        path_params.append([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])

    return path_params

def path_generation_CCCCC(ini_config, fin_config, r, R, path_type = 'lrlrl', tol_path = 10**(-4)):
    '''
    In this function, the chosen path of type LRLRL or RLRLR to connect the chosen initial and final configurations
    are constructed, if they exist. Note that such paths can be optimal for r/R > 1/sqrt(2).
    
    Parameters
    ----------
    ini_config : Numpy array
        Contains the initial configuration. The syntax followed is as
        follows:
            The first column contains the position.
            The second column contains the tangent vector.
            The third column contains the tangent-normal vector.
    fin_config : Numpy array
        Contains the final configuration. The same syntax used for ini_config is
        used here.
    r : Scalar
        Radius of the tight turn.
    R : Scalar
        Radius of the sphere.
    path_type : String, optional
        Defines the type of path considered. The default is 'lrlrl'.

    Returns
    -------
    path_params : Numpy array
        Contains the parameters associated with the generated path. Note that multiple paths
        of the same type can be possibly generated. For each path, the path length, phi1, phi2, phi2, phi2,
        phi3 are included as an array, where phii denotes the arc angle of the ith segment.
        Note that in this case, the angle of the middle three segments are the same and equal to phi2.

    '''

    # Modifying the configurations and the parameters of the turn
    fin_config_mod = modifying_initial_final_configurations_unit_sphere(ini_config, fin_config, R)
    r_mod = r/R
    
    path_types = ['lrlrl', 'rlrlr']
    
    if path_type not in path_types:
        
        raise Exception('Incorrect path type passed.')
    
    # If the path considered is an RLRLR path, we reflect the final configuration about the XY plane
    # and construct an LRLRL path
    if path_type == 'rlrlr':

        fin_config_mod_path = copy.deepcopy(fin_config_mod)
        # We reflect the final location and tangent vector about the XY plane
        for i in range(3):
            fin_config_mod_path[2, i] = -fin_config_mod[2, i]
        # We reflect the tangent-normal about the XY plane, but also reverse it
        # after to ensure that we have a rotation matrix
        for i in range(3):
            fin_config_mod_path[i, 2] = -fin_config_mod_path[i, 2]

    else:

        fin_config_mod_path = fin_config_mod

    # We now construct an LRL path since we have switched the configurations for the RLR path  
    # Defining variables corresponding to the final configuration
    alpha11 = fin_config_mod_path[0, 0]; alpha12 = fin_config_mod_path[0, 1]; alpha13 = fin_config_mod_path[0, 2];
    alpha21 = fin_config_mod_path[1, 0]; alpha22 = fin_config_mod_path[1, 1]; alpha23 = fin_config_mod_path[1, 2];
    alpha31 = fin_config_mod_path[2, 0]; alpha32 = fin_config_mod_path[2, 1]; alpha33 = fin_config_mod_path[2, 2];
    
    # Storing the details of the path
    path_params = []

    # Now, we construct an LRLRL path, wherein the angle of the middle segments are equal and greater than pi.
    # First, we obtain the solutions for phi2.
    # To this end, we solve a cubic equation
    a = 16*r_mod**6*(1 - r_mod**2); b = 16*r_mod**4*(2 - 5*r_mod**2 + 3*r_mod**4)
    c = -16*r_mod**2*(1 - r_mod**2)**2*(3*r_mod**2 - 1)
    d = 16*r_mod**8 - 48*r_mod**6 + 48*r_mod**4 - 16*r_mod**2 + 1 - (alpha11*(1 - r_mod**2) + r_mod*(sqrt(1 - r_mod**2)*(alpha13 + alpha31) + alpha33*r_mod))

    # We solve the cubic equation
    solns = solve(a, b, c, d)

    cphi2soln = []
    for soln in solns:

        # We check if the obtained solution is real
        if soln.imag <= 10**(10):

            cphi2soln.append(soln.real)

    if len(cphi2soln) == 0: # We cannot obtain a solution for phi2

        return np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
    
    # We run through each solution
    phi2_arr = []
    for cphi2 in cphi2soln:

        if abs(cphi2) > 1 and abs(cphi2) <= 1 + 10**(-12):
        
            cphi2 = np.sign(cphi2)

        elif abs(cphi2) >= 1 - 10**(-12):
            
            cphi2 = np.sign(cphi2)

        # We check if the path exists
        if abs(cphi2) < 1: # We do not consider the case of cphi2 = +-1.

            # We pick the solution such that it is greater than pi
            phi2_arr.append(2*math.pi - math.acos(cphi2))

    # We run through all possible solutions for phi2
    for phi2 in phi2_arr:

        # We check if we can solve for phi1 and phi3
        if abs(math.cos(phi2) - (1 - 1/(r_mod**2))) <= 10**(-6): # Check the tolerance

            # In this case, we set phi3 to zero and solve for phi
            cosphi1 = -(1/r_mod**4)*(sqrt(2*r_mod**2 - 1)*r_mod*alpha12 - r_mod**2*(r_mod**2 - 1)*alpha22)
            sinphi1 = -(1/r_mod**4)*((r_mod**2 - 1)*r_mod*alpha12 + r_mod**2*sqrt(2*r_mod**2 - 1)*alpha22)

            # We check that sinphi1**2 + cosphi1**2 = 1
            if abs(sinphi1**2 + cosphi1**2 - 1) <= 10**(-12):

                # We now solve for phi1
                phi1 = np.mod(math.atan2(sinphi1, cosphi1), 2*math.pi)
                phi3 = 0

                # We check if the final configuration is attained
                fin_config_path_construct = points_path(np.identity(3), r_mod, 1, [phi1, phi2, phi2, phi2, phi3], path_type)[3]

                if abs(max(map(max, fin_config_path_construct - fin_config_mod))) <= tol_path\
                    and abs(min(map(min, fin_config_path_construct - fin_config_mod))) <= tol_path:
                        
                    path_params.append([r*(phi1 + 3*phi2 + phi3), phi1, phi2, phi2, phi2, phi3])

        else: # In this case, we obtain the solutions for phi1 and phi3

            A = 16*r_mod**2*(r_mod**2 - 1)*(sin(phi2/2))**2*(-6*r_mod**6 + 11*r_mod**4 - 7*r_mod**2 + (r_mod**4 - 2*r_mod**6)*cos(2*phi2) + (8*r_mod**4 - 12*r_mod**2 + 3)*r_mod**2*cos(phi2) + 1)
            B = 8*r_mod**2*(1 - r_mod**2)*sin(phi2)*(r_mod**4*cos(2*phi2) + 3*r_mod**4 - 3*r_mod**2 + (3*r_mod**2 - 4*r_mod**4)*cos(phi2) + 1)
            C = (1 - 2*r_mod**2)*(4*r_mod**8*cos(3*phi2) - 40*r_mod**8 - 4*r_mod**6*cos(3*phi2) + 88*r_mod**6 - 64*r_mod**4 + 16*r_mod**2 - 8*(3*r_mod**4 - 5*r_mod**2 + 2)*r_mod**4*cos(2*phi2)\
                                  + 4*(15*r_mod**6 - 31*r_mod**4 + 20*r_mod**2 - 4)*r_mod**2*cos(phi2) - 1)

            phi1RHS = (alpha11*(r_mod**2 - 1) + r_mod*(sqrt(1 - r_mod**2)*(alpha31 - alpha13) + alpha33*r_mod) - C)/sqrt(A**2 + B**2)
            phi3RHS = (alpha11*(r_mod**2 - 1) + r_mod*(sqrt(1 - r_mod**2)*(alpha13 - alpha31) + alpha33*r_mod) - C)/sqrt(A**2 + B**2)

            beta = math.atan2(B, A)

            # Checking if solution for phi1 and phi3 can be obtained withih
            # a tolerance
            if abs(phi1RHS) > 1 and abs(phi1RHS) <= 1 + 10**(-8):

                phi1RHS = np.sign(phi1RHS)
                
            if abs(phi3RHS) > 1 and abs(phi3RHS) <= 1 + 10**(-8):

                phi3RHS = np.sign(phi3RHS)
            
            # print('RHS for phi1 solution is ', phi1RHS)

            # Checking condition for phi1 and phi3 cannot be solved for
            if abs(phi1RHS) > 1:
                
                phi1_array = []
                
            # Checking if one or two solutions exist for phi1
            elif abs(phi1RHS) == 1:
                
                # Only one solution exists for phi1
                phi1_array = [np.mod(math.acos(phi1RHS) + beta, 2*math.pi)]
                
                # Checking if the angle is nearly 2*math.pi
                if 2*math.pi - phi1_array[0] <= 10**(-8):
                    
                    phi1_array[0] = 0
                
            else:
                
                # Obtaining the two possible solutions for phi1
                phi1_array = [np.mod(math.acos(phi1RHS) + beta, 2*math.pi),\
                                np.mod(2*math.pi - math.acos(phi1RHS) + beta, 2*math.pi)]
                
                # Checking if one of the solutions is nearly 2*math.pi
                if 2*math.pi - phi1_array[0] <= 10**(-8):
                    
                    phi1_array[0] = 0
                    
                if 2*math.pi - phi1_array[1] <= 10**(-8):
                    
                    phi1_array[1] = 0
            
            # Checking if one or two solutions exist for phi3
            if abs(phi3RHS) > 1:

                phi3_array = []

            elif abs(phi3RHS) == 1:
                
                # Only one solution exists for phi1
                phi3_array = [np.mod(math.acos(phi3RHS) + beta, 2*math.pi)]
                
                # # Checking if one of the angles is nearly 2*math.pi
                if 2*math.pi - phi3_array[0] <= 10**(-8):
                    
                    phi3_array[0] = 0
                
            else:
                
                # Obtaining the two possible solutions for phi3
                phi3_array = [np.mod(math.acos(phi3RHS) + beta, 2*math.pi),\
                                np.mod(2*math.pi - math.acos(phi3RHS) + beta, 2*math.pi)]
                    
                # Checking if one of the solutions is nearly 2*math.pi
                if 2*math.pi - phi3_array[0] <= 10**(-8):
                    
                    phi3_array[0] = 0
                    
                if 2*math.pi - phi3_array[1] <= 10**(-8):
                    
                    phi3_array[1] = 0

            # print('Solutions for phi 1 are ' + str(phi1_array))
            # print('Solutions for phi 3 are ' + str(phi3_array))

            # We now run through the obtained solutions for phi1 and phi3
            for phi1 in phi1_array:
                        
                for phi3 in phi3_array:

                    # We check if the final configuration is attained
                    fin_config_path_construct = points_path(np.identity(3), r_mod, 1, [phi1, phi2, phi2, phi2, phi3], path_type)[3]

                    if abs(max(map(max, fin_config_path_construct - fin_config_mod))) <= tol_path\
                        and abs(min(map(min, fin_config_path_construct - fin_config_mod))) <= tol_path:
                            
                        path_params.append([r*(phi1 + 3*phi2 + phi3), phi1, phi2, phi2, phi2, phi3])

    # We check if a solution was obtained
    if len(path_params) == 0:
            
        # print(path_type.upper() + ' path does not exist.')
        path_params.append([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])

    return path_params

def optimal_path_sphere(ini_config, fin_config, r, R, visualization = 1, filename = 'paths_sphere.html'):
    '''
    In this function, the optimal three-segment path, i.e., of least length, is returned.
    Furthermore, it is also visualized.
    
    Parameters
    ----------
    ini_config : Numpy array
        Contains the initial configuration. The syntax followed is as
        follows:
            The first column contains the position.
            The second column contains the tangent vector.
            The third column contains the tangent-normal vector.
    fin_config : Numpy array
        Contains the final configuration. The same syntax used for ini_config is
        used here.
    r : Scalar
        Radius of the tight turn and tight turn.
    R : Scalar
        Radius of the sphere.
    visualization : Scalar, optional
        If equal to 1, the paths are visualized and saved in the provided filename (html file).
        If not, the paths are not visualized.
    filename : String, optional
        Defines the name of the file in which the sphere segments ought to be visualized in.

    Returns
    -------
    least_cost_path : String
        Provides the least cost path type.
    least_cost_path_length : String
        Provides the cost of the optimal path.
    least_cost_path_params: Array
        Contains the angles of the three segments of the path.
    x_coords, y_coords, z_coords, Tx, Ty, Tz: Arrays
        Contains the coordinations of the path and the direction cosines of the tangent vector along the path.
    '''
    
    # Path types
    path_types_three_seg = np.array(['lgl', 'rgr', 'lgr', 'rgl', 'lrl', 'rlr'])
    # path_types_three_seg = np.array(['lrl', 'rlr'])

    # Additional path types, which are considered depending on the turning radius
    if r/R > 1/sqrt(2):
        path_types_three_seg_abnormal = np.array(['lrl', 'rlr'])
        path_types_five_seg = np.array(['lrlrl', 'rlrlr'])
    else:
        path_types_three_seg_abnormal = np.array([])
        path_types_five_seg = np.array([])

    if r/R > 1/2:
        path_types_four_seg = np.array(['lrlr', 'rlrl'])
    else:
        path_types_four_seg = np.array([])
    # path_types_four_seg = np.array([])
    
    least_cost_path = 'lgl'
    least_cost_path_length = np.infty
    least_cost_path_params = []

    # We also record the parameters of all possible paths
    possible_path_types = []
    possible_path_params = []
    
    # Generating the paths
    for path in path_types_three_seg:
        
        path_params = path_generation_sphere_three_seg(ini_config, fin_config, r, R, path)
        
        # Checking if the considered path exists
        for possible_path in path_params:
            
            if ~np.isnan(possible_path[0]): # Checking if path exists

                # We append the possible path
                possible_path_types.append(path)
                possible_path_params.append(possible_path)
            
                # Updating the minimum cost path
                if possible_path[0] < least_cost_path_length:
                    
                    least_cost_path = path
                    least_cost_path_length = possible_path[0]
                    least_cost_path_params = possible_path[1:]

    # Generating the additional paths
    for path in path_types_three_seg_abnormal:
        
        path_params = path_generation_C_Cpi_C(ini_config, fin_config, r, R, path)
        
        # Checking if the considered path exists
        for possible_path in path_params:
            
            if ~np.isnan(possible_path[0]): # Checking if path exists

                # We append the possible path
                possible_path_types.append(path)
                possible_path_params.append(possible_path)
            
                # Updating the minimum cost path
                if possible_path[0] < least_cost_path_length:
                    
                    least_cost_path = path
                    least_cost_path_length = possible_path[0]
                    least_cost_path_params = possible_path[1:]

    for path in path_types_four_seg:
        
        path_params = path_generation_CCCC(ini_config, fin_config, r, R, path)
        
        # Checking if the considered path exists
        for possible_path in path_params:
            
            if ~np.isnan(possible_path[0]): # Checking if path exists

                # We append the possible path
                possible_path_types.append(path)
                possible_path_params.append(possible_path)
            
                # Updating the minimum cost path
                if possible_path[0] < least_cost_path_length:
                    
                    least_cost_path = path
                    least_cost_path_length = possible_path[0]
                    least_cost_path_params = possible_path[1:]

    for path in path_types_five_seg:
        
        path_params = path_generation_CCCCC(ini_config, fin_config, r, R, path)
        
        # Checking if the considered path exists
        for possible_path in path_params:
            
            if ~np.isnan(possible_path[0]): # Checking if path exists

                # We append the possible path
                possible_path_types.append(path)
                possible_path_params.append(possible_path)
            
                # Updating the minimum cost path
                if possible_path[0] < least_cost_path_length:
                    
                    least_cost_path = path
                    least_cost_path_length = possible_path[0]
                    least_cost_path_params = possible_path[1:]

    # Obtaining the coordinates of the path
    x_coords_path, y_coords_path, z_coords_path, _, _, _, _, Tx, Ty, Tz =\
        points_path(ini_config, r, R, least_cost_path_params, least_cost_path)
                    
    # Plotting the optimal path
    if visualization == 1:
        
        fig_3D = plotting_functions()
        # Plotting the sphere
        fig_3D.surface_3D(generate_points_sphere(np.array([0, 0, 0]), R)[0],\
                          generate_points_sphere(np.array([0, 0, 0]), R)[1],\
                          generate_points_sphere(np.array([0, 0, 0]), R)[2], 'grey',\
                          'Sphere', 0.7)
        # Plotting the initial and final configurations
        fig_3D.points_3D([ini_config[0, 0]], [ini_config[1, 0]], [ini_config[2, 0]], 'red',\
                         'Initial point', 'circle')
        fig_3D.points_3D([fin_config[0, 0]], [fin_config[1, 0]],\
                         [fin_config[2, 0]], 'black', 'Final point', 'diamond')
        # Adding initial and final tangent vectors
        fig_3D.arrows_3D([ini_config[0, 0]], [ini_config[1, 0]], [ini_config[2, 0]],\
                          [ini_config[0, 1]], [ini_config[1, 1]], [ini_config[2, 1]],\
                          'orange', 'oranges', 'Initial tangent vector', 3, 1, 0.5, 'n')
        fig_3D.arrows_3D([fin_config[0, 0]], [fin_config[1, 0]],\
                         [fin_config[2, 0]], [fin_config[0, 1]],\
                         [fin_config[1, 1]], [fin_config[2, 1]],\
                          'green', 'greens', 'Final tangent vector', 3, 1, 0.5, 'n')
    
        fig_3D.update_layout_3D('X (m)', 'Y (m)', 'Z (m)', 'Initial and final configurations')
        # Writing the figure on the html file
        fig_3D.writing_fig_to_html(filename, 'w')               
        
        # Plotting the path
        fig_3D_copy = copy.deepcopy(fig_3D)
        fig_3D_copy.scatter_3D(x_coords_path, y_coords_path, z_coords_path, 'blue', False)
        
        fig_3D_copy.update_layout_3D('X (m)', 'Y (m)', 'Z (m)', least_cost_path.upper() + ' path')
        # Writing the figure on the html file
        fig_3D_copy.writing_fig_to_html(filename, 'a')
    
    return least_cost_path, least_cost_path_length, least_cost_path_params,\
          x_coords_path, y_coords_path, z_coords_path, Tx, Ty, Tz, possible_path_types, possible_path_params

def generate_random_configs(R):
    '''
    This function generates random initial and final configurations on a sphere
    of radius R centered at the origin.

    Parameters
    ----------
    R : Scalar
        Radius of the sphere.

    Returns
    -------
    ini_config : Numpy array
        Contains the initial configuration. The syntax followed is as
        follows:
            The first column contains the position.
            The second column contains the tangent vector.
            The third column contains the tangent-normal vector.
    fin_config : Numpy array
        Contains the final configuration. The same syntax used for ini_config is
        used here.

    '''
    
    # Generate random longitude and colatitude for the sphere for the initial position
    phi_ini = np.random.rand()*math.pi
    theta_ini = np.random.rand()*2*math.pi
    xini = np.array([R*sin(phi_ini)*cos(theta_ini), R*sin(phi_ini)*sin(theta_ini), R*cos(phi_ini)])
    # Generating the final location
    phi_fin = np.random.rand()*math.pi
    theta_fin = np.random.rand()*2*math.pi
    xfin = np.array([R*sin(phi_fin)*cos(theta_fin), R*sin(phi_fin)*sin(theta_fin), R*cos(phi_fin)])
    
    # Generating random initial and final heading vectors and orthonormalizing them
    # with respect to the respect location's outward facing surface normal
    rand_ini_vect = np.random.rand(3,)
    rand_fin_vect = np.random.rand(3,)
    
    # Orthonormalizing rand_ini_vect with respect to a unit vector along xini
    t_ini = rand_ini_vect - np.dot(rand_ini_vect, xini/R)*xini/R
    if np.linalg.norm(t_ini) <= 10**(-6):
        
        raise Exception('Regenerate the initial tangent vector.')
        
    else:
        
        t_ini = t_ini/np.linalg.norm(t_ini)
    
    # Orthonormalizing rand_fin_vect with respect to a unit vector along xfin
    t_fin = rand_fin_vect - np.dot(rand_fin_vect, xfin/R)*xfin/R
    if np.linalg.norm(t_fin) <= 10**(-6):
        
        raise Exception('Regenerate the final tangent vector.')
        
    else:
        
        t_fin = t_fin/np.linalg.norm(t_fin)
        
    # Constructing the initial configuration matrix
    # Obtaining the tangent-normal vector
    T_ini = np.cross(xini/R, t_ini)
    ini_config = np.array([[xini[0], t_ini[0], T_ini[0]],\
                           [xini[1], t_ini[1], T_ini[1]],\
                           [xini[2], t_ini[2], T_ini[2]]])
    
    # Constructing the final configuration matrix
    # Obtaining the tangent-normal vector
    T_fin = np.cross(xfin/R, t_fin)
    fin_config = np.array([[xfin[0], t_fin[0], T_fin[0]],\
                           [xfin[1], t_fin[1], T_fin[1]],\
                           [xfin[2], t_fin[2], T_fin[2]]])
        
    return ini_config, fin_config    
        
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