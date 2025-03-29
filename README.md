# Motion-planning-on-sphere
The code for motion planning for a Dubins vehicle moving on a sphere is shown. The functions for path construction and obtaining configurations along the path are provided in Path_generation_sphere.py.

## Candidate optimal paths

The candidate optimal paths for the Dubins problem on a sphere are shown to be of type CGC, CCC, or a degenerate path for r <= 1/2. For r <= 1/sqrt(2), CCCC path is also optimal. Finally, for r <= sqrt(3)/2, CCpiC and CCCCC paths are also optimal.

## Implementation of path

All the candidate optimal paths are constructed in the function 'Path_generation_sphere.py'. The function optimal_path_sphere.py runs through all candidate paths (depending on the turning radius) and returns the optimal path. Note that in this function, the radius of the sphere need not necessarily be one; arbitrary sphere sizes can be passed. The configurations are scaled such that the path construction is performed on a unit sphere, and the final path is scaled back up to return the optimal path on the original sphere.

The paths can be visualized in a html file, which uses functions in the plotting_class.py file.

The equations for construction of the path is attached in a pdf file in this repository.

## Numerical results for the paper

Sample use of the optimal_path function in the Path_generation script is given in numerical_results_paper.ipynb, which contains the scenarios used to show optimality of CCpiC and CCCC paths. Furthermore, visualization of the path

## Visualization of the path

Any path can be visualization using an animation as well. An example implementation is shown in example_paths_with_visualization.ipynb, whose output is shown at https://www.youtube.com/watch?v=hjuDgD-WeZk. Alternately, visualization can also be performed using plot_trajectory function in visualization_simulation.py function.

For visualization, a majority of the scripts and the stl file (in Visualization folder) was taken from the mavsim_public repository (available at https://github.com/byu-magicc/mavsim_public?tab=readme-ov-file).