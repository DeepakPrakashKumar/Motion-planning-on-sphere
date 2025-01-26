# Motion-planning-on-sphere
The code for motion planning for a Dubins vehicle moving on a sphere is shown. The functions for path construction and obtaining configurations along the path are provided in Path_generation_sphere.py.

## Candidate optimal paths

The candidate optimal paths for the Dubins problem on a sphere are shown to be of type CGC, CCC, or a degenerate path for r <= 1/2. For r <= 1/sqrt(2), CCCC path is also optimal. Finally, for r <= sqrt(3)/2, CCpiC and CCCCC paths are also optimal.

## Unit tests for determining the tolerances in the code for accounting for numerical issues.

The paths are constructed using inverse kinematics, the details for which are provided in the document attached in the repository.

### Tolerance for three segment paths

Finally, a tolerance of 10^(-4) is used to check if the constructed path attains the desired configuration. To this end, the difference in rotation matrices between the desired final configuration and actual final configuration is considered. The maximum and minimum difference must be within the tolerance for the constructed path to be acceptable.

### Tolerances for CCpiC path

The CCpiC path construction depends on the turning radius. If the modified turning radius on the unit sphere is 1/sqrt(2), then the path is constructed through an alternate method. To account for this difference, a tolerance of 0.0001 is selected to distinguish between these cases.

Similar to the three segment paths, a tolerance of 10^(-8) is selected to checking if the angle is nearly zero. That is, if the angle is within 2*pi - 10^(-8), it is rounded to zero.

Finally, a tolerance of 10^(-4) is used to check if the constructed path attains the desired configuration, similar to the three segment paths.

These tolerances were selected based on the unit tests performed in "forward_kinematics_path_testing_CCpiC_path.ipynb" file.