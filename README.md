# Motion planning for Dubins vehicle on sphere
The code for motion planning for a Dubins vehicle moving on a sphere is provided, which is used as a part of the paper "A New Approach to Motion Planning in 3D for a Dubins Vehicle: Special Case on a Sphere". The paper is available at https://arxiv.org/abs/2504.01215. The vehicle considered is moving over a sphere, and has a minimum turning radius. The goal is to plan the optimal path to travel from one configuration to another. 

<!-- A representation of the minimum turning radius is shown below. -->
<!-- [![Watch the video](https://img.youtube.com/vi/-0TfJhciwR0/hqdefault.jpg)](https://www.youtube.com/watch?v=-0TfJhciwR0) -->

The functions for path construction and obtaining configurations along the path are provided in Path_generation_sphere.py.
<!-- The obtained path can also be visualized through an animation to obtain a path as shown below! -->
<!-- [![Watch the video](https://img.youtube.com/vi/hjuDgD-WeZk/hqdefault.jpg)](https://www.youtube.com/watch?v=hjuDgD-WeZk) -->

## Primitive segments

The optimal paths for the motion planning problem on the sphere for the Dubins vehicle are composed of three primitive (optimal) segments: a left turn $L$ of minimum turning radius $r$, a right turn $R$ of minimum turning radius $r$, or a great circular arc $G$. These segments are illustrated below.

<table>
  <tr>
    <td align="center"><b>Left turn (<i>L</i>)</b><br><img src="src/GIFs for paths/trajectory_left_0_4.gif"/></td>
    <td align="center"><b>Right turn (<i>R</i>)</b><br><img src="src/GIFs for paths/traj_right_0_4.gif"/></td>
    <td align="center"><b>Great circular arc (<i>G</i>)</b><br><img src="src/GIFs for paths/trajectory_g.gif"/></td>
  </tr>
</table>

## Candidate optimal paths

Let $C = L, R$ denotes a left or right turn of minimum turning radius $r$, and $G$ denotes a great circular arc. The key result of this paper is show that the candidate optimal paths changes with changing $r$. The result depends on whether $r$ is such that (i) $r \leq \frac{1}{2}$, (ii) $r \leq \frac{1}{\sqrt{2}}$, or (iii) $r \leq \frac{\sqrt{3}}{2}$.

### Case $r \leq \frac{1}{2}$

The candidate optimal paths for $r \leq \frac{1}{2}$ is of type $CGC$ ($LGL$, $LGR$, $RGL$, $RGR$) or $CCC$ ($LRL$, $RLR$), or a degenerate path of the same.

<table>
  <tr>
    <td align="center"><b>LGL</b><br><img src="src/GIFs for paths/lgl_path_r_0_4_angles_1_2rad_0_6rad_1_4rad.gif"/></td>
    <td align="center"><b>LGR</b><br><img src="src/GIFs for paths/lgr_path_r_0_4_angles_1_2rad_0_6rad_1_4rad.gif"/></td>
    <td align="center"><b>RGL</b><br><img src="src/GIFs for paths/rgl_path_r_0_4_angles_1_2rad_0_6rad_1_4rad.gif"/></td>
  </tr>
  <tr>
    <td align="center"><b>RGR</b><br><img src="src/GIFs for paths/rgr_path_r_0_4_angles_1_2rad_0_6rad_1_4rad.gif"/></td>
    <td align="center"><b>LRL</b><br><img src="src/GIFs for paths/lrl_path_r_0_4_angles_1_5rad_3pi_by2rad_1_4rad.gif"/></td>
    <td align="center"><b>RLR</b><br><img src="src/GIFs for paths/rlr_path_r_0_4_angles_1_5rad_3pi_by2rad_1_4rad.gif"/></td>
  </tr>
</table>

### Case $r \leq \frac{1}{\sqrt{2}}$

For $r \leq \frac{1}{\sqrt{2}}$, the previous candidate optimal paths are optimal. However, in addition to the $CGC$ and $CCC$ paths, the $CCCC$ path ($LRLR$, $RLRL$) is also a candidate optimal path.

<table>
  <tr>
    <td align="center"><b>LRLR</b><br><img src="src/GIFs for paths/traj_lrlr_0_55_0_35_rad_3_54575rad.gif"/></td>
    <td align="center"><b>RLRL</b><br><img src="src/GIFs for paths/traj_rlrl_0_55_0_35_rad_3_54575rad.gif"/></td>
  </tr>
</table>

### Case $r \leq \frac{\sqrt{3}}{2}$

For $r \leq \frac{\sqrt{3}}{2}$, in addition to the paths above, the $CC_{\pi}C$ ($LRL$ and $RLR$ with the middle segment subtending angle $\pi$) and $CCCCC$ paths are candidate optimal paths.

<table>
  <tr>
    <td align="center"><b>LRL (CC<sub>π</sub>C)</b><br><img src="src/GIFs for paths/lrl_path_r_0_71_angles_0_7rad_pi_rad_0_7rad.gif"/></td>
    <td align="center"><b>RLR (CC<sub>π</sub>C)</b><br><img src="src/GIFs for paths/rlr_path_r_0_71_angles_0_7rad_pi_rad_0_7rad.gif"/></td>
  </tr>
</table>

> **Remark:** Though $CCCCC$ is a candidate optimal path, we were unable to find an initial and final configuration for which a $CCCCC$ path was indeed optimal. Such a path could potentially be non-optimal, and may be removable from the candidate set. However, since we are currently unable to prove non-optimality of CCCCC, it remains to be in the candidate optimal list. The reader is referred to the paper for a more detailed discussion.

## Implementation of path

All the candidate optimal paths are constructed in the function 'Path_generation_sphere.py'. The function optimal_path_sphere.py runs through all candidate paths (depending on the turning radius) and returns the optimal path. Note that in this function, the radius of the sphere need not necessarily be one; arbitrary sphere sizes can be passed. The configurations are scaled such that the path construction is performed on a unit sphere, and the final path is scaled back up to return the optimal path on the original sphere.

The paths can be visualized in a html file, which uses functions in the plotting_class.py file.

The analytical construction of these paths is provided in another document on Arxiv, which is available at [https://arxiv.org/abs/2504.11832](https://arxiv.org/abs/2504.11832).

For CCCCC path construction, a cubic function needs to be solved; to this end, a cubic equation solver, given in https://github.com/shril/CubicEquationSolver, is used.

## Numerical results for the paper

Sample use of the optimal_path function in the Path_generation script is given in numerical_results_paper.ipynb, which contains the scenarios used to show optimality of $CC_{\pi}C$ and $CCCC$ paths.

## Visualization of the path

Any path can be visualization using an animation as well. An example implementation is shown in example_paths_with_visualization.ipynb, whose output was previously shown. Alternately, visualization can also be performed using plot_trajectory function in visualization_simulation.py function.

For visualization, a majority of the scripts and the stl file (in Visualization folder) was taken from the mavsim_public repository (available at https://github.com/byu-magicc/mavsim_public?tab=readme-ov-file).

## 📖 References

If you use this work, please cite:

```bibtex
@article{deepak3D,
  title   = {A New Approach to Motion Planning in 3D for a Dubins Vehicle: Special Case on a Sphere},
  author  = {Kumar, Deepak Prakash and Darbha, Swaroop and Manyam, Satyanarayana Gupta and Casbeer, David},
  journal = {arXiv preprint arXiv:2504.01215},
  year    = {2025}
}
```

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited. AFRL-2025-0643; Cleared 05 Feb 2025.
