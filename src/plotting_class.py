# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 10:17:19 2022

@author: deepak
"""

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'
from plotly.figure_factory import create_quiver

class plotting_functions:
    
    def __init__(self):
        
        # Declaring as plotly graphic objects figure environment
        
        self.fig = go.Figure()
    
    # Function for adding a trace
    def add_trace(self, data):
        
        self.fig.add_traces(data)
    
    # Function for adding a 2D scatter plot
    def scatter_2D(self, xdata, ydata, color_plot, legend, linewidth = 3, linestyle = 'solid'):
        
        if legend != False:
        
            self.fig.add_trace(go.Scatter
                               (x = xdata,
                                y = ydata,
                                mode = 'lines',
                                line = dict(color = color_plot,\
                                            width = linewidth, dash = linestyle),
                                name = legend
                                ))
            
        else:
            
            self.fig.add_trace(go.Scatter
                               (x = xdata,
                                y = ydata,
                                mode = 'lines',
                                line = dict(color = color_plot, colorscale = 'Viridis',\
                                            width = linewidth, dash = linestyle),
                                showlegend = False
                                ))
    
    # Function for adding points to 2D plot
    def points_2D(self, xcoord, ycoord, color_marker, legend, marker_type = 'circle', marker_size = 8):
        
        if legend != False:
        
            self.fig.add_trace(go.Scatter
                               (x = xcoord,
                                y = ycoord,
                                mode = 'markers',
                                marker = dict(size = marker_size, colorscale = 'Viridis',\
                                              symbol = marker_type, color = color_marker),
                                name = legend
                                ))
            
        else:
            
            self.fig.add_trace(go.Scatter
                               (x = xcoord,
                                y = ycoord,
                                mode = 'markers',
                                marker = dict(size = marker_size, colorscale = 'Viridis',\
                                              symbol = marker_type, color = color_marker),
                                showlegend = False
                                ))
                
    # Function for adding arrows to 2D plot
    def arrows_2D(self, xstart, ystart, dir_cos_x, dir_cos_y, color_arrow, legend,\
                  arrowsize = 3, linewidth = 3):
        
        if legend != False:
        
            # Creating a temporary new figure
            temp = create_quiver(x = xstart,
                                 y = ystart,
                                 u = dir_cos_x,
                                 v = dir_cos_y,
                                 line = dict(width = linewidth, color = color_arrow),
                                 scale = arrowsize,
                                 name = legend)
            
        else:
            
            # Creating a temporary new figure
            temp = create_quiver(x = xstart,
                                 y = ystart,
                                 u = dir_cos_x,
                                 v = dir_cos_y,
                                 line = dict(width = linewidth, color = color_arrow),
                                 scale = arrowsize,
                                 showlegend = False)
            
        # Adding trace to the figure
        self.fig.add_traces(data = temp.data)
        
    # Function for adding a surface to 3D plot
    def surface_3D(self, xdata, ydata, zdata, color, legend, opacity_surf):
        # Here, xdata and ydata should be such that it is in a mesh format.
        # zdata should be the coordinate for each x, y coordinate in the grid.
        
        if legend != False:
        
            # Adding the surface
            self.fig.add_trace(go.Surface(x = xdata,
                                y = ydata,
                                z = zdata,
                                name = legend,
                                showscale = False,
                                colorscale = [[0, color], [1, color]],
                                opacity = opacity_surf))
            
        else:
            
            # Adding the surface
            self.fig.add_trace(go.Surface
                               (x = xdata,
                                y = ydata,
                                z = zdata,
                                showscale = False,
                                colorscale = [[0, color], [1, color]],
                                opacity = opacity_surf,
                                showlegend = False))
            
    # Function for adding a 3D scatter plot
    def scatter_3D(self, xdata, ydata, zdata, color_plot, legend, linewidth = 5, linestyle = 'solid',\
                   text = 'n', text_str = 'none', text_position = 'top right', text_size = 18):
        
        if text.lower() == 'n':
        
            if legend != False:
            
                self.fig.add_trace(go.Scatter3d
                                   (x = xdata,
                                    y = ydata,
                                    z = zdata,
                                    mode = 'lines',
                                    line = dict(color = color_plot, colorscale = 'Viridis',\
                                                width = linewidth, dash = linestyle),
                                    name = legend
                                    ))
                
            else:
                
                self.fig.add_trace(go.Scatter3d
                                   (x = xdata,
                                    y = ydata,
                                    z = zdata,
                                    mode = 'lines',
                                    line = dict(color = color_plot, colorscale = 'Viridis',\
                                                width = linewidth, dash = linestyle),
                                    showlegend = False
                                    ))
                    
        elif text.lower() == 'y':
            
            if legend != False:
            
                self.fig.add_trace(go.Scatter3d
                                   (x = xdata,
                                    y = ydata,
                                    z = zdata,
                                    mode = 'lines+text',
                                    line = dict(color = color_plot, colorscale = 'Viridis',\
                                                width = linewidth, dash = linestyle),
                                    name = legend,
                                    text = ['', text_str],
                                    textposition = text_position,
                                    textfont_size = text_size
                                    ))
                
            else:
                
                self.fig.add_trace(go.Scatter3d
                                   (x = xdata,
                                    y = ydata,
                                    z = zdata,
                                    mode = 'lines+text',
                                    line = dict(color = color_plot, colorscale = 'Viridis',\
                                                width = linewidth, dash = linestyle),
                                    text = ['', text_str],
                                    textposition = text_position,
                                    textfont_size = text_size,
                                    showlegend = False
                                    ))
    
    # Function for adding points to 3D plot
    def points_3D(self, xcoord, ycoord, zcoord, color_marker, legend, marker_type = 'circle'\
                  , marker_size = 8):
                
        if legend != False:
        
            self.fig.add_trace(go.Scatter3d
                               (x = xcoord,
                                y = ycoord,
                                z = zcoord,
                                mode = 'markers',
                                marker = dict(size = marker_size, colorscale = 'Viridis',\
                                              symbol = marker_type, color = color_marker),
                                name = legend
                                ))
            
        else:
            
            self.fig.add_trace(go.Scatter3d
                               (x = xcoord,
                                y = ycoord,
                                z = zcoord,
                                mode = 'markers',
                                marker = dict(size = marker_size, colorscale = 'Viridis',\
                                              symbol = marker_type, color = color_marker),
                                showlegend = False
                                ))
    
    # Function for adding arrow to 3D plot
    def arrows_3D(self, xstart, ystart, zstart, dir_cos_x, dir_cos_y, dir_cos_z, color_arrow_line,\
                  color_arrow_tip, legend, linewidth = 3, arrowsize = 3, arrowtipsize = 3, text = 'n',\
                  text_str = 'none', text_position = 'top right', text_size = 18):
        
        # Adding the line corresponding to the arrow using the scatter_3D function
        self.scatter_3D([xstart[0], xstart[0] + dir_cos_x[0]*arrowsize],\
                        [ystart[0], ystart[0] + dir_cos_y[0]*arrowsize],\
                        [zstart[0], zstart[0] + dir_cos_z[0]*arrowsize],\
                        color_arrow_line, legend, linewidth, 'solid', text,\
                        text_str, text_position, text_size)
        # Adding a cone for the end of the arrow
        self.fig.add_trace(go.Cone
                           (x = [xstart[0] + dir_cos_x[0]*arrowsize],
                            y = [ystart[0] + dir_cos_y[0]*arrowsize],
                            z = [zstart[0] + dir_cos_z[0]*arrowsize],
                            u = [dir_cos_x[0]*arrowtipsize],
                            v = [dir_cos_y[0]*arrowtipsize],
                            w = [dir_cos_z[0]*arrowtipsize],
                            colorscale = color_arrow_tip,
                            showscale = False,
                            showlegend = False
                               ))
    
    # Updating layout for 2D plot
    def update_layout_2D(self, xlabel, xlim, ylabel, ylim, plot_title):
        
        self.fig.update_layout(xaxis_title = 'x (m)',\
                               yaxis_title = 'z (m)',\
                               title_text = plot_title,\
                               xaxis_range = xlim,\
                               yaxis_range = ylim)
        
    # Updating layout for 3D plot
    def update_layout_3D(self, xlabel, ylabel, zlabel, plot_title):
        
        self.fig.update_layout(
            # width = 800,
            # height = 700,
            # autosize = False,
            scene = dict(
                camera = dict(
                    up = dict(
                        x = 0,
                        y = 0,
                        z = 1
                    ),
                    eye=dict(
                        # x = 0,
                        # y = 1.0707,
                        # z = 1,
                        x = 1.2,
                        y = 1.0707,
                        z = 1
                    )
                ),
                # aspectratio = dict(x = 0.75, y = 0.75, z = 0.5),
                # aspectmode = 'manual',
                aspectmode = 'cube',
                xaxis_title = xlabel,
                yaxis_title = ylabel,
                zaxis_title = zlabel
            ),
            title_text = plot_title
        )
    
    def writing_fig_to_html(self, filename, mode = 'a'):
        
        with open(filename, mode) as f:
            
            f.write(self.fig.to_html(full_html = False, include_plotlyjs = 'cdn'))

    def write_fig_to_image(self, filename, mode = 'w'):

        with open(filename, mode) as f:

            f.write(self.fig.write_image(filename.split(".")[0] + ".png"))