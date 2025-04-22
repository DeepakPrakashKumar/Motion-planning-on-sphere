# import pyqtgraph.opengl as gl
# import pyqtgraph.Vector as Vector
# from viewers.draw_mav import DrawMav
from draw_mav_stl_matplotlib import DrawMav
from time import time

class MavViewer():
    def __init__(self, fig, scale_aircraft = 3.0, ts_refresh=1./30.):
        self.scale = scale_aircraft
        # initialize Qt gui application and window
#         self.app = app  # initialize QT, external so that only one QT process is running
#         self.window = gl.GLViewWidget()  # initialize the view object
#         #gl.GLViewWidget.getViewport().setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False)
#         self.window.setWindowTitle('MAV Viewer')
#         grid = gl.GLGridItem() # make a grid to represent the ground
#         grid.scale(20, 20, 20) # set the size of the grid (distance between each line)
#         self.window.addItem(grid) # add grid to viewer
#         self.window.setCameraPosition(distance=200) # distance from center of plot to camera
#         self.window.setBackgroundColor('k')  # set background color to black
#         self.window.setGeometry(0, 0, 750, 750)  # args: upper_left_x, upper_right_y, width, height
#         # center = self.window.cameraPosition()
#         # center.setX(250)
#         # center.setY(250)
#         # center.setZ(0)
#         # self.window.setCameraPosition(pos=center, distance=self.scale, elevation=50, azimuth=-90)
# #        self.window.viewport().setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False)
#         self.window.show()  # display configured window
        # self.window.raise_() # bring window to the front
        self.fig = fig

        self.plot_initialized = False # has the mav been plotted yet?
        self.mav_plot = []
        self.ts_refresh = ts_refresh
        self.t = time()
        self.t_next = self.t

    def update(self, state):
        # initialize the drawing the first time update() is called
        if not self.plot_initialized:
            self.mav_plot = DrawMav(state, self.fig, self.scale)
            self.plot_initialized = True
        # else update drawing on all other calls to update()
        else:
            # t = time()
            # if t-self.t_next > 0.0:
            self.mav_plot.update(state)
            # self.t = t
            # self.t_next = t + self.ts_refresh
        # update the center of the camera view to the mav location
        # view_location = Vector(state.east, state.north, state.altitude)  # defined in ENU coordinates
        # self.window.opts['center'] = view_location
        # redraw
    
    # def process_app(self):
    #     self.app.processEvents()

    # def clear_viewer(self):
    #     self.window.clear()

