"""
mavsim: manage_viewers
    - Beard & McLain, PUP, 2012
    - Update history:
        3/11/2024 - RWB
"""
# import pyqtgraph as pg
from mav_viewer import MavViewer
from video_writer import VideoWriter
from msg_state import MsgState

class ViewManager:
    def __init__(self, fig,
                 video: bool=False, 
                 animation: bool=False,
                 video_name: str=[],
                 ts_video: float=0.1,
                 scale_aircraft: float = 3.0):
        self.video_flag = video
        self.animation_flag = animation
        # initialize video 
        if self.video_flag is True:
            from video_writer import VideoWriter
            self.video = VideoWriter(
                video_name=video_name
                # bounding_box=(0, 0, 750, 750),
                # output_rate = ts_video
                )
        # initialize the other visualization
        if self.animation_flag: 
            # self.app = pg.QtWidgets.QApplication([]) 
            if self.animation_flag:
                # self.mav_view = MavViewer(app=self.app)
                self.mav_view = MavViewer(fig, scale_aircraft)

    def update(self, fig,
               sim_time: float,
               true_state: MsgState=None):
        if self.animation_flag: 
            self.mav_view.update(true_state)
        # if self.animation_flag or self.data_plot_flag or self.sensor_plot_flag: 
        #     self.app.processEvents()
        if self.video_flag is True: 
            self.video.update(fig)

    def close(self, dataplot_name: str=[], sensorplot_name: str=[]):
        # Save an Image of the Plot
        if self.video_flag: 
            self.video.close()

