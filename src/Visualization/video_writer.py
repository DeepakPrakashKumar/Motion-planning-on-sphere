"""
mavsimPy: video making function
    - Beard & McLain, PUP, 2012
    - Update history:  
        1/10/2019 - RWB
"""
# import numpy as np
# import cv2
# from PIL import ImageGrab


# class VideoWriter():
#     def __init__(self, video_name="video.avi", bounding_box=(0, 0, 1000, 1000), output_rate = 0.1):
#         # bbox specifies specific region (bbox= top_left_x, top_left_y, width, height)
#         # set up video writer by grabbing first image and initializing
#         img = ImageGrab.grab(bbox=bounding_box)
#         img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#         height, width, channels = img.shape
#         # Define the codec and create VideoWriter object
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         self.video = cv2.VideoWriter(video_name, fourcc, 20.0, (width, height))
#         self.bounding_box = bounding_box
#         self.output_rate = output_rate
#         self.time_of_last_frame = 0

#     ###################################
#     # public functions
#     def update(self, time):
#         if (time-self.time_of_last_frame) >= self.output_rate:
#             img = ImageGrab.grab(bbox=self.bounding_box)
#             img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#             self.video.write(img)
#             self.time_of_last_frame = time

#     def close(self):
#         self.video.release()


import numpy as np
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

class VideoWriter():
    def __init__(self, video_name="video.avi"):
        # Initialize the video writer
        self.video_name = video_name
        # self.output_rate = output_rate
        self.time_of_last_frame = 0
        
        # Create a temporary figure to determine size
        temp_fig = Figure()
        temp_canvas = FigureCanvas(temp_fig)
        temp_canvas.draw()
        width, height = temp_canvas.get_width_height()

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video = cv2.VideoWriter(video_name, fourcc, 10.0, (width, height))

    def update(self, fig):
        # if (time - self.time_of_last_frame) >= self.output_rate:
        # Render the Matplotlib figure to an image
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        width, height = canvas.get_width_height()
        img = buf.reshape(height, width, 3)

        # Convert RGB to BGR for OpenCV compatibility
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.video.write(img_bgr)
        # self.time_of_last_frame = time

    def close(self):
        print('Closing video writer')
        self.video.release()

# Example usage:
# if __name__ == "__main__":
#     import time
    
#     # Create a Matplotlib figure
#     fig = Figure()
#     ax = fig.add_subplot(111)
#     ax.plot([0, 1, 2], [0, 1, 4])

#     # Initialize the video writer
#     video_writer = VideoWriter("output_video.avi")

#     start_time = time.time()
#     for i in range(10):
#         current_time = time.time() - start_time
        
#         # Update the plot (example animation)
#         ax.clear()
#         ax.plot([0, 1, 2], [0, 1, 4])
#         ax.set_title(f"Frame {i}")

#         # Write the current figure to the video
#         video_writer.update(fig, current_time)

#         time.sleep(0.5)  # Simulate time delay between frames

#     # Close the video writer
#     video_writer.close()
