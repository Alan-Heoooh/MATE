'''
RealSense Camera.
'''

import numpy as np
import pyrealsense2 as rs


class RealSenseRGBDCamera:
    '''
    RealSense RGB-D Camera.
    '''
    def __init__(
        self, 
        serial, 
        frame_rate = 30, 
        resolution = (1280, 720),
        align = True,
    ):
        '''
        Initialization.

        Parameters:
        - serial: str, required, the serial number of the realsense device;
        - frame_rate: int, optional, default: 30, the framerate of the realsense camera;
        - resolution: (int, int), optional, default: (1280, 720), the resolution of the realsense camera;
        - align: bool, optional, default: True, whether align the frameset with the RGB image.
        '''
        super(RealSenseRGBDCamera, self).__init__()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.serial = serial
        # =============== Support L515 Camera ============== #
        self.is_radar = str.isalpha(serial[0])
        depth_resolution = (1024, 768) if self.is_radar else resolution
        if self.is_radar:
            frame_rate = max(frame_rate, 30)
            self.depth_scale = 4000.0
        else:
            self.depth_scale = 1000.0
        # ================================================== #
        self.config.enable_device(self.serial)
        self.config.enable_stream(rs.stream.depth, depth_resolution[0], depth_resolution[1], rs.format.z16, frame_rate)
        self.config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.rgb8, frame_rate)
        self.profile = self.pipeline.start(self.config)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.with_align = align

    @property
    def intrinsics(self):
        profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = profile.get_intrinsics()
        return np.array([[intrinsics.fx, 0.0, intrinsics.ppx],
                         [0.0, intrinsics.fy, intrinsics.ppy], [0.0, 0.0, 1.0]])

    def get_rgb_image(self):
        '''
        Get the RGB image from the camera.
        '''
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data()).astype(np.uint8)
        return color_image


    def get_rgbd_image(self):
        '''
        Get the RGB image along with the depth image from the camera.
        '''
        frameset = self.pipeline.wait_for_frames()
        if self.with_align:
            frameset = self.align.process(frameset)
        color_image = np.asanyarray(frameset.get_color_frame().get_data()).astype(np.uint8)
        depth_image = np.asanyarray(frameset.get_depth_frame().get_data()).astype(np.float32)
        return color_image, depth_image
