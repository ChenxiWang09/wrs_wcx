import cv2
import pyrealsense2 as rs
import numpy as np
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# Start streaming
pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

def camera_pointselect():
    counter = 0
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow('RealSense', color_image)
        key = cv2.waitKey(1)
        address ='yolov5/'+str(counter)+'.jpg'
        if key == ord('q'):
            cv2.imwrite(address,color_image)
            counter+=1
        if counter == 10:
            break

camera_pointselect()