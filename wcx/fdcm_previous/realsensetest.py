import realsense
import cv2
realsense_client=realsense.RealSense()
while True:
    x=320
    y=240
    depth_pixel=[x,y]
    coordinate=realsense_client.get_3d_camera_coordinate(depth_pixel)
    print(coordinate)
    rgb=realsense_client.get_rgb()
    cv2.imshow('rgb',rgb)
    key=cv2.waitKey(1)
    if key == ord('q'):
        break