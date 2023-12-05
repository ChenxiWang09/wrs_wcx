import wcx.utils1.realsense as rs
import cv2

# start_mat = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('data/intact_ex/plenty_cam/dual_arm_2.mp4', fourcc, 30.0, (
    1280, 720))  # output.mp4 is the name of the output file, 20.0 is the FPS, (640, 480) is the frame size

# out_2 = cv2.VideoWriter('data/intact_ex/plenty_cam/Astep_2_1.mp4', four
# cc, 30.0, (
#     1280, 720))  # output.mp4 is the name of the output file, 20.0 is the FPS, (640, 480) is the frame size



part_4_initial_pose = [672.17529754, 26.92461055, 874]

realsense_client = rs.RealSense(color_frame_size=(1280, 720), depth_frame_size=(1280, 720),
                                color_frame_framerate=30, depth_frame_framerate=30,camera_id=0)
# realsense_client_1 = rs.RealSense(color_frame_size=(1280, 720), depth_frame_size=(1280, 720),
#                                 color_frame_framerate=30, depth_frame_framerate=30,camera_id=1)

s_key = False
a_key = False
while True:
    frame = realsense_client.get_rgb()
    # frame_1 = realsense_client_1.get_rgb()

    if a_key:
        text = "Unavailable position! Please continue assembly!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (50, 50)  # (x, y) coordinates of the starting point
        font_scale = 1
        color = (0, 255, 0)  # (B, G, R) color value
        thickness = 2
        line_type = cv2.LINE_AA
        cv2.putText(frame, text, position, font, font_scale, color, thickness, line_type)
    if s_key:
        text = "Please continue to screw in!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (50, 50)  # (x, y) coordinates of the starting point
        font_scale = 1
        color = (0, 255, 0)  # (B, G, R) color value
        thickness = 2
        line_type = cv2.LINE_AA
        cv2.putText(frame, text, position, font, font_scale, color, thickness, line_type)

    out.write(frame)
    # out_2.write(frame_1)
    cv2.imshow('frame', frame)
    # cv2.imshow('frame2', frame_1)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        break
    elif key == ord('0'):
        realsense_client.change_camera(camera_id=0)
    elif key == ord('1'):
        realsense_client.change_camera(camera_id=1)
    elif key == ord('2'):
        realsense_client.change_camera(camera_id=2)
    elif key == ord('s'):
        cv2.imwrite('/wcx/grasptask/data/part_edge.pgm', frame)
        print('succuss!')
    elif key == ord('w'):
        s_key = True
    elif key == ord('e'):
        s_key = False
    elif key == ord('r'):
        a_key = True
    elif key == ord('t'):
        a_key = False


# Release the resources
out.release()
# out_2.release()
cv2.destroyAllWindows()