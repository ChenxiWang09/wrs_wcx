# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import pyrealsense2 as rs
import mediapipe as mp
import cv2
import numpy as np
import datetime as dt
import datetime
import csv
import pygame
from pygame.locals import *

# モード選択
DEPTH_MODE = True
REMOVE_MODE = False
ORIENT = True

# 手の検出と追跡の出力を表示するOpenCVウィンドウのフォーマットを設定
font = cv2.FONT_HERSHEY_SIMPLEX
org = (20, 30)
fontScale = 1
color = (255, 255, 255)
thickness = 2

FLAG = True

filter_flag = True
undistortion_flag = False

calib_data = []
calibration_num = 0
max_calibration_num = 50

# ====== Realsense ======
realsense_ctx = rs.context()
connected_devices = []  # List of serial numbers for present cameras
for i in range(len(realsense_ctx.devices)):
    detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
    print(f"{detected_camera}")
    connected_devices.append(detected_camera)
device = connected_devices[0]  # In this example we are only using one camera
pipeline = rs.pipeline()
config = rs.config()
background_removed_color = 153  # Grey


# ====== Mediapipe ======
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils
INDEX_FINGER_TIP = 8

# ====== Enable Streams ======
config.enable_device(device)

# For worse FPS, but better resolution:
stream_res_x = 1280
stream_res_y = 720
# # For better FPS. but worse resolution:
# stream_res_x = 640
# stream_res_y = 480

stream_fps = 30

config.enable_stream(rs.stream.depth, stream_res_x, stream_res_y, rs.format.z16, stream_fps)
config.enable_stream(rs.stream.color, stream_res_x, stream_res_y, rs.format.bgr8, stream_fps)
profile = pipeline.start(config)

depth_intrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.depth)).get_intrinsics()
color_intrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.color)).get_intrinsics()


print("depth_intrinsics")
print(depth_intrinsics)
print()
print("color_intrinsics")
print(color_intrinsics)
print()

align_to = rs.stream.color
align = rs.align(align_to)

# ====== Get depth Scale ======
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"\tDepth Scale for Camera SN {device} is: {depth_scale}")

# ====== Set clipping distance ======
clipping_distance_in_meters = 10
clipping_distance = clipping_distance_in_meters / depth_scale
print(f"\tConfiguration Successful for SN {device}")

# ====== Get and process images ======
print(f"Starting to capture images on SN: {device}")


# decimarion_filterのパラメータ
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 1)

# spatial_filterのパラメータ(平滑化処理)
spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 1)
spatial.set_option(rs.option.filter_smooth_alpha, 0.25)
spatial.set_option(rs.option.filter_smooth_delta, 50)

# hole_filling_filterのパラメータ(欠損値処理)
fill_from_left, farest_from_around, nearest_from_around = 0, 1, 2
hole_filling = rs.hole_filling_filter(nearest_from_around)

# disparity
depth_to_disparity = rs.disparity_transform(False)
disparity_to_depth = rs.disparity_transform(False)

pygame.joystick.init()
try:
    # ジョイスティックインスタンスの生成
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print('Name of joystick:', joystick.get_name())
    print('Num of button :', joystick.get_numbuttons())
except pygame.error:
    print('Joystick is not connected')

# pygameの初期化
pygame.init()

#read numpy file
mtx = np.load("mtx.npy")
dist = np.load("dist.npy")


while FLAG:
    start_time = dt.datetime.today().timestamp()

    # Get and align frames
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    if filter_flag:
        aligned_depth_frame = decimate.process(aligned_depth_frame)
        aligned_depth_frame = depth_to_disparity.process(aligned_depth_frame)
        aligned_depth_frame = spatial.process(aligned_depth_frame)
        aligned_depth_frame = disparity_to_depth.process(aligned_depth_frame)
        aligned_depth_frame = hole_filling.process(aligned_depth_frame)

    color_frame = aligned_frames.get_color_frame()

    if not aligned_depth_frame or not color_frame:
        continue

    # Process images
    depth_image = np.asanyarray(aligned_depth_frame.get_data())

    # 画像を反転させる
    depth_image_flipped = cv2.flip(depth_image, 1)
    color_image = np.asanyarray(color_frame.get_data())

    # 深度画像は1つのチャンネルであり、カラー画像は3つであるため、深度画像を3回スタックする必要がある
    depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
    # depth_image_3d = hole_filling.process(depth_image_3d)
    if REMOVE_MODE:
        background_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0),
                                      background_removed_color, color_image)
    else:
        background_removed = color_image

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

    if DEPTH_MODE:
        images1 = cv2.flip(depth_colormap, 1)
        images2 = cv2.flip(background_removed, 1)
        images = cv2.addWeighted(src1=images1, alpha=0.7, src2=images2, beta=0.5, gamma=0)
    else:
        images = cv2.flip(background_removed, 1)
    color_image = cv2.flip(color_image, 1)
    color_images_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    #歪み補正
    if undistortion_flag:
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (stream_res_x, stream_res_y), 1, (stream_res_x, stream_res_y))
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (stream_res_x, stream_res_y), 5)
        color_images_rgb = cv2.remap(color_images_rgb, mapx, mapy, cv2.INTER_LINEAR)


    # Process hands
    results = hands.process(color_images_rgb)
    if results.multi_hand_landmarks:
        number_of_hands = len(results.multi_hand_landmarks)
        i = 0
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(images, handLms, mpHands.HAND_CONNECTIONS)
            org2 = (20, org[1] + (40 * (i + 1)))
            hand_side_classification_list = results.multi_handedness[i]
            hand_side = hand_side_classification_list.classification[0].label

            # 人差し指のTIPの深度を取得
            index_finger_tip = results.multi_hand_landmarks[i].landmark[INDEX_FINGER_TIP]
            # print(depth_image_flipped[0])
            x_tip = int(index_finger_tip.x * len(depth_image_flipped[0]))
            y_tip = int(index_finger_tip.y * len(depth_image_flipped))
            if x_tip >= len(depth_image_flipped[0]):
                x_tip = len(depth_image_flipped[0]) - 1
            if y_tip >= len(depth_image_flipped):
                y_tip = len(depth_image_flipped) - 1
            tip_distance = depth_image_flipped[y_tip, x_tip] * depth_scale  # meters

            images = cv2.putText(images,
                                 f"{hand_side} x: {x_tip} px, y: {y_tip} px, Depth(z): {tip_distance:0.3} m, calibration num: {calibration_num}",
                                 org2, font, fontScale, color, thickness, cv2.LINE_AA)

            i += 1
        images = cv2.putText(images, f"Hands: {number_of_hands}", org, font, fontScale, color, thickness, cv2.LINE_AA)
    else:
        images = cv2.putText(images, "No Hands", org, font, fontScale, color, thickness, cv2.LINE_AA)

    # Display FPS
    time_diff = dt.datetime.today().timestamp() - start_time
    fps = int(1 / time_diff)
    org3 = (20, org[1] + 120)
    images = cv2.putText(images, f"FPS: {fps}", org3, font, fontScale, color, thickness, cv2.LINE_AA)

    name_of_window = 'SN: ' + str(device)

    # Display images
    cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name_of_window, images)

    key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        print(f"User pressed break key for SN: {device}")
        break

    # 動画・軌道を保存する場合
    for e in pygame.event.get():
        if (e.type == pygame.locals.JOYBUTTONDOWN and e.button == 0) or max_calibration_num == calibration_num:
            FLAG = False
            dt_now = datetime.datetime.now()
            filepath = './calibration_data/calibration_realsense_{year}{month}{day}{hour}{minute}.csv'.format(
                year=dt_now.year, month=dt_now.month,
                day=dt_now.day, hour=dt_now.hour,
                minute=dt_now.minute)

            with open(filepath, "w", newline='') as data:
                writer = csv.writer(data)
                [writer.writerow([calib_data[i][0], calib_data[i][1], calib_data[i][2]]) for i in range(len(calib_data))]

        elif e.type == pygame.locals.JOYBUTTONDOWN and e.button == 1:
            calib_data.append([x_tip, y_tip, round(tip_distance,3)])
            cv2.imwrite('./calibration_data/calibration_point_{}.jpg'.format(calibration_num+1), images)
            print(f"get calibration data!!!!!")
            calibration_num += 1


print(f"Application Closing")
pipeline.stop()
print(f"Application Closed.")