#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import pyrealsense2 as rs
import numpy as np
import csv


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

align_to = rs.stream.color
align = rs.align(align_to)


square_size = 2        # 正方形の1辺のサイズ[cm]
pattern_size = (7, 7)  # 交差ポイントの数

reference_img = 300 # 参照画像の枚数

pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 ) #チェスボード（X,Y,Z）座標の指定 (Z=0)
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size
objpoints = []
imgpoints = []


while len(objpoints) < reference_img:
# 画像の取得
#     ret, img = capture.read()
#     height = img.shape[0]
#     width = img.shape[1]

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not aligned_depth_frame or not color_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    img = cv2.flip(color_image, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # チェスボードのコーナーを検出
    ret, corner = cv2.findChessboardCorners(gray, pattern_size)
    # コーナーがあれば
    if ret == True:
        print("detected coner!")
        print(str(len(objpoints)+1) + "/" + str(reference_img))
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(gray, corner, (5,5), (-1,-1), term)
        imgpoints.append(corner.reshape(-1, 2))   #appendメソッド：リストの最後に因数のオブジェクトを追加
        objpoints.append(pattern_points)

    cv2.imshow('image', img)
    # 毎回判定するから 200 ms 待つ．遅延するのはココ
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

print("calculating camera parameter...")
# 内部パラメータを計算
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 計算結果を保存
np.save("mtx", mtx) # カメラ行列
np.save("dist", dist.ravel()) # 歪みパラメータ
# 計算結果を表示
print("RMS = ", ret)
print("mtx = \n", mtx)
print("dist = ", dist.ravel())