import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matlab.engine
import pylab as pl
import math
import cv2.aruco as aruco

counter = 0
coordinates = np.zeros((4, 2), np.int32)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# Start streaming
pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

index=1

def camera_pointselect():
    while True:
        # with aruco markers
        color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame = get_aligned_images()
        intr_matrix = np.array([
            [color_intrin.fx, 0, color_intrin.ppx], [0, color_intrin.fy, color_intrin.ppy], [0, 0, 1]
        ])
        intr_coeffs=color_intrin.coeffs
        # 获取dictionary, 4x4的码，指示位50个
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        # 创建detector parameters
        parameters = aruco.DetectorParameters_create()
        # 输入rgb图, aruco的dictionary, 相机内参, 相机的畸变参数
        corners, ids, rejected_img_points = aruco.detectMarkers(img_color, aruco_dict, parameters=parameters)
        try:
            n=len(ids)
            k=0
            for i in range (n):
                if ids[i] == 0:
                    k=i
                    print(k)
                    print(corners[k])
                    break
        except:
            k=0
        # 估计出aruco码的位姿，0.045对应markerLength参数，单位是meter
        cv2.imshow('RealSense', img_color)
        cv2.setMouseCallback('RealSense', mousePoints)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

    return corners[k],img_color


def mousePoints(event, x, y, flags, params):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(x, y)
        coordinates[counter] = x, y
        counter += 1
        print(coordinates)

def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧，获取颜色和深度的框架集
    aligned_frames = align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐

    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的的depth帧
    aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧

    #### 获取相机参数 ####
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参

    #### 将images转为numpy arrays ####
    img_color = np.asanyarray(aligned_color_frame.get_data())  # RGB图
    img_depth = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）

    return color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame

def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
    x = depth_pixel[0]
    y = depth_pixel[1]
    dis = aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
    # print ('depth: ',dis)       # 深度单位是m
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
    # print ('camera_coordinate: ',camera_coordinate)
    return dis, camera_coordinate

if __name__ == "__main__":
    # --------------------------------
    # Recognition part
    # --------------------------------
    #query part for homography

    coordinates,image1=camera_pointselect()
    cv2.imwrite('template2.jpg',image1)
    pipeline.stop()
    pts_src=np.float32([coordinates[0][0], coordinates[0][1], coordinates[0][2], coordinates[0][3]])
    im_src = image1
    pointseleceted_dictionary = np.load(r'intelrealsense_wcx\newtem.npy', allow_pickle=True).item()
    # Number of parts in this group
    n=1
    # homography transformation and FDCM in matlab
    eng = matlab.engine.start_matlab()
    partnumber_matched=''
    shapecost=np.zeros(n,dtype=float)
    times = 10       #times of the img which used in define the difference between the stantard bounding box and recognition result
        # --------------------------------
        #attribute: shape(contour) by using fdcm_previous
        # --------------------------------
    for i in range(n):
        j=i+1

        size_max = np.amax(pointseleceted_dictionary[j], axis=0)
        size_min = np.amin(pointseleceted_dictionary[j], axis=0)
        size_max_x = int(size_max[0])
        size_max_y = int(size_max[1])
        size_min_x = int(size_min[0])
        size_min_y = int(size_min[1])

        #homography transformation
        pts_dst = np.float32(pointseleceted_dictionary[j])
        im_dst=cv2.imread("C:/Users/wangq/PycharmProjects/pythonProject/intelrealsense_wcx/templateimage/newtemplate/" + str(j) + ".jpg")
        h, status = cv2.findHomography(pts_src, pts_dst)
        im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
        pl.figure(), pl.imshow(im_out[:, :, ::-1]), pl.title('homography transformation result'), pl.show()  # show dst

        sav = str(j)+'.jpg'
        cv2.imwrite(sav, im_out)

        # matlab :fast directional chamfer matching
        partnumber ,value = eng.chamfermatching_nopic_forpython(sav, j, size_min_x, size_min_y, size_max_x, size_max_y,times , nargout=2)
        print("partnumber: ",j, "shapecost: ",value)

        shapecost[i]=value

        # --------------------------------
        # attribute: size
        # --------------------------------
    camera_coordinates = np.zeros((4, 3), dtype=float)
    camera_coordinate_matrix  = np.zeros(3, dtype = float)
    camera_coordinate = np.zeros(3, dtype = float)
    boxsize=2 # halfsize


    # the box of computing the average points
    count=0
    boxsize=2
    pipeline.start(config)
    for i in range(4):
        while True:
        # Get the matched depth img and color img
            color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame = get_aligned_images()  # 获取对齐图像与相机参数
            depth_pixel = coordinates[i]
        # compute 9 points' average position
            count = 0
            camera_coordinate = np.zeros(3, dtype = float)
            distance=0
            for j in range(-boxsize, 1 + boxsize):
                for k in range(-boxsize, 1 + boxsize):
                    dis, camera_coordinate_matrix = get_3d_camera_coordinate([depth_pixel[0] + j, depth_pixel[1] + k],
                                                                             aligned_depth_frame, depth_intrin)
                    if camera_coordinate_matrix[2] != 0:
                        count += 1
                        camera_coordinate = camera_coordinate + camera_coordinate_matrix
                    distance=dis+distance
            camera_coordinate = camera_coordinate / count
            cv2.imshow('RealSense', img_color)
            key = cv2.waitKey(1)
            if key == ord('q'):
                print (camera_coordinate[0],camera_coordinate[1],camera_coordinate[2])
                camera_coordinates[i][:] = camera_coordinate
                break
    obj_legth = math.sqrt((camera_coordinates[0][0] - camera_coordinates[1][0]) ** 2 + (
                camera_coordinates[0][1] - camera_coordinates[1][1]) ** 2 + (
                                      camera_coordinates[0][2] - camera_coordinates[1][2]) ** 2)
    obj_width = math.sqrt((camera_coordinates[1][0] - camera_coordinates[2][0]) ** 2 + (
                camera_coordinates[1][1] - camera_coordinates[2][1]) ** 2 + (
                                      camera_coordinates[1][2] - camera_coordinates[2][2]) ** 2)
    objsize=np.load('objsize.npy', allow_pickle=True).item()

    print("bounding box limit: ",times, "boxsize: ", boxsize, "Distance: ", dis)
    print("object length is: %.4f"% obj_legth,"  object width is: %.4f"% obj_width)

    #params in final cost compotation
    A=1
    B=1
    chosen=100
    chosencost = 100
    sizecost=np.zeros(n,dtype=float)
    for i in range(n):
        j=i+1
        size=objsize[j]
        sizecost[i]=abs(size[0]-obj_legth)+abs(size[1]-obj_width)
        cost=A*sizecost[i]+B*shapecost[i]
        if(chosencost>cost):
            chosen=j
            chosencost=cost

        print("partnumber: %d"% j,"  sizecost: %.4f"% sizecost[i], "  shapecost: %.4f"% shapecost[i],"  cost: %.4f"% cost)
    print("The recognition result is(part number): ",chosen," cost is : %.4f"% chosencost)
    # --------------------------------
    # Position detection part
    # --------------------------------