import pyrealsense2 as rs2
import numpy as np
import open3d as o3d
import cv2
from cv2 import aruco as aruco

class RealSense(object):

    def __init__(self, color_frame_size=(640, 480), depth_frame_size=(640, 480),
                 color_frame_format=rs2.format.bgr8, depth_frame_format=rs2.format.z16,
                 color_frame_framerate=30, depth_frame_framerate=30):
        self.__pipeline = rs2.pipeline()
        self.__config = rs2.config()

        self.__config.enable_stream(rs2.stream.color, color_frame_size[0], color_frame_size[1],
                                    color_frame_format, color_frame_framerate)
        self.__config.enable_stream(rs2.stream.depth, depth_frame_size[0], depth_frame_size[1],
                                    depth_frame_format, depth_frame_framerate)
        self.__flag = False
        profile = self.__pipeline.start(self.__config)

        intr = profile.get_stream(rs2.stream.color).as_video_stream_profile().get_intrinsics()
        self.intr = {'width': intr.width, 'height': intr.height, 'fx': intr.fx, 'fy': intr.fy,
                     'ppx': intr.ppx, 'ppy': intr.ppy}

    def get_rgb(self):
        frames = self.__pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        return color_image

    def get_gray(self):
        frames = self.__pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        return gray

    def get_depth(self):
        frames = self.__pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_image=np.asanyarray(depth_frame.get_data())

        return depth_image

    def get_aligned_frames(self):
        frames = self.__pipeline.wait_for_frames()
        align_to = rs2.stream.color
        align = rs2.align(align_to)
        aligned_frames = align.process(frames) #aligned frame

        return aligned_frames

    def depth2pcd(self, toggledebug=False):
        aligned_frames = self.get_aligned_frames()
        depthimg = np.asanyarray(aligned_frames.get_depth_frame().get_data())
        pinhole_camera_intrinsic = \
        o3d.camera.PinholeCameraIntrinsic(self.intr['width'], self.intr['height'],
                                              self.intr['fx'], self.intr['fy'], self.intr['ppx'], self.intr['ppy'])
        depthimg = o3d.geometry.Image(depthimg)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depthimg, pinhole_camera_intrinsic)
        if toggledebug:
            o3d.visualization.draw_geometries([pcd])
            print(np.asarray(pcd.points))
        return np.asarray(pcd.points)

    def rgbd2pcd(self, toggledebug=False):
        aligned_frames = self.get_aligned_frames()
        depthimg = np.asanyarray(aligned_frames.get_depth_frame().get_data())
        rgbimg = np.asanyarray(aligned_frames.get_color_frame().get_data())
        pinhole_camera_intrinsic = \
        o3d.camera.PinholeCameraIntrinsic(self.intr['width'], self.intr['height'],
                                              self.intr['fx'], self.intr['fy'], self.intr['ppx'], self.intr['ppy'])
        img_depth = o3d.geometry.Image(depthimg)
        img_color = o3d.geometry.Image(cv2.cvtColor(rgbimg, cv2.COLOR_BGR2RGB))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        if toggledebug:
            o3d.visualization.draw_geometries([pcd])
        return np.asarray(pcd.points)

    def get_3d_camera_coordinate(self, depth_pixel):
        x = depth_pixel[0]
        y = depth_pixel[1]
        aligned_frames = self.get_aligned_frames()
        aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的的depth帧
        dis = aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        # print ('depth: ',dis)       # 深度单位是m
        camera_coordinate = rs2.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
        # print ('camera_coordinate: ',camera_coordinate)
        coordinate=np.array([camera_coordinate[0],camera_coordinate[1],camera_coordinate[2]])
        return coordinate

    def getcenter(self, tgtids=[0]):
        """
        get the center of two markers

        :param img:
        :param pcd:
        :return:

        author: yuan gao, ruishuang, revised by weiwei
        date: 20161206
        """
        # while True:
        #     cv2.imshow('img1', img)
        #     key = cv2.waitKey(1)
        #     if key==ord('q'):
        #         break
        img = self.get_rgb()

        parameters = aruco.DetectorParameters_create()
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

        width = img.shape[1]
        # First, detect markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
        if len(corners) < len(tgtids):
            return None
        if len(ids) != len(tgtids):
            return None
        if ids[0] not in tgtids:
            return None
        center = np.mean(np.mean(corners, axis=0), axis=1)[0]
        center = [int(center[0]),int(center[1])]
        # print(center)
        pos = self.get_3d_camera_coordinate(center)
        pos = pos*1000

        return pos

    def get_extrinsic_matrix(self, aruco_size,  tgtids=[0],trigger=False):

        self.waitforframe() # make frame stable

        color_intrisic=self.get_aligned_frames().get_color_frame().profile.as_video_stream_profile().intrinsics
        intr_matrix = np.array([
            [color_intrisic.fx, 0, color_intrisic.ppx], [0, color_intrisic.fy, color_intrisic.ppy], [0, 0, 1]
        ])
        distortion=np.array(color_intrisic.coeffs)
        img = self.get_rgb()

        parameters = aruco.DetectorParameters_create()
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

        width = img.shape[1]
        # First, detect markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners, aruco_size/1000, intr_matrix, distortion)
        if len(corners) < len(tgtids):
            return None
        if len(ids) != len(tgtids):
            return None
        if ids[0] not in tgtids:
            return None
        if trigger == True:
            while True:
                try:
                    # 在图片上标出aruco码的位置
                    aruco.drawDetectedMarkers(img, corners)
                    # 根据aruco码的位姿标注出对应的xyz轴, 0.05对应length参数，代表xyz轴画出来的长度
                    aruco.drawAxis(img, intr_matrix, distortion, rvec[0], tvec[0], 0.05)
                    cv2.imshow('RGB image', img)
                except:
                    cv2.imshow('RGB image', img)
                key = cv2.waitKey(1)
                # 按键盘q退出程序
                if key & 0xFF == ord('q') or key == 27:
                    break
                # 按键盘s保存图片
                elif key == ord('s'):
                    n = n + 1
                    # 保存rgb图
                    cv2.imwrite('./img/rgb' + str(n) + '.jpg', img)
            cv2.destroyAllWindows()

        rvec_matrix_1,jacibian=cv2.Rodrigues(rvec)
        tvec_matrix_1=np.array(tvec[0]).T*1000
        extrinsic_matrix=np.hstack((rvec_matrix_1,tvec_matrix_1))
        extrinsic_matrix=np.vstack((extrinsic_matrix,np.array([0,0,0,1])))

        return extrinsic_matrix

    def waitforframe(self):
        for i in range(100):
            while 1:
                depth_frame=self.get_aligned_frames().get_depth_frame()
                color_frame=self.get_aligned_frames().get_color_frame()
                if not color_frame or not depth_frame:
                    continue
                break

    def stop(self):
        self.__pipeline.stop()
        self.__flag = False