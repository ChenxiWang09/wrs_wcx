import pyrealsense2 as rs2
import numpy as np
import open3d as o3d
import cv2
from cv2 import aruco as aruco

class RealSense(object):
    def __init__(self, color_frame_size=(640, 480), depth_frame_size=(640, 480),
                 color_frame_format=rs2.format.bgr8, depth_frame_format=rs2.format.z16,
                 color_frame_framerate=30, depth_frame_framerate=30, camera_id=0):

        self.__pipeline = rs2.pipeline()

        self.__config= rs2.config()

        self.__config.enable_stream(rs2.stream.color, color_frame_size[0], color_frame_size[1],
                                    color_frame_format, color_frame_framerate)
        self.__config.enable_stream(rs2.stream.depth, depth_frame_size[0], depth_frame_size[1],
                                    depth_frame_format, depth_frame_framerate)
        self.__flag = False
        connect_device = []
        for d in rs2.context().devices:
            print('Found device: ',
                  d.get_info(rs2.camera_info.name), ' ',
                  d.get_info(rs2.camera_info.serial_number))
            if d.get_info(rs2.camera_info.name).lower() != 'platform camera':
                connect_device.append(d.get_info(rs2.camera_info.serial_number))

        self.__config.enable_device(connect_device[camera_id])

        profile = self.__pipeline.start(self.__config)

        colorIntr = profile.get_stream(rs2.stream.color).as_video_stream_profile().get_intrinsics()
        depthIntr = profile.get_stream(rs2.stream.depth).as_video_stream_profile().get_intrinsics()
        self.fov = 2*np.arctan(colorIntr.width/(2*colorIntr.fx))
        self.colorCoeffs = colorIntr.coeffs
        self.height = colorIntr.height
        self.width = colorIntr.width

        self.colorIntrinsicMat = np.array([[colorIntr.fx, 0, colorIntr.ppx], [0, colorIntr.fy, colorIntr.ppy], [0, 0, 1]])
        self.depthIntrinsicMat = np.array([[depthIntr.fx, 0, depthIntr.ppx], [0, depthIntr.fy, depthIntr.ppy], [0, 0, 1]])

    def change_camera(self,camera_id=0):
        '''
        Change rs camera
        :param camera_id: the camera you want to switch
        :return: None
        '''
        connect_device = []
        self.__pipeline.stop()
        for d in rs2.context().devices:
            print('Found device: ',
                  d.get_info(rs2.camera_info.name), ' ',
                  d.get_info(rs2.camera_info.serial_number))
            if d.get_info(rs2.camera_info.name).lower() != 'platform camera':
                connect_device.append(d.get_info(rs2.camera_info.serial_number))
        print('change to the device:', d.get_info(rs2.camera_info.name), '', d.get_info(rs2.camera_info.serial_number))
        self.__config.enable_device(connect_device[camera_id])
        self.__pipeline.start(self.__config)

    def get_rgb(self):
        frames = self.__pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        # cv2.imshow('img',color_image)
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
        '''
        Get both rgb and depth image and make them aligned
        :return: aligned frames
        '''
        frames = self.__pipeline.wait_for_frames()
        align_to = rs2.stream.color
        align = rs2.align(align_to)
        aligned_frames = align.process(frames) #aligned frame

        return aligned_frames

    def get_position_camera_coordinate(self, depth_pixel, aligned_depth_frame, depth_intrin):
        '''
        By using the pixel coordinate to get its camera coordinate
        :param depth_pixel:
        :param aligned_depth_frame:
        :param depth_intrin:
        :return:
        '''
        x = depth_pixel[0]
        y = depth_pixel[1]

        aligned_frames = self.get_aligned_frames()
        aligned_depth_frame = aligned_frames.get_depth_frame()  # get depth frame
        dis = aligned_depth_frame.get_distance(x, y)  # get distance (z)
        camera_coordinate = rs2.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)

        return dis, camera_coordinate

    def get_undistored_image(self, trigger=False):

        dist_coeffs = np.array(self.colorCoeffs)
        width, height = self.width, self.height
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.colorIntrinsicMat, dist_coeffs, (width, height), 0)
        map_x, map_y = cv2.initUndistortRectifyMap(self.colorIntrinsicMat, dist_coeffs, None, new_camera_matrix,
                                                   (width, height), cv2.CV_32FC1)
        undistorted_image = cv2.remap(self.get_rgb(), map_x, map_y, cv2.INTER_LINEAR)
        while trigger:
            undistorted_image = cv2.remap(self.get_rgb(), map_x, map_y, cv2.INTER_LINEAR)
            cv2.imshow('Undistorted Image', undistorted_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        return undistorted_image

    def depth2pcd(self, toggledebug=False):
        aligned_frames = self.get_aligned_frames()
        depthimg = np.asanyarray(aligned_frames.get_depth_frame().get_data())
        pinhole_camera_intrinsic = \
        o3d.camera.PinholeCameraIntrinsic(self.intr['width'], self.intr['height'],
                                              self.intr['fx'], self.intr['fy'], self.intr['ppx'], self.intr['ppy'])
        depthimg = o3d.geometry.Image(depthimg)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depthimg, pinhole_camera_intrinsic)
        mid_point = np.asarray(pcd.points)*1000
        mid = o3d.geometry.PointCloud()
        mid.points = o3d.utility.Vector3dVector(mid_point)

        if toggledebug:
            o3d.visualization.draw_geometries([pcd])
            print(np.asarray(pcd.points))
        return mid

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

    def get_3d_camera_coordinate(self, depth_pixel: object) -> object:
        pixel=[depth_pixel[0], depth_pixel[1]]
        x = int(depth_pixel[0])
        y = int(depth_pixel[1])
        aligned_frames = self.get_aligned_frames()
        aligned_depth_frame = aligned_frames.get_depth_frame()  # get depth frame
        dis = aligned_depth_frame.get_distance(x, y)  # get distance (z)
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        # print ('depth: ',dis)       # 深度单位是m
        camera_coordinate = rs2.rs2_deproject_pixel_to_point(depth_intrin, pixel, dis)
        # print ('camera_coordinate: ',camera_coordinate)
        coordinate=np.array([[camera_coordinate[0],camera_coordinate[1],camera_coordinate[2]]])*1000
        return coordinate

    def getcenter(self, tgtids=[0]):
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
            print("no ditection!")
            return None
        if len(ids) != len(tgtids):
            print("the aruco marker number is not correct!")
            return None
        if ids[0] not in tgtids:
            print("the aruco marker number is not correct!")
            return None
        center = np.mean(np.mean(corners, axis=0), axis=1)[0]
        center = [int(center[0]),int(center[1])]
        # print(center)
        pos = self.get_3d_camera_coordinate(center)
        pos = pos

        return pos
    def get_intrinsic_matrix_distortion(self):
        color_intrisic = self.get_aligned_frames().get_color_frame().profile.as_video_stream_profile().intrinsics
        intr_matrix = np.array([
            [color_intrisic.fx, 0, color_intrisic.ppx], [0, color_intrisic.fy, color_intrisic.ppy], [0, 0, 1]
        ])
        distortion = np.array(color_intrisic.coeffs)
        return  intr_matrix, distortion

    def get_extrinsic_matrix(self, tgtids=[0],trigger=False):

        # self.waitforframe() # make frame stable

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
        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners, 0.100, intr_matrix, distortion)
        if trigger:
            while True:
                img = self.get_rgb()
                # try:
                aruco.drawDetectedMarkers(img, corners)
                aruco.drawAxis(img, intr_matrix, distortion, rvec[0], tvec[0], 0.05)
                cv2.imshow('RGB image', img)
                # except:
                #     cv2.imshow('RGB image', img)
                key = cv2.waitKey(1)
                # 'q' exit
                if key & 0xFF == ord('q') or key == 27:
                    break
                # 's' save
                elif key == ord('s'):
                    n = n + 1
                    # saving
                    cv2.imwrite('./img/rgb' + str(n) + '.jpg', img)
            cv2.destroyAllWindows()
        if len(corners) < len(tgtids):
            print("no ditection!")
            return None
        if len(ids) != len(tgtids):
            print("the aruco marker number is not correct!")
            return None
        if ids[0] not in tgtids:
            print("the aruco marker number is not correct!")
            return None

        rvec_matrix_1,jacbian=cv2.Rodrigues(rvec[0])
        tvec_matrix_1=np.array(tvec[0]).T*1000
        extrinsic_matrix=np.hstack((rvec_matrix_1,tvec_matrix_1))
        extrinsic_matrix=np.vstack((extrinsic_matrix,np.array([0,0,0,1])))

        return extrinsic_matrix

    def getcenter_two_aruco_maker(self, tgtids=[0,1]):
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
            print("no enough ditection!")
            return None
        if len(ids) != len(tgtids):
            print("no enough ditection!! ")
            return None
        if ids[0] not in tgtids or ids[1] not in tgtids:
            print("the aruco marker number is not correct!")
            return None
        center = np.mean(np.mean(corners, axis=0), axis=1)[0]
        center = [int(center[0]),int(center[1])]
        # print(center)
        pos = self.get_3d_camera_coordinate(center)
        pos = pos

        return pos

    def get_extrinsic_matrix_two_aruco_maker(self,aruco_size=0.045, tgtids=[0,1],trigger=False):

        # self.waitforframe() # make frame stable

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
        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners, aruco_size, intr_matrix, distortion)
        if len(corners) < len(tgtids):
            print("no ditection!")
            return None
        if len(ids) != len(tgtids):
            print("the aruco marker number is not correct!")
            return None
        if ids[0] not in tgtids or ids[1] not in tgtids:
            print("the aruco marker number is not correct!")
            return None
        if trigger == True:
            while True:
                img = self.get_rgb()
                # try:
                aruco.drawDetectedMarkers(img, corners)
                aruco.drawAxis(img, intr_matrix, distortion, rvec[0], tvec[0], 0.05)
                aruco.drawAxis(img, intr_matrix, distortion, rvec[1], tvec[1], 0.05)
                cv2.imshow('RGB image', img)
                # except:
                #     cv2.imshow('RGB image', img)
                key = cv2.waitKey(1)
                # 'q' exit
                if key & 0xFF == ord('q') or key == 27:
                    break
                # 's' save
                elif key == ord('s'):
                    n = n + 1
                    # saving
                    cv2.imwrite('./img/rgb' + str(n) + '.jpg', img)
            cv2.destroyAllWindows()

        rvec_matrix_1,jacbian=cv2.Rodrigues(rvec[0])
        tvec_matrix_1=np.array(tvec[0]).T*1000
        rvec_matrix_2,jacbian=cv2.Rodrigues(rvec[1])
        tvec_matrix_2=np.array(tvec[1]).T*1000
        rvec_matrix=(rvec_matrix_1+rvec_matrix_2)/2
        tvec_matrix=(tvec_matrix_1+tvec_matrix_2)/2
        extrinsic_matrix=np.hstack((rvec_matrix,tvec_matrix))
        extrinsic_matrix=np.vstack((extrinsic_matrix,np.array([0,0,0,1])))

        return extrinsic_matrix

    def waitforframe(self) -> object:
        for i in range(20):
            while 1:
                depth_frame=self.get_aligned_frames().get_depth_frame()
                color_frame=self.get_aligned_frames().get_color_frame()
                if not color_frame or not depth_frame:
                    continue
                break

    def stop(self):
        self.__pipeline.stop()
        self.__flag = False

if __name__ == '__main__':
    realsense_client = RealSense(color_frame_size=(1280, 720), depth_frame_size=(1280, 720),
                                     color_frame_framerate=30, depth_frame_framerate=30, camera_id=0)
    intrinsic = realsense_client.colorIntrinsicMat
    print('intrinsic:', intrinsic)
    print('fov:',realsense_client.fov)
    while True:
        rgb = realsense_client.get_undistored_image()
        cv2.imshow('realsense_rgb_demo', rgb)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
