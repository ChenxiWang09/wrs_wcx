import numpy as np
import wcx.utils1.rotate_matrix as rot_m
import wcx.utils1.move_action as ur_ma
import wcx.utils1.realsense as rs2
import cv2

'''
p1
[476.17873673, -26.78202343, 780]
p2
[476.17873673, 448.4285685,  780]
p3
[860.87519164, 448.4285685,  780]
p4
[860.87519164, -26.78202343,  780]
[[159, 456], [951, 170]]


[[ 206.97656748]
 [-204.05292101]
 [ 733.34442984]
 [   1.        ]]
 
 [[-278.91221642   52.67148465  536.00001335]]
 [[-313.37106984]
 [   8.26309764]
 [ 492.61216315]
 35
 44
 45
+-20
'''
class RealityQueryExtraction():

    def __init__(self, ur_rgt, realsense_client):
        self.__ur_rgt = ur_rgt
        self.__realsense_client = realsense_client

    def cameraRegistration(self):
        '''
        Camera registrate with Robot
        :param ur_rgt: Right arm Robot object
        :param cameraIntrinsicMat: Intrinsic matrix of camera
        :return: Homogeneous matrix, Extrinsic Matrix
        Info: P_image = M_intr * M_GC * inv(M_ee) * P_world / Z_c
        '''
        ZeroCol = np.array([[0, 0, 0]]).T
        TGCCol = np.array([[0, 0, 0]]).T
        ZeroRow = np.array([[0, 0, 0, 1]])
        cameraIntrinsicMat = self.__realsense_client.colorIntrinsicMat
        cameraIntrinsicMat = np.hstack((cameraIntrinsicMat, ZeroCol))

        ee_pos, ee_rot_mat, jnts = self.__ur_rgt.get_pos_rot_jnts_right_now(data_need=True)
        ee_pos = np.array([ee_pos]).T
        # gripperToCamera
        TransGC = rot_m.matrix_generate(0, 'z')
        TransGC = np.hstack((TransGC, ZeroCol))
        TransGC = np.vstack((TransGC, ZeroRow))

        trans_mid = np.hstack((ee_rot_mat, ee_pos))
        TransGR = np.vstack((trans_mid, ZeroRow))
        TransRG = np.linalg.inv(TransGR)

        Extrinsic_mat = np.dot(TransGC, TransRG)
        Extrinsic_mat[0][3] += 51.78587893
        Extrinsic_mat[1][3] += 35.1687166
        Extrinsic_mat[2][3] += 26.02808058


        Homoge_mat = np.dot(cameraIntrinsicMat, Extrinsic_mat)

        return Homoge_mat, Extrinsic_mat

    def computeDistortion(self):
        p1 = np.array([[476.17873673, -26.78202343, 780, 1]]).T
        p2 = np.array([[476.17873673, 448.4285685, 780, 1]]).T
        p3 = np.array([[860.87519164, 448.4285685, 780, 1]]).T
        p4 = np.array([[860.87519164, -26.78202343, 780, 1]]).T
        test_target = []
        test_target.append(p1)
        test_target.append(p2)
        test_target.append(p3)
        test_target.append(p4)
        ideal_coordinates = [[164, 460], [522, 105], [949, 169], [854, 639]]

        Homoge_mat, Extrinsic_mat = self.cameraRegistration(self.__ur_rgt, self.__realsense_client.colorIntrinsicMat)
        diff = [0, 0, 0]
        for i in range(4):
            camera_coordinate = np.dot(Extrinsic_mat, test_target[i])
            print('Camera coordinate:', camera_coordinate)
            pixel_coordinate = np.dot(Homoge_mat, test_target[i])/camera_coordinate[2]
            print('Pixel coordinate:', pixel_coordinate)
            ideal_camera_coordinate = self.__realsense_client.get_3d_camera_coordinate(ideal_coordinates[i]).T
            print('Ideal Camera coordinate:', ideal_camera_coordinate)
            print('ideal Pixel coordinate:', ideal_coordinates[i])

            diff[0] += ideal_camera_coordinate[0][0]-camera_coordinate[0][0]
            diff[1] += ideal_camera_coordinate[1][0]-camera_coordinate[1][0]
            diff[2] += ideal_camera_coordinate[2][0]-camera_coordinate[2][0]
        diff = np.array(diff)/4
        print('distortion:', diff)

    def extractQuadrilateralPixels(self, trigger=False):
        image = self.__realsense_client.get_rgb()
        p1 = np.array([[471.14640771, -118.09715487,  780.51498586, 1]]).T
        p2 = np.array([[471.17873673, 377.4285685, 780, 1]]).T
        p3 = np.array([[883.48525438, 377.76153914, 780.66399609, 1]]).T
        p4 = np.array([[883.87519164, -118.09715487, 780, 1]]).T
        Homoge_mat, Extrinsic_mat = self.cameraRegistration()
        test_target = []
        test_target.append(p1)
        test_target.append(p2)
        test_target.append(p3)
        test_target.append(p4)
        pixel_coordinates = []
        for i in range(4):
            camera_coordinate = np.dot(Extrinsic_mat, test_target[i])
            # print('Camera coordinate:', camera_coordinate)
            pixel_coordinate = np.dot(Homoge_mat, test_target[i])/camera_coordinate[2]
            pixel_coordinate = np.delete(pixel_coordinate, 2, axis=0)
            pixel_coordinate = pixel_coordinate.reshape(2)
            pixel_coordinates.append(pixel_coordinate)
        quadrilateral = pixel_coordinates
        # Create a mask image of the same shape as the input image
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # Reshape the quadrilateral points to a 2D array
        pts = np.array(quadrilateral).astype(np.int32)

        # Draw a filled polygon on the mask using the quadrilateral points
        cv2.fillPoly(mask, [pts], 255)

        # Bitwise AND the input image with the mask to extract the pixels within the quadrilateral
        extracted_pixels = cv2.bitwise_and(image, image, mask=mask)
        while trigger:
            print('Pixel coordinate:', pixel_coordinates)
            cv2.imshow('Extracted Pixels', extracted_pixels)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return extracted_pixels

    def PointToPoint(self, p_world, trigger=False):
        image = self.__realsense_client.get_rgb()
        Homoge_mat, Extrinsic_mat = self.cameraRegistration()
        camera_coordinate = np.dot(Extrinsic_mat, p_world)
        pixel_coordinate = np.dot(Homoge_mat, p_world / camera_coordinate[2])
        pixel_coordinate = np.delete(pixel_coordinate, 2, axis=0)
        pixel_coordinate = pixel_coordinate.reshape(2)
        while trigger:
            cv2.circle(image, (int(pixel_coordinate[0]), int(pixel_coordinate[1])), radius=5, color=(255,0,255), thickness=3)
            cv2.imshow('Point in image', image)
            cv2.waitKey(0)

if __name__ == '__main__':
    # ur_lft = ur_ma.ur3eusing_example(name='lft')
    # ur_lft.get_pos_rot_jnts_right_now(trigger=True)
    ur_rgt = ur_ma.ur3eusing_example(name='rgt')
    realsense_client = rs2.RealSense(color_frame_size=(1280, 720), depth_frame_size=(1280, 720),
                                color_frame_framerate=30, depth_frame_framerate=30,camera_id=0)
    # ur_rgt.get_pos_rot_jnts_right_now(trigger=True)
    # # computeDistortion(ur_rgt, realsense_client)
    ImageExtraction = RealityQueryExtraction(ur_rgt, realsense_client)
    center = [608.89888926, 45.84135886, 780.60438596, 1]
    ImageExtraction.PointToPoint(center, trigger=True)
    # ImageExtraction.extractQuadrilateralPixels(trigger=True)
