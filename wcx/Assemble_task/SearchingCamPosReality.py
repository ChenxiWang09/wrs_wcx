import math

import wcx.Assemble_task.ImageExtraction as ImageExtraction
import numpy as np
import wcx.utils1.rotate_matrix as rot_m
import wcx.utils1.move_action as ur_ma
import wcx.utils1.realsense as rs2
import cv2
import wcx.utils1.sphere_coordinate as sphere_coordinate
import wcx.Assemble_task.ImageExtraction as ImageExtraction
import wcx.utils1.canny as canny
import wcx.Assemble_task.gradient_descent as gradient_descent

class FDCMGradientDescent():

    def __init__(self, ur_rgt, realsense_client, center):

        self.__ur_rgt = ur_rgt
        self.__realsense_client = realsense_client
        self.__center = center
        self.__image_extraction = ImageExtraction.RealityQueryExtraction(ur_rgt, realsense_client)

    def transformation_matrix(self, p2):
        p1 = self.__center
        w_prime = np.array(p1) - np.array(p2)
        w_prime /= np.linalg.norm(w_prime)
        v_prime = np.cross(w_prime, np.array([0, 0, 1]))
        v_prime /= np.linalg.norm(v_prime)
        u_prime = np.cross(w_prime, v_prime)
        rot_mat = np.zeros((3,3))
        rot_mat[:3, 0] = v_prime
        rot_mat[:3, 1] = u_prime
        rot_mat[:3, 2] = w_prime

        T = np.zeros((4, 4))
        T[:3, 0] = v_prime
        T[:3, 1] = u_prime
        T[:3, 2] = w_prime
        T[:3, 3] = p2
        T[3, 3] = 1

        return T, rot_mat

    def image_taken(self, p2, sequence,  radius_range=[380, 600]):

        T, rot_mat = self.transformation_matrix(p2)
        # print("rotmat:", rot_mat)
        if self.__ur_rgt.spherical_move(target_rot_mat=rot_mat, target_pos=p2, center=self.__center,
                                        radius_range=radius_range) is False:
            return False

        img = self.__image_extraction.extractQuadrilateralPixels()
        # cv2.imwrite('data/watch_image/screenshot_'+str(sequence)+'.png', img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_edge = cv2.Canny(img_gray, 100, 50)
        cv2.imwrite('data/watch_image/screenshot_'+str(sequence)+'.pgm', img_edge)

        return True



    def gradient_descent(self, epsilon = 2, learning_rate = 0.1, shreshold = [0.25, 0.1], radius_range=[350, 600]):

        epsilon = epsilon*math.pi/180

        start_jnts = np.load('data/RgtRealityCamPosSearching_jnts.npy')
        self.__ur_rgt.jnts_based_move(start_jnts)
        ee_pos, ee_rot_mat, jnts = self.__ur_rgt.get_pos_rot_jnts_right_now(data_need=True)
        time = 0
        p_origin = ee_pos
        while time < 20:
            p_polar = sphere_coordinate.modify_spherical_coordinates(p_origin, self.__center, dradius=-1, dtheta=epsilon,
                                                                     dphi=0)
            p_azimuthal = sphere_coordinate.modify_spherical_coordinates(p_origin, self.__center, dradius=-1, dtheta=0,
                                                                         dphi=epsilon)

            time_origin = 0
            print('Origin Projecting..')
            while self.image_taken(p_origin, sequence=0, radius_range=[350, 600]) is False and time_origin < 5:
                time_origin += 1

            rgb = self.__realsense_client.get_rgb()
            ee_pos, ee_rot_mat, jnts = self.__ur_rgt.get_pos_rot_jnts_right_now(data_need=True)
            cv2.imwrite('data/routine_pictures/reality/routine_'+str(time)+'.png', rgb)
            np.save('data/routine_pictures/reality/routine_'+str(time)+'.npy', jnts)

            print('Theta Projecting..')
            time_theta = 0
            while self.image_taken(p_polar, sequence=1, radius_range=[350, 600]) is False and time_theta < 10:
                p_polar = sphere_coordinate.modify_spherical_coordinates(p_polar, self.__center, dradius=-1,
                                                                         dtheta=0.1*epsilon,
                                                                         dphi=0)
                time_theta += 1

            print('p_polar:', p_polar)
            print('Phi Projecting..')
            time_phi = 0
            while self.image_taken(p_azimuthal, sequence=2, radius_range=[350, 600]) is False and time_phi < 10:
                p_azimuthal = sphere_coordinate.modify_spherical_coordinates(p_azimuthal, self.__center, dradius=-1,
                                                                         dtheta=epsilon,
                                                                         dphi=0)
                time_phi += 1
            print('p_phi:', p_azimuthal)
            new_point, state, cost = gradient_descent.compute_gradients(initial_point=p_origin, center_point=self.__center,
                                                                  shreshold=shreshold, dtheta=epsilon + time_theta * 0.1*epsilon,
                                                                  dphi=epsilon + time_theta * 0.1*epsilon, learning_rate=learning_rate)
            np.save('data/routine_pictures/reality/cost_'+str(time)+'.npy', cost)
            p_origin = new_point
            if state:
                print("Success find the target pos in "+str(time+1)+"times!")
                time += 1
                break
            else:
                time += 1
                print('Continue searching...')
        self.image_taken(p_origin, 0)
        rgb = self.__realsense_client.get_rgb()
        ee_pos, ee_rot_mat, jnts = self.__ur_rgt.get_pos_rot_jnts_right_now(data_need=True)
        cv2.imwrite('data/routine_pictures/reality/routine_' + str(time) + '.png', rgb)
        np.save('data/routine_pictures/reality/routine_' + str(time) + '.npy', jnts)

    def set_start_pos(self):
        ee_pos, ee_rot_mat, jnts = self.__ur_rgt.get_pos_rot_jnts_right_now(data_need=True)
        T, R = FDCM_GD.transformation_matrix(ee_pos)
        self.__ur_rgt.spherical_move(target_rot_mat=R, target_pos=ee_pos, center=center, radius_range=[400, 600])
        self.__ur_rgt.set_xx_pos(name='RgtRealityCamPosSearching')

if __name__ == '__main__':
    center = [608.89888926,  45.84135886, 780.60438596]
    ur_rgt = ur_ma.ur3eusing_example(name='rgt')
    ur_rgt.get_pos_rot_jnts_right_now(trigger=True)
    tgt_pos = [794.238991566393, -223.7921109409798, 890.2092005959655]

    # ee_pos, ee_rot_mat, jnts = ur_rgt.get_pos_rot_jnts_right_now(trigger=True, data_need=True)
    # print(jnts)
    #
    realsense_client = rs2.RealSense(color_frame_size=(1280, 720), depth_frame_size=(1280, 720),
                                color_frame_framerate=30, depth_frame_framerate=30,camera_id=0)
    FDCM_GD = FDCMGradientDescent(ur_rgt=ur_rgt, realsense_client=realsense_client, center=center)
    # FDCM_GD.image_taken(p2=tgt_pos, sequence=4,  radius_range=[320, 600])
    # FDCM_GD.set_start_pos()
    FDCM_GD.gradient_descent()
