import math
import time

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


def transformation_matrix(p1,  p2):
    w_prime = np.array(p1) - np.array(p2)
    w_prime /= np.linalg.norm(w_prime)
    v_prime = np.cross(w_prime, np.array([0, 0, 1]))
    v_prime /= np.linalg.norm(v_prime)
    u_prime = np.cross(w_prime, v_prime)
    rot_mat = np.zeros((3, 3))
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

def Routine_Display(ur_rgt, jnts_directory, jnts_amount, epsilon=2):
    center = [608.89888926, 45.84135886, 780.60438596]
    epsilon = epsilon * math.pi / 180
    for i in range(jnts_amount):
        jnts = np.load(jnts_directory+'routine_'+str(i)+'.npy')
        ur_rgt.jnts_based_move(jnts)
        if(i==0):
            time.sleep(2)
        ee_pos, ee_rot_mat, ee_jnts = ur_rgt.get_pos_rot_jnts_right_now(data_need=True,trigger=True)
        p_polar = sphere_coordinate.modify_spherical_coordinates(ee_pos, center, dradius=-1, dtheta=epsilon,
                                                                 dphi=0)
        p_azimuthal = sphere_coordinate.modify_spherical_coordinates(ee_pos, center, dradius=-1, dtheta=0,
                                                                     dphi=epsilon)
        print(p_polar, p_azimuthal)
        T, polar = transformation_matrix(center, p_polar)
        T, azimuthal = transformation_matrix(center, p_azimuthal)
        ur_rgt.spherical_move(target_rot_mat=polar, target_pos=p_polar, center=center, radius_range=[350, 600])
        ur_rgt.jnts_based_move(jnts)
        ur_rgt.spherical_move(target_rot_mat=azimuthal, target_pos=p_azimuthal, center=center, radius_range=[350, 600])
        time.sleep(4)

    tgt_pos = [794.238991566393, -223.7921109409798, 890.2092005959655]
    T, tgt_rot_mat = transformation_matrix(center, tgt_pos)
    ur_rgt.spherical_move(target_rot_mat=tgt_rot_mat, target_pos=tgt_pos, center=center, radius_range=[320, 600])





if __name__ == '__main__':

    ur_rgt = ur_ma.ur3eusing_example(name='rgt')

    # directory = '/home/hlabbunri/Desktop/chenxi wang/wrs-wcx/wcx/Assemble_task/data/routine_pictures/reality/t4/'
    # Routine_Display(ur_rgt, directory, 11)