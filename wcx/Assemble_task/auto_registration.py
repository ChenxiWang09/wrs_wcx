import wcx.utils1.realsense as rs
import wcx.utils1.move_action as ur3e_ma
import numpy as np
import wcx.utils1.realsense as rs
import wcx.utils1.mouse_input as mouse_input
import wcx.utils1.rotate_matrix as rotate_matrix

import math



if __name__ == '__main__':
    standard_coordinate = np.array([[686.95225614, 139.83053244, 794.90646307]])
    x_diff_final = 0
    y_diff_final = 0
    z_diff_final = 0
    error = 50
    ur3e_lft = ur3e_ma.ur3eusing_example()

    # ur3e_lft.set_xx_pos(name='registration')
    # pos, rot_mat, jnts = ur3e_lft.get_pos_rot_jnts_right_now(data_need=True)
    # print(pos)
    ur3e_lft.move_to_xx_pos(name='registration')
    realsense_client = rs.RealSense(color_frame_size=(1280, 720), depth_frame_size=(1280, 720),
                                    color_frame_framerate=30, depth_frame_framerate=30)
    realsense_client.waitforframe()
    # pos, rot_mat, jnts = ur3e_lft.get_pos_rot_jnts_right_now(data_need=True)
    # print(pos)
    while error > 5:
        x_diff = 0
        y_diff = 0
        z_diff = 0
        ur3e_lft.move_to_xx_pos(name='registration')
        ur3e_lft.axis_line_move(-40, 'x', move=True)
        for i in range(8):
            ur3e_lft.axis_line_move(10, 'x', move=True)
            camera_coordinate = realsense_client.getcenter_two_aruco_maker()
            pos, rot_mat, jnts = ur3e_lft.get_pos_rot_jnts_right_now(data_need=True)
            camera_coordinate = np.hstack((camera_coordinate, np.ones((1, 1)))).T
            # image_coordinate = mouse_input.input(rgb)
            r_mat = np.dot(rotate_matrix.matrix_generate(180, 'z'), rot_mat)
            t_mat = np.array([[pos[0], pos[1], pos[2]]]).T
            homo_mat = np.hstack((r_mat, t_mat))
            homo_mat = np.vstack((homo_mat, np.array([[0, 0, 0, 1]])))
            world_coordinate = np.dot(homo_mat, camera_coordinate)
            print(world_coordinate)
            x_diff = x_diff + (standard_coordinate[0][0]-world_coordinate[0][0])
            y_diff = y_diff + (standard_coordinate[0][1]-world_coordinate[1][0])
            z_diff = z_diff + (standard_coordinate[0][2]-world_coordinate[2][0])

        ur3e_lft.move_to_xx_pos(name='registration')
        ur3e_lft.axis_line_move(-40, 'y', move=True)
        for i in range(8):
            ur3e_lft.axis_line_move(10, 'y', move=True)

            camera_coordinate = realsense_client.getcenter_two_aruco_maker()
            pos, rot_mat, jnts = ur3e_lft.get_pos_rot_jnts_right_now(data_need=True)
            camera_coordinate = np.hstack((camera_coordinate, np.ones((1, 1)))).T

            # image_coordinate = mouse_input.input(rgb)
            r_mat = np.dot(rotate_matrix.matrix_generate(180, 'z'), rot_mat)
            t_mat = np.array([[pos[0], pos[1], pos[2]]]).T
            homo_mat = np.hstack((r_mat, t_mat))
            homo_mat = np.vstack((homo_mat, np.array([[0, 0, 0, 1]])))
            world_coordinate = np.dot(homo_mat, camera_coordinate)
            print(world_coordinate)
            x_diff = x_diff + (standard_coordinate[0][0]-world_coordinate[0][0])
            y_diff = y_diff + (standard_coordinate[0][1]-world_coordinate[1][0])
            z_diff = z_diff + (standard_coordinate[0][2]-world_coordinate[2][0])

        x_diff = x_diff/16
        y_diff = y_diff/16
        z_diff = z_diff/16
        standard_coordinate = standard_coordinate - np.array([[x_diff, y_diff, z_diff]])
        error = math.sqrt(x_diff**2 + y_diff**2 + z_diff**2)
        x_diff_final += x_diff
        y_diff_final += y_diff
        z_diff_final += z_diff
        print('diff:', x_diff, y_diff, z_diff)

    print('final diff:', x_diff_final, y_diff_final, z_diff_final)











