import time

import numpy as np
import wcx.utils1.rotate_matrix as rotate_matrix
import wcx.utils1.SBD_screw_holes_detection as SBD
import wcx.utils1.realsense as rs
import wcx.utils1.move_action as ur3e_ma

def depth_computation(realsense_client, screw_hole):

    x = screw_hole[0]
    y = screw_hole[1]
    r = screw_hole[2]
    camera_coordinate = realsense_client.get_3d_camera_coordinate(depth_pixel=[x, y])
    if np.any(camera_coordinate != 0):
        return camera_coordinate
    else:
        angles = np.linspace(0, 2 * np.pi, 100)
        r1 = x + r * np.cos(angles)
        r2 = y + r * np.sin(angles)

        r1 = np.round(r1).astype(int)
        r2 = np.round(r2).astype(int)
        camera_coordinates = []
        for i in range(len(r1)):
            camera_coordinate = realsense_client.get_3d_camera_coordinate(depth_pixel=[r1[i],r2[i]])
            camera_coordinates.append(camera_coordinate)
        camera_coordinates = np.array(camera_coordinates)
        column_averages = np.mean(camera_coordinates, axis=0)
        return column_averages

def screw_holes_size(screw_holes, size):
    new_screw_holes = []
    for item in screw_holes:
        if item[2] > size:
            new_screw_holes.append(item)

    return  new_screw_holes

def sort_coordinates(robot_coordinates):
    length = len(robot_coordinates)
    x_average = 0
    y_average = 0
    for i in range(length):
        x_average += robot_coordinates[i][0]
        y_average += robot_coordinates[i][1]

    x_average = x_average/length
    y_average = y_average/length
    symbol_axis = np.zeros((4,2))
    for i in range(length):
        if robot_coordinates[i][0] > x_average:
            symbol_axis[i][0] = 1
        else:
            symbol_axis[i][0] = 0
        if robot_coordinates[i][1] > y_average:
            symbol_axis[i][1] = 1
        else:
            symbol_axis[i][1] = 0
    rotate = np.zeros(4)
    for i in range(length):
        if symbol_axis[i][0] == 1 and symbol_axis[i][1] == 1:
            rotate[i] = 1

        elif symbol_axis[i][0] == 0 and symbol_axis[i][1] == 0:
            rotate[i] = 1
        else:
            rotate[i] = -1

    return rotate







if __name__ == '__main__':
    # height for assemble 7 and 4 825mm
    ur3e_lft = ur3e_ma.ur3eusing_example()
    # ur3e_lft.set_xx_pos()
    ur3e_lft.move_to_xx_pos()

    pos, rot_mat, jnts = ur3e_lft.get_pos_rot_jnts_right_now(data_need=True)
    realsense_client = rs.RealSense(color_frame_size=(1280, 720), depth_frame_size=(1280, 720),
                     color_frame_framerate=30, depth_frame_framerate=30)
    realsense_client.waitforframe()
    gray = realsense_client.get_gray()

    screw_holes = SBD.detection(gray, trigger=True)
    screw_holes = screw_holes_size(screw_holes, size=11)

    screw_holes_coordinates = []
    amount = len(screw_holes)
    for i in range(amount):
        screw_holes_coordinate = depth_computation(realsense_client, screw_holes[i])
        screw_holes_coordinates.append(screw_holes_coordinate)

    one = np.ones((amount, 1))
    screw_holes_coordinates = np.array(screw_holes_coordinates)
    screw_holes_coordinates = screw_holes_coordinates.reshape(amount, 3)

    screw_holes_coordinates = np.hstack((screw_holes_coordinates, one)).T



    r_mat = np.dot(rotate_matrix.matrix_generate(180, 'z'), rot_mat)
    t_mat = np.array([[pos[0], pos[1], pos[2]]]).T
    homo_mat = np.hstack((r_mat, t_mat))
    homo_mat = np.vstack((homo_mat, np.array([[0, 0, 0, 1]])))

    x_diff = -29
    y_diff = 57
    z_diff = 30.40

    screw_holes_robot_coordinates = np.dot(homo_mat, screw_holes_coordinates)
    screw_holes_robot_coordinates = screw_holes_robot_coordinates.T
    screw_holes_robot_coordinates = np.delete(screw_holes_robot_coordinates, 3, axis=1)

    print('screw_holes_robot_coordinates:', screw_holes_robot_coordinates)
    rotate = sort_coordinates(screw_holes_robot_coordinates)
    rotate_count = 0
    # for coordinate in screw_holes_robot_coordinates:
    #     coordinate[0] += x_diff
    #     coordinate[1] += y_diff
    #     coordinate[2] = 900
    #     print('screw_holes_robot_coordinate:', coordinate)
    #     ur3e_lft.axis_line_move([coordinate, rot_mat], axis='free_move', move=True)
    #     ur3e_lft.rotate_move(-45*rotate[rotate_count], 'z', move=True)
    #     rotate_count += 1
    #     ur3e_lft.axis_line_move(-75, axis='z', move=True)
    #     # ur3e_lft.gripper(action='close')
    #
    #
    #     ur3e_lft.move_to_xx_pos()
    #     # ur3e_lft.gripper(action='open')
    #     time.sleep(1)
    #
    # print('screw_holes_robot_coordinate:', screw_holes_robot_coordinates)
    for coordinate in screw_holes_robot_coordinates:
        coordinate[0] += x_diff
        coordinate[1] += y_diff
        coordinate[2] = 900
        print('screw_holes_robot_coordinate:', coordinate)
        ur3e_lft.axis_line_move([coordinate, rot_mat], axis='free_move', move=True)
        ur3e_lft.rotate_move(-45*rotate[rotate_count], 'z', move=True)
        rotate_count += 1
        ur3e_lft.axis_line_move(-22.5, axis='z', move=True)
        for i in range(70):
            ur3e_lft.gripper(action='close')
            time.sleep(0.05)
            ur3e_lft.gripper(action='open')
        ur3e_lft.axis_line_move(+16.5, axis='z', move=True)
        ur3e_lft.move_to_xx_pos()
        time.sleep(1)






