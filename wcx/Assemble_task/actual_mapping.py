import numpy as np
from scipy.spatial.transform import Rotation
import numpy as np
import wcx.utils1.move_action as ma
from wcx.utils1.rotate_matrix import matrix_generate as mg

def get_points_on_line_same_slope(point1, point2, new_point, distance, num_points):
    # Convert the points to NumPy arrays
    point1 = np.array(point1)
    point2 = np.array(point2)
    new_point = np.array(new_point)

    # Calculate the direction vector of the original line
    vector1 = point2 - point1

    # Calculate the new line
    new_line_point1 = new_point
    new_line_point2 = new_point + vector1

    # Calculate the unit vector in the direction of the line
    unit_vector = vector1 / np.linalg.norm(vector1)

    # Calculate the step size to increment along the line
    step_size = distance / np.linalg.norm(vector1)*10

    # Initialize the list to store the points
    points = []

    # Generate the points at the desired distance from the new point
    for i in range(num_points):
        # Calculate the distance from the new point
        current_distance = 200 + i * step_size

        # Calculate the point at the desired distance
        current_point = new_point - unit_vector * current_distance

        points.append(current_point)

    return points

def robot_matrix(target_position, center_position):


    # 机器人的起始坐标（世界坐标系）
    robot_start = np.array([0, 0, 0])  # 机器人起始坐标示例

    target_position = np.array([target_position]).reshape(-1)
    center_position = np.array([center_position]).reshape(-1)

    # 计算方向向量
    line_direction = center_position - target_position
    line_direction /= np.linalg.norm(line_direction)

    # Define the target direction as the line direction
    target_direction = line_direction

    # Calculate the rotation angle as the arccosine of the dot product between [0, 0, 1] and the line direction
    dot_product = np.dot(np.array([0, 0, 1]), target_direction)
    rotation_angle = np.arccos(dot_product)

    # Calculate the rotation axis as the cross product between [0, 0, 1] and the line direction
    rotation_axis = np.cross(np.array([0, 0, 1]), target_direction)
    rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize the rotation axis

    # Construct the rotation matrix using the axis-angle representation
    skew_symmetric_matrix = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                                      [rotation_axis[2], 0, -rotation_axis[0]],
                                      [-rotation_axis[1], rotation_axis[0], 0]])
    rotation_matrix = np.eye(3) + np.sin(rotation_angle) * skew_symmetric_matrix + \
                      (1 - np.cos(rotation_angle)) * np.dot(skew_symmetric_matrix, skew_symmetric_matrix)


    return  rotation_matrix, target_position


if __name__ == '__main__':
    target_pos = np.load('data/routine_pictures/routine_10.npy')
    lookatpos = [820, 120, 820]
    rbt = ma.ur3eusing_example(name='lft')

    rbt.axis_line_move('z', 20)
    # points = get_points_on_line_same_slope(target_pos, lookatpos, [626.00145943, 128.70595986, 808.53288402], 300, 20)
    # for point in points:
    #     rotation_matrix, target_position = robot_matrix(point, [626.00145943, 128.70595986, 808.53288402])
    #     print('point:', point, 'target_pos:', target_position)
    #     rot_z= mg(0, axis='z')
    #     rotation_matrix = np.dot(rotation_matrix, rot_z)
    #     rbt.specific_move(target_rot_mat=rotation_matrix, target_pos=target_position, move=True)
    #
    #     rbt.get_pos_rot_jnts_right_now(trigger=True)
    #     print('rotate_matrix:', rotation_matrix)

