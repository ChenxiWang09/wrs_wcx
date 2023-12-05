import numpy as np
import matplotlib.pyplot as plt
import wcx.utils1.move_action as ma
from mpl_toolkits.mplot3d import Axes3D

def fibonacci_half_sphere(center=[0,0,0], num_points=20, radius=1):
    cx, cy, cz = center
    points = []

    for _ in range(num_points):
        # Random radius

        # Inclination (theta) between 0 and π/2 to ensure Z > 0
        theta = np.random.uniform(0, np.pi / 2)
        # Azimuth (phi) between 0 and 2π
        phi = np.random.uniform(0, 2 * np.pi)

        # Convert spherical to Cartesian coordinates
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)

        # Translate point to the desired center
        x += cx
        y += cy
        z += cz

        points.append((x, y, z))

    return points

def transformation_matrix(p1=[0,0,0], p2=[0,0,0]):

    w_prime = np.array(p1) - np.array(p2)
    w_prime = w_prime.astype(np.float64)
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

def plot_coordinate_systems(center, points, transformation_matrices):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制球心
    ax.scatter(center[0], center[1], center[2], color='k', s=100, label='球心')

    # 遍历每个点及其变换矩阵
    for point, matrix in zip(points, transformation_matrices):
        # 绘制点
        ax.scatter([point[0]], [point[1]], [point[2]], color='b')

        # 为每个坐标轴绘制一条线
        for i, color in zip(range(3), ['r', 'g', 'b']):  # X轴为红色，Y轴为绿色，Z轴为青色
            start = point
            vector = matrix[:3, i].T
            end = point + vector # 取矩阵的前三个元素（忽略齐次坐标）
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.legend()
    plt.show()


def IkTest(center=[400, 100, 900]):
    points = fibonacci_half_sphere(center=center, num_points=50, radius=150)

    Ts = []
    for point in points:
        T, rot_mat = transformation_matrix(p1=center, p2=point)
        Ts.append(T)
    #
    # plot_coordinate_systems(center, points, transformation_matrices=Ts)

    sorted_Ts = sorted(Ts, key=lambda x: (x[0, 3], x[1, 3]))
    check_Ts = np.array(sorted_Ts)
    n = len(Ts)
    rot_mats = []
    points = []
    for i in range(n):
        rot_mats.append(sorted_Ts[i][:3, :3])
        points.append(sorted_Ts[i][:3, 3])
    points = np.array(points)
    ur_lft = ma.ur3eusing_example(simulation=True)
    ur_lft.planning(tgt_poss=points, tgt_rot_mats=rot_mats)

def Real_Movement(center=[400, 100, 900]):
    points = fibonacci_half_sphere(center=center, num_points=50, radius=150)

    Ts = []
    for point in points:
        T, rot_mat = transformation_matrix(p1=center, p2=point)
        Ts.append(T)
    #
    # plot_coordinate_systems(center, points, transformation_matrices=Ts)

    sorted_Ts = sorted(Ts, key=lambda x: (x[0, 3], x[1, 3]))
    check_Ts = np.array(sorted_Ts)
    n = len(Ts)
    rot_mats = []
    points = []
    for i in range(n):
        rot_mats.append(sorted_Ts[i][:3, :3])
        points.append(sorted_Ts[i][:3, 3])
    points = np.array(points)
    ur_lft = ma.ur3eusing_example(simulation=True)
    for i in range(n):
        if ur_lft.specific_move(target_rot_mat=rot_mats[i], target_pos=points[i]):
            






IkTest()


