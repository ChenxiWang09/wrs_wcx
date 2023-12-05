import cv2
import numpy as np


def point_extraction(temp_edge):
    n = len(temp_edge)
    m = len(temp_edge[0])
    points = []
    for i in range(n):
        for j in range(m):
            if temp_edge[i][j] == 255 or temp_edge[i][j] == 1:
                point = [i, j]
                points.append(point)
    return points, temp_edge


def line_fit(edge_map):
    m = len(edge_map)
    n = len(edge_map[0])
    points, temp_edge = point_extraction(edge_map)
    points = np.array(points)
    line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

    k = line[1] / line[0]
    b = line[3] - k * line[2]
    if b < 0:
        start_point = [int(-b / k), 0]
    else:
        start_point = [0, int(b)]
    end_point = [300, int(n / 2 * k + b)]

    return k, start_point, end_point


if __name__ == "__main__":
    i = 1
    temp_path = 'C:/wrs-cx/wcx/outline/data/temp_outline/' + str(i) + '.pgm'
    temp_edge = cv2.imread(temp_path, 0)

    line_fit(edge_map=temp_edge, trigger=True)
