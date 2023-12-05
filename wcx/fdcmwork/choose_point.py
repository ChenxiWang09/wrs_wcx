import cv2
import copy
import matplotlib.pyplot as plt
import numpy as np


def top_right_botton_left(edge_map, trigger=False):
    corners = cv2.goodFeaturesToTrack(edge_map, 100, 0.005, 5)
    points = []
    for item in corners:
        x, y = item.ravel()
        point = [y, x]
        points.append(point)
    p_min = [10000, 10000]
    p_max = [0, 0]
    for item in points:
        if item[0]+item[1] < p_min[0]+p_min[1]:
            p_min = item
        if item[0]+item[1] > p_max[0]+p_max[1]:
            p_max = item
    corner=[]
    corner.append(p_min)
    corner.append(p_max)
    corner = np.array(corner)
'/home/hlabbunri/Desktop/wcx/wrs-wcx/wcx/fdcmwork/data/temp/temp_outline/'

'/home/hlabbunri/Desktop/wcx/wrs-wcx/wcx/fdcmwork/data/result/0124/rotate_data'
    if trigger == True:
        edge_copy = copy.deepcopy(edge_map)
        for i in range(2):
            x = corner[i][1]
            y = corner[i][0]
            corner[i][0] = x
            corner[i][1] = y
            # 5 图像展示
            cv2.circle(edge_copy, (int(x), int(y)), 10, (0, 0, 255), -1)
        while True:
            cv2.imshow('screw_hole',edge_copy)
            key=cv2.waitKey(1)
            if key == ord('q'):
                break
    return corner
if __name__=='__main__':
    for i in range(1,5):
        for j in range(1,9):
            edge_map = cv2.imread('/home/hlabbunri/Desktop/wcx/wrs-wcx/wcx/fdcmwork/data/result/0124/rotate_data/'+str(i)+'_'+str(j)+'.pgm', 0)
            top_right_botton_left(edge_map, trigger=False)
            n