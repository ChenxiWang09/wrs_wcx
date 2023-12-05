import numpy as np
import cv2
import open3d as o3d
import wcx.utils1.realsense as rs
import copy
import math

def jugde(point1,point2,threshold):
    if math.fabs(point1[0]-point2[0])+math.fabs(point1[1]-point2[1])+math.fabs(point1[2]-point2[2])>threshold:
        return True
    else:
        return False

def screw_detection(gray, trigger = False):

    grayimg = copy.deepcopy(gray)
    edge = cv2.Canny(gray, 200, 100)
    # edge = cv2.Canny(edge, 200, 100)
    while trigger:
        cv2.imshow('grayimg', edge)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('s'):
            np.save('/home/hlabbunri/Desktop/wcx/wrs-wcx/wcx/outline/data/phx_edge_depth_' + str(4_5_8) + '.npy', edge_map_1)
            print('Successfully saving!')
    '''
    Hough_screw_holes_detection  to dp the first detection of holes is failed.
    '''
    circles = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=150, param2=12, minRadius=2, maxRadius=10)
    circles = np.uint16(np.around(circles))

    for i in circles[0,:]:
        cv2.circle(grayimg,(i[0], i[1]),i[2],(122,121,0),5)
        cv2.circle(grayimg,(i[0], i[1]),2,(122,122,0),5)

    while trigger:
        cv2.imshow('grayimg', grayimg)
        key = cv2.waitKey(0)
        if key == ord('s'):
            np.save('data/data_obj_gray_2.npy',grayimg)
            print('Successfully save!')
        if key == ord('q'):
            break
    return circles



if __name__ == '__main__':
    '''
    edge map generation method 1 , based on distance, not good effect 
    '''
    # distance_threshold=3
    # n=len(depthnparray_float32)
    # m=len(depthnparray_float32[0])
    # edge_map_1=np.zeros((n,m))
    #
    # for i in range(n-1):
    #     for j in range(m-1):
    #         if jugde(pcd[i*m+j],pcd[(i+1)*m+j],distance_threshold):
    #             edge_map_1[i+1][j]=1
    #         if jugde(pcd[i*m+j],pcd[i*m+j+1],distance_threshold):
    #             edge_map_1[i][j] = 1
    #         # if depthnparray_float32[i+1][j]-depthnparray_float32[i][j]>distance_threshold:
    #         #     edge_map_1[i+1][j]=1
    #         # if depthnparray_float32[i][j+1]-depthnparray_float32[i][j]>distance_threshold:
    #         #     edge_map_1[i][j]=1


