import cv2
import numpy as np
import math
def outline_extraction(edge_map, more_angle=False):
    m=len(edge_map)
    n=len(edge_map[0])
    outline=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            judge_top = 0
            judge_button = 0
            judge_left = 0
            judge_right = 0
            judge_top_left=1
            judge_top_right=1
            judge_button_left=1
            judge_button_right=1
            if more_angle:
                judge_top_left = 0
                judge_top_right = 0
                judge_button_left = 0
                judge_button_right = 0

            if edge_map[i][j]==255:
                for k in range(0,i):
                    if edge_map[k][j]==255:
                        judge_top=1
                        break
                for k in range(i+1,m):
                    if edge_map[k][j]==255:
                        judge_button=1
                for k in range(0,j):
                    if edge_map[i][k]==255:
                        judge_left=1
                        break
                for k in range(j+1,n):
                    if edge_map[i][k]==255:
                        judge_right=1
                        break
                if more_angle:
                    i_0=i
                    j_0=j
                    while i_0!=0 and i_0!=m-1 and j_0!=0 and j_0!=n-1:
                        if edge_map[i_0-1][j_0-1]==255:
                            judge_top_left=1
                            break
                        else:
                            i_0-=1
                            j_0-=1
                    i_0=i
                    j_0=j
                    while i_0!=0 and i_0!=m-1 and j_0!=0 and j_0!=n-1:
                        if edge_map[i_0+1][j_0+1]==255:
                            judge_button_right=1
                            break
                        else:
                            i_0+=1
                            j_0+=1
                    i_0=i
                    j_0=j
                    while i_0!=0 and i_0!=m-1 and j_0!=0 and j_0!=n-1:
                        if edge_map[i_0-1][j_0+1]==255:
                            judge_top_right=1
                            break
                        else:
                            i_0-=1
                            j_0+=1
                    i_0=i
                    j_0=j
                    while i_0!=0 and i_0!=m-1 and j_0!=0 and j_0!=n-1:
                        if edge_map[i_0+1][j_0-1]==255:
                            judge_button_left=1
                            break
                        else:
                            i_0+=1
                            j_0-=1

                if judge_top == 0 or judge_button == 0 or judge_left == 0 or judge_right == 0:
                    outline[i][j]=255
                if more_angle:
                    if judge_button_right == 0 or judge_button_left == 0 or judge_top_right == 0 or judge_top_left == 0:
                        outline[i][j] = 255
    return outline




if __name__ == "__main__":

    temp = cv2.imread('C:/wrs-cx/wcx/outline/data/temp/5.png')
    gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    edge_map = cv2.Canny(gray, 20, 160)
    # edge_map = cv2.imread('C:/wrs-cx/wcx/outline/data/temp/5.jpg', 0)

    outline=outline_extraction(edge_map)
    # amount = 7
    # for i in range(7,amount+1):
    #     temp=cv2.imread('C:/wrs-cx/wcx/outline/data/temp/'+str(i)+'.png')
    #     gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)  # 灰度图像
    #     # r,edgemap = cv2.threshold(gray,120,255,cv2.THRESH_TOZERO_INV)
    #     edge_map_1 = cv2.Canny(gray, 200, 250)
    #     edge_map = outline_extraction(edge_map_1)

