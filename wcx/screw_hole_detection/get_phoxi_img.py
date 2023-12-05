import numpy as np
import cv2
import open3d as o3d
import utils1.phoxi as phoxi
import copy
import math

if __name__ == '__main__':
    phxi_host = "127.0.0.1:18300"
    phxi_client = phoxi.Phoxi(host=phxi_host)
    grayimg, depthnparray_float32, pcd = phxi_client.getalldata()
    while True:
        cv2.imshow('gray_img',grayimg)
        key=cv2.waitKey(0)
        if key==ord('q'):
            cv2.destroyWindow('gray_image')
            break
        if key==ord('s'):
            cv2.imwrite('data/gray_img_1.bmp',grayimg)
            print('Successfully saving!')
