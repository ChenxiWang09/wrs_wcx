import cv2
import numpy as np
import pylab as pl

coordinates1 = np.zeros((4, 2), np.int32)  # (4, 2):需要四个点，每个点都有x, y两个像素坐标；np.int32:转换为int型
coordinates2 = np.zeros((4, 2), np.int32)  # (4, 2):需要四个点，每个点都有x, y两个像素坐标；np.int32:转换为int型
counter = 0  # 用于计算点击、保存坐标的次数
img1 = cv2.imread(r"C:\Users\wangq\PycharmProjects\pythonProject\intelrealsense_wcx\templateimage\another\3.png")
img2 = cv2.imread(r"C:\Users\wangq\PycharmProjects\pythonProject\intelrealsense_wcx\template1.jpg")

# 通过在图像中进行鼠标左键点击，获取鼠标点击位置的像素坐标（注意点击顺序：左上 -> 右上 -> 右下 -> 左下）
def mousePoints1(event, x, y, flags, params):
    global counter  # 使外边定义的counter在本函数中转换全局变量
    if counter < 5 and event == cv2.EVENT_LBUTTONDOWN:
        # print(x, y)
        coordinates1[counter] = x, y
        counter += 1
        print(coordinates1)

def mousePoints2(event, x, y, flags, params):
    global counter  # 使外边定义的counter在本函数中转换全局变量
    if counter < 5 and event == cv2.EVENT_LBUTTONDOWN:
        # print(x, y)
        coordinates2[counter] = x, y
        counter += 1
        print(coordinates2)
while True:

    if counter == 4:
        break
    cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
    cv2.imshow('img1', img1)
    cv2.circle(img1, (coordinates1[counter][0], coordinates1[counter][1]), 15, [255, 0, 255], thickness=-1)
    cv2.setMouseCallback('img1', mousePoints1)
    if cv2.waitKey()==ord('q'):
        break


im_src = img1
# Four corners of the book in source image
pts_src = np.float32([coordinates1[0], coordinates1[1], coordinates1[2], coordinates1[3]])


counter=0
while True:

    if counter == 4:
        break
    cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
    cv2.imshow('img2', img2)
    cv2.setMouseCallback('img2', mousePoints2)
    cv2.circle(img2, (coordinates2[counter][0], coordinates2[counter][1]), 15, [255, 0, 255], thickness=-1)
    if cv2.waitKey()==ord('q'):
        break

# Read destination image.
im_dst = img2
# Four corners of the book in destination image.
# pts_dst = np.float32([[ 61,  40],
#  [ 61, 263],
#  [ 18, 250],
#  [ 18,  27]])
pts_dst = np.float32([coordinates2[0], coordinates2[1], coordinates2[2], coordinates2[3]])


# Calculate Homography
h, status = cv2.findHomography(pts_src, pts_dst)

# Warp source image to destination based on homography
im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
pl.figure(), pl.imshow(im_src[:, :, ::-1]), pl.title('src'),
pl.figure(), pl.imshow(im_dst[:, :, ::-1]), pl.title('dst')
pl.figure(), pl.imshow(im_out[:, :, ::-1]), pl.title('out'), pl.show()  # show dst

