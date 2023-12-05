import cv2
import numpy as np


def get_input(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:

        points.append([x,y])
        print(points)


def input(img):
    global points
    points = []
    cv2.namedWindow('img')
    cv2.setMouseCallback('img', get_input)
    cv2.imshow('img', img)
    while True:
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
    points = np.array(points)
    return points


# if __name__=='__main__':
