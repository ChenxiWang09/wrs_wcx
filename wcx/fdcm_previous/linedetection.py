import cv2
import numpy as np

def main(img):
    # img = cv2.imread("1.jpg")
    cv2.imshow('origin_img', img)
    height = img.shape[0]  # 高度
    width = img.shape[1]  # 宽度
    cut_img = img

    gray = cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray_img', gray)
    cv2.waitKey(0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    cv2.imshow('edgemap',edges)
    cv2.waitKey(0)
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 118)
    result = cut_img.copy()
    minLineLength = 30  # height/32
    maxLineGap = 10  # height/40
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength, maxLineGap)

    for item in lines:
        x1=item[0][0]
        y1=item[0][1]
        x2=item[0][2]
        y2=item[0][3]

        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
