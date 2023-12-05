import cv2
import pylab as pl
n=8

for i in range(1, n+1):
        inputpath = 'C:/Users/wangq/PycharmProjects/pythonProject/intelrealsense_wcx/templateimage/smallsize/' + str(
                i) + '.png'
        outputpath = 'C:/Users/wangq/PycharmProjects/pythonProject/intelrealsense_wcx/templateimage/smallsize/' + str(
                i) + '.pgm'
        img=cv2.imread(inputpath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        pl.figure(), pl.imshow(edges[:, :]), pl.title('edgemap'), pl.show()  # show dst
        cv2.imwrite(outputpath, edges)