# import matlab
import matlab.engine
import cv2
from PIL import Image



img=cv2.imread(r'C:\Users\wangq\PycharmProjects\pythonProject\intelrealsense_wcx\Figure_2.png')
sav='1.jpg'
cv2.imwrite(sav,img)
eng = matlab.engine.start_matlab()
k=2
number, value=eng.chamfermatching_nopic_forpython(sav, k, nargout=2)
print(number)
print(value)