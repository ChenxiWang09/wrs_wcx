import numpy as np
import cv2
import pylab as pl
n=8
coordinates = np.zeros((4, 2), np.int32)  # (4, 2):需要四个点，每个点都有x, y两个像素坐标；np.int32:转换为int型

counter = 0  # 用于计算点击、保存坐标的次数
#size detection stantard size
def mousePoints(event, x, y, flags, params):
    global counter  # 使外边定义的counter在本函数中转换全局变量
    if counter < 4 and event == cv2.EVENT_LBUTTONDOWN:
        # print(x, y)
        coordinates[counter] = x, y
        counter += 1
        print(coordinates)
# convertpoint_dictionary={1:np.array([[175., 437.],
#         [101., 366.],
#         [188., 311.],
#         [259., 370.]], dtype=float)}
# np.save('newtem.npy', convertpoint_dictionary)
cd={}
for i in range(1, n+1):
    inputpath = 'C:/Users/wangq/PycharmProjects/pythonProject/intelrealsense_wcx/templateimage/supersmallsize/' + str(
                i) + '.png'
    img = cv2.imread(inputpath)
    counter = 0
    while True:

        if counter == 4:
            break
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', img)
        cv2.circle(img, (coordinates[counter][0], coordinates[counter][1]), 15, [255, 0, 255], thickness=-1)
        cv2.setMouseCallback('img', mousePoints)
        if cv2.waitKey()==ord('q'):
            break
    cd[i]=coordinates
print(cd)
# cd={1:[[371,288],[241,222],[226,122],[458,198]],
#     2:[[180,69],[182,174],[154,170],[153,65]],
#     3:[[213,116],[215,227],[143,210],[140,95]],
#     4:[[116,213],[115,117],[268,110],[280,196]],
#     5:[[111,175],[85,125],[229,104],[295,150]],
#     6:[[180,232],[232,38],[260,36],[200,234]],
#     7:[[243,242],[77,127],[198,67.6],[352,155]],
#     8:[[130,307],[137,121],[236,67],[145,307]]
# }

np.save('pointselected_twochair_dictionary_supersmallsize.npy', cd)

# convertpoint_dictionary={1:np.array([0.535,0.285]),
#                          2:np.array([0.2,0.049]),
#                          3:np.array([0.2,0.161]),
#                          4:np.array([0.25,0.39]),
#                          5:np.array([0.33,0.4]),
#                          6:np.array([0.41,0.04]),
#                          7:np.array([0.269,0.25]),
#                          8:np.array([0.31,0.23])}
# np.save('objsize.npy', convertpoint_dictionary)