import cv2
import numpy as np
import pylab as pl
# Read source image.
im_src = cv2.imread('C:\\Users\qiumian\\Desktop\\mastercourse\\research\\homography transformation\\data\\1.jpg')
# Four corners of the book in source image
pts_src = np.array([[3.590000e+02 ,2.775000e+02 ], [5.900000e+01 ,1.261500e+03], [9.740000e+02 ,1.519500e+03], [1.019000e+03 ,3.255000e+02]])

# Read destination image.
im_dst = cv2.imread('C:\\Users\\qiumian\Desktop\\mastercourse\\research\\homography transformation\data\\1.png')
# Four corners of the book in destination image.
pts_dst = np.array([[6.800000e+01 ,3.090000e+02 ], [4.550000e+02 ,4.660000e+02 ], [6.840000e+02 ,2.610000e+02], [131 ,53 ]])

# Calculate Homography
h, status = cv2.findHomography(pts_src, pts_dst)

# Warp source image to destination based on homography
im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))

pl.figure(), pl.imshow(im_src[:, :, ::-1]), pl.title('src'),
pl.figure(), pl.imshow(im_dst[:, :, ::-1]), pl.title('dst')
pl.figure(), pl.imshow(im_out[:, :, ::-1]), pl.title('out'), pl.show()  # show dst