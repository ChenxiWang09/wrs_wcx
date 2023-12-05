import cv2
import numpy as np
import wcx.utils1.phoxi as phx
import os
def calib_camera(img_gray, pattern_size=(7, 5), draw_points=False):
    """
    calibrate camera
    :param calib_dir: str
    :param pattern_size: (x, y), the number of points in x, y axes in the chessboard
    :param draw_points: bool, whether to draw the chessboard points
    """
    # store 3d object points and 2d image points from all the images
    object_points = []
    image_points = []

    # 3d object point coordinate
    xl = np.linspace(0, pattern_size[0], pattern_size[0], endpoint=False)
    yl = np.linspace(0, pattern_size[1], pattern_size[1], endpoint=False)
    xv, yv = np.meshgrid(xl, yl)
    object_point = np.insert(np.stack([xv, yv], axis=-1), 2, 0, axis=-1).astype(np.float32).reshape([-1, 3])

    # set termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 26, 0.001)
    img = img_gray

    # find chessboard points
    ret, corners = cv2.findChessboardCorners(img_gray, patternSize=pattern_size)
    if ret:
        # add the corresponding 3d points to the summary list
        object_points.append(object_point)
        # if chessboard points are found, refine them to SubPix level (pixel location in float)
        corners_refined = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
        # add the 2d chessboard points to the summary list
        image_points.append(corners.reshape([-1, 2]))
        # visualize the points
        if draw_points:
            cv2.drawChessboardCorners(img, pattern_size, corners_refined, ret)
            if img.shape[0] * img.shape[1] > 1e6:
                scale = round((1. / (img.shape[0] * img.shape[1] // 1e6)) ** 0.5, 3)
                img_draw = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            else:
                img_draw = img

            cv2.imshow('img', img_draw)
            cv2.waitKey(0)

    assert len(image_points) > 0, 'Cannot find any chessboard points, maybe incorrect pattern_size has been set'
    # calibrate the camera, note that ret is the rmse of reprojection error, ret=1 means 1 pixel error
    reproj_err, k_cam, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points,
                                                                       image_points,
                                                                       img_gray.shape[0:2],
                                                                       None,
                                                                       None,
                                                                       criteria=criteria)

    return k_cam, dist_coeffs

if __name__ == '__main__':

    phx_client=phx.Phoxi(host="127.0.0.1:18300")
    grayimg, depthnparray_float32, pcd = phx_client.getalldata()
    n = len(grayimg)
    m = len(grayimg[0])

    while True:
        cv2.imshow('img',grayimg)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('data/chessboard_ca.pgm', grayimg)
    calib_dir = 'data/chessboard_ca.pgm'
    k_cam, dist_coeffs = calib_camera(grayimg,draw_points=True)
    print(k_cam)
    print(dist_coeffs)
