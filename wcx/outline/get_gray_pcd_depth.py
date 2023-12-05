import utils1.phoxi as phoxi
import utils1.realsense as rs
import cv2
import numpy as np

def get_data(num, trigger=False):

    realsense_client: object = rs.RealSense()
    while trigger:
        rs_depth = realsense_client.get_depth()
        rs_rgb = realsense_client.get_rgb()
        cv2.imshow('rs image' , rs_rgb)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite('data/rs_gray_'+str(num)+'.png', rs_rgb)
            np.save('data/rs_depth_' + str(num) + '.npy', rs_depth)
            print('Successfully save rs image ' + str(num) + '!')

    phxi_host = "127.0.0.1:18300"
    phoxi_client = phoxi.Phoxi(host=phxi_host)
    phx_gray, phx_depthnparray_float32, phx_pcd = phoxi_client.getalldata()
    while trigger:
        cv2.imshow('phx image' , phx_gray)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite('data/phx_gray_'+str(num)+'.png', phx_gray)
            np.save('data/phx_depth_' + str(num) + '.npy', phx_depthnparray_float32)
            np.save('data/phx_pcd_' + str(num) + '.npy', phx_pcd)
            print('Successfully save phx image ' + str(num) + '!')

    return rs_rgb,rs_depth,phx_gray,phx_depthnparray_float32,phx_pcd
if __name__=="__main__":
    rs_rgb, rs_depth, phx_gray, phx_depthnparray_float32, phx_pcd = get_data('4_5_8_d',trigger=True)




