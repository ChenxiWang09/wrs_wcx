import wcx.utils1.furniture as furniture
import wcx.utils1.phoxi as phoxi
import wcx.utils1.realsense as rs
import cv2

if __name__=='__main__':

    phxi_host = "127.0.0.1:18300"
    phxi_client = phoxi.Phoxi(host=phxi_host)
    realsense_client: object = rs.RealSense()
    rs_id=1
    phx_grayimg, phx_depthnparray_float32, phx_pcd = phxi_client.getalldata()
    num = 1
    time = 5
    while True:
        rs_depth = realsense_client.get_depth()
        rs_rgb = realsense_client.get_rgb()
        cv2.imshow('rs image' , rs_rgb)
        cv2.imshow('phx_gray',phx_grayimg)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('data/query/0201/'+str(num)+'/rs_rgb_'+'_'+str(time)+'_'+str(rs_id)+'.jpg', rs_rgb)
            print('Successfully save '+str(rs_id)+' camera rgb image ' + str(num) + '!')
        elif key ==ord('g'):
            cv2.imwrite('data/query/0201/'+str(num)+'/phx_rgb_'+str(time) + '.jpg', phx_grayimg)
            print('Successfully save phx image ' + str(num) + '!')

        elif key ==ord('1'):
            realsense_client.change_camera(camera_id=0)
            rs_id = 1
        elif key ==ord('2'):
            realsense_client.change_camera(camera_id=1)
            rs_id = 2
        elif key ==ord('3'):
            realsense_client.change_camera(camera_id=2)
            rs_id = 3



