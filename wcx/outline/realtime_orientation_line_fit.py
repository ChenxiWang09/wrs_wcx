import cv2
import math
from wcx.utils1 import furniture
import imutils

def rotation_matching

if __name__ == "__main__":

    '''
    offline project
    '''
    query_rgb=cv2.imread(r'C:\wrs-cx\wcx\outline\data\query\6.jpg')
    q_fur_5 = furniture.furniture_image(query_rgb, 'query', '6')
    temp_rgb = cv2.imread(r'C:\wrs-cx\wcx\outline\data\temp\5.png')
    t_fur_5 = furniture.furniture_image(temp_rgb, 'temp', '5')

    q_edge = q_fur_5.get_edge_map(threshold=[40, 200], trigger=True)
    t_edge = t_fur_5.get_edge_map(threshold=[40,200], trigger=True)

    q_outline = q_fur_5.get_outline(q_edge,trigger=True)
    t_outline = t_fur_5.get_outline(t_edge, trigger=True)

    q_slope = q_fur_5.line_fit(q_outline, trigger=True)
    t_slope = t_fur_5.line_fit(t_outline, trigger=True)
    rotation = math.atan(t_slope) - math.atan(q_slope)
    rotate_angle = math.degrees(rotation)

    q_rotate = imutils.rotate(q_edge, rotate_angle)
    while True:
        cv2.imshow('result', q_rotate)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break






    rotation=math.atan(prin)-math.atan(origin)
    cv


