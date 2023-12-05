import cv2
import numpy as np
import math
import wcx.outline.outline_extract as outline_extract
import wcx.outline.orientation_line_fit as orientation_line_fit

def get_gray(origin_image, save_path, method=cv2.COLOR_BGR2GRAY, trigger=False):
    gray = cv2.cvtColor(origin_image, method)
    while trigger:
        cv2.imshow('gray', gray)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite(save_path, gray)
            print('Successfully save gray!')
    return gray

def get_edge_map(threshold, gray, save_path, trigger=False):
    edge_map = cv2.Canny(gray, threshold[0], threshold[1])
    while trigger:
        cv2.imshow('edge_map', edge_map)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite(save_path , edge_map)
            print('Successfully save edge map!')
    return edge_map

def get_outline(edge_map, save_path, trigger=False):

    outline = outline_extract.outline_extraction(edge_map)
    while trigger:
        cv2.imshow('outline', outline)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite(save_path, outline)
            print('Successfully save outline!')
    return outline

def line_fit(edge_map, save_path, trigger=False):

    slope, start_point, end_point = orientation_line_fit.line_fit(edge_map)
    while trigger:
        cv2.line(edge_map, start_point, end_point, color=(155), thickness=10)
        cv2.imshow('line_fit image', edge_map)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite(save_path, edge_map)
            print('Successfully save line_fit image!')

    return slope



# if __name__=="__main__":
