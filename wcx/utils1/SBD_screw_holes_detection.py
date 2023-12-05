import cv2
import numpy as np
import copy
import wcx.utils1.realsense as rs
import itertools

def detection(gray_image, trigger = False):
    '''
    :param gray_image: image need to be detected
    :param trigger:  image showing trigger
    :return: center and radius (x,y,r)
    '''
    image = copy.deepcopy(gray_image)
    while trigger:
        cv2.imshow('image', image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Initialize SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5
    params.filterByCircularity = True
    params.minCircularity = 0.7
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.filterByInertia = True
    params.minInertiaRatio = 0.7

    # Create a SimpleBlobDetector object
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs (holes)
    keypoints = detector.detect(image)


    # Draw circles around detected holes
    screw_holes = []
    for kp in keypoints:
        x, y = np.round(kp.pt).astype(int)
        r = np.round(kp.size / 2).astype(int)
        screw_holes.append([x,y,r])
        if trigger:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)


    # Display the output image
    if trigger:
        cv2.imshow('Holes Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('parts amount:', len(keypoints))
        print('screw holes:', screw_holes)

    return screw_holes

def select_square(gray):
    screw_holes = detection(gray_image=gray)
    combinations = itertools.combinations(screw_holes, 4)
    for combo in combinations:
        error = is_square(combo)
        print(error)

    # Define a function to check if four points form a square (with tolerance)
def is_square(combo, tol=0.1):
        # Calculate the distances between the four points
    p1 = combo[0]
    p2 = combo[1]
    p3 = combo[2]
    p4 = combo[3]
    d2 = (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2
    d3 = (p3[0] - p2[0]) ** 2 + (p3[1] - p2[1]) ** 2
    d4 = (p4[0] - p3[0]) ** 2 + (p4[1] - p3[1]) ** 2
    d1 = (p1[0] - p4[0]) ** 2 + (p1[1] - p4[1]) ** 2
    error = abs(d1-d2)+abs(d2 - d3) +abs(d3 - d4)+abs(d4 - d1)+abs(d2 - d4)+abs(d1 - d3)+abs(d2 + d4 - 2 * d3)
    return error


if __name__ == '__main__':
    realsense_client = rs.RealSense(color_frame_size=(1280, 720), depth_frame_size=(1280, 720),
                                color_frame_framerate=30, depth_frame_framerate=30,camera_id=1)
    realsense_client.waitforframe()
    gray = realsense_client.get_gray()
    select_square(gray)

