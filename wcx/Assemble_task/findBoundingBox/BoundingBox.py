import numpy as np
import cv2
def isBackground(image, range_value=[90, 120]):
    n = len(image)
    m = len(image[0])


    is_background = np.zeros((n, m))
    bg_legth_range = np.full((n, 2), -1)
    bg_height_range = np.full((m, 2), -1)

    for i in range(n):
        for j in range(m):
            if image[i][j] >= range_value[0] and image[i][j] <= range_value[1]:

                is_background[i][j] = 1

                if bg_legth_range[i][0] == -1:
                    bg_legth_range[i][0] = j
                    bg_legth_range[i][1] = j
                if bg_height_range[j][0] == -1:
                    bg_height_range[j][0] = i
                    bg_height_range[j][1] = i

                bg_legth_range[i][0] = min(bg_legth_range[i][0], j)
                bg_legth_range[i][1] = max(bg_legth_range[i][1], j)
                bg_height_range[j][0] = min(bg_height_range[j][0], i)
                bg_height_range[j][1] = max(bg_height_range[j][1], i)

    return is_background, bg_legth_range, bg_height_range

def createBoundingBox(image, background_color=[90, 120]):
    n = len(image)
    m = len(image[0])

    is_background, bg_legth_range, bg_height_range = isBackground(image, background_color)
    bounding_box = [[0, 0], [0, 0]] #left, right, top, botton
    initial = True
    for i in range(n):
        for j in range(m):
            if is_background[i][j] != 1:
                if j > bg_height_range[j][0] and i < bg_height_range[j][1] and j > bg_legth_range[i][0] and j < bg_legth_range[i][1]:
                    if initial:
                        bounding_box[0][0] = j
                        bounding_box[0][1] = i
                        bounding_box[1][0] = j
                        bounding_box[1][1] = i
                        initial = False

                    bounding_box[0][0] = min(j, bounding_box[0][0])
                    bounding_box[0][1] = min(i, bounding_box[0][1])
                    bounding_box[1][0] = max(j, bounding_box[1][0])
                    bounding_box[1][1] = max(i, bounding_box[1][1])

    return bounding_box

def c_lh_rate(image, background_color=[90, 120]):
    bounding_box = createBoundingBox(image, background_color)
    l_h_rate = (bounding_box[1][0] - bounding_box[0][0]) / (bounding_box[1][1] - bounding_box[0][1])

    return l_h_rate


def draw_boundingBox(boundingbox, image):
    cv2.rectangle(image, (boundingbox[0][0],boundingbox[0][1]), (boundingbox[1][0],boundingbox[1][1]), (0, 255, 0), 2)
    cv2.imshow("Image with Bounding Box", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    '''
    This is a method to build bounding box for object. please notice to modify the background color, and no similar color out of the background
    '''

    # image = cv2.imread('/home/hlabbunri/Desktop/chenxi wang/wrs-wcx/wcx/Assemble_task/data/simulation_image/screenshot_3.png')
    image = cv2.imread('/wcx/Assemble_task/data/routine_pictures/routine_5.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bounding_box = createBoundingBox(image, background_color=[85, 130])

    l_h_rate = (bounding_box[1][0]-bounding_box[0][0])/(bounding_box[1][1]-bounding_box[0][1])

    np.save("data/temp_lhrate.npy", l_h_rate)
    print("Succuss!")

    draw_boundingBox(bounding_box, image)