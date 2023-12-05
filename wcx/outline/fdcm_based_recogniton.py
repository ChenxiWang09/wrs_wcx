
import numpy as np
import cv2
import matlab.engine
import math
import time
import orientation_line_fit

class detected_obj(object):

    def __init__(self):
        self.possiblility ={}

    def add_detection(self, obj_num, value):
        self.possiblility[obj_num]=value



if __name__ == "__main__":
    # --------------------------------
    # Recognition part
    # --------------------------------
    #query part for homography

    # Number of parts in this group
    eng = matlab.engine.start_matlab()
    partnumber ,value = eng.chamfermatching_nopic_forpython(sav, j, size_min_x, size_min_y, size_max_x, size_max_y,times , nargout=2)
    print("partnumber: ",j, "shapecost: ",value)
