import cv2
import numpy as np
import math
import wcx.outline.outline_extract as outline_extract
import wcx.outline.orientation_line_fit as orientation_line_fit
import copy
import imutils

class furniture_image(object):

    def __init__(self, rgb_image, name, save_path):
        self.type = type
        self.name = name
        self.origin_image = rgb_image
        self.save_path = save_path
        self.rotated_outline = {}


    def get_gray(self, method=cv2.COLOR_BGR2GRAY, trigger=False):
        self.gray = cv2.cvtColor(self.origin_image, method)
        while trigger:
            cv2.imshow(self.name+'_gray', self.gray)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('s'):
                cv2.imwrite(self.save_path+self.name+'.pgm', self.gray)
                print('Successfully save gray '+self.name+'!')

    def get_edge_map(self, threshold, trigger=False):
        self.edge_map = cv2.Canny(self.gray, threshold[0], threshold[1])
        while trigger:
            cv2.imshow(self.name+'_edge_map', self.edge_map)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('s'):
                cv2.imwrite(self.save_path+self.name+'.pgm', self.edge_map)
                print('Successfully save edge map '+self.name+'!')

    def get_outline(self, trigger=False):

        self.outline = outline_extract.outline_extraction(self.edge_map)
        while trigger:
            cv2.imshow(self.name+'_outline', self.outline)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('s'):
                cv2.imwrite(self.save_path+self.name+'.pgm', self.outline)
                print('Successfully save outline '+self.name+'!')

    def line_fit(self, trigger=False):
        outline = copy.deepcopy(self.outline)
        slope, start_point, end_point = orientation_line_fit.line_fit(outline)
        self.slope = slope
        while trigger:
            cv2.line(outline, (start_point[1], start_point[0]), (end_point[1], end_point[0]), color=(155), thickness=5)
            cv2.imshow(self.name+'_line_fit image', outline)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('s'):
                cv2.imwrite(self.save_path+self.name+'.pgm', outline)
                print('Successfully save line_fit image '+self.name+'!')

    def outline_rotation(self, template_outline_path: object, trigger: object = False) -> object:
        template_outline = []
        template_slope = np.load(template_outline_path + 'outline_slope.npy')
        for i in range(8):
            template_outline.append(cv2.imread(template_outline_path+str(i+1)+'.pgm'))
            rotation = math.atan(template_slope[i]) - math.atan(self.slope)
            rotate_angle = math.degrees(rotation)
            self.rotated_outline[i+1] = imutils.rotate_bound(self.outline, rotate_angle)
            while trigger:
                cv2.imshow('outline_rotation_result_'+ str(i+1), self.rotated_outline[i+1])
                key = cv2.waitKey(0)
                if key == ord('s'):
                    cv2.imwrite('/home/hlabbunri/Desktop/wcx/wrs-wcx/wcx/fdcmwork/data/result/0124/rotate_data/'+str(self.name)+'_'+str(i+1)+'.pgm',self.rotated_outline[i+1])
                    print('Successfully save '+str(self.name)+'_'+str(i+1)+'!')
                if key == ord('q'):
                    break





if __name__=="__main__":

    obj = cv2.imread(r'/home/hlabbunri/Desktop/chenxi wang/wrs-wcx/wcx/fdcmwork/data/query/0110/rs1_rgb_2.jpg')
    temp_dic = '/home/hlabbunri/Desktop/chenxi wang/wrs-wcx/wcx/fdcmwork/data/temp/temp_outline/'
    obj2 = cv2.imread('/home/hlabbunri/Desktop/chenxi wang/wrs-wcx/wcx/fdcmwork/data/temp/temp/2.png')
    fur_1 = furniture_image(rgb_image=obj, name='2', save_path='wcx/outline/data/')
    # gray=fur_1.get_gray(trigger=True)
    fur_1.get_gray()
    fur_1.get_edge_map(threshold=[40,160],trigger=True)
    fur_1.get_outline(trigger=True)
    fur_1.line_fit(trigger=True)
    fur_1.outline_rotation(template_outline_path=temp_dic,trigger=True)

    # slope = fur_1.line_fit(edge_map,trigger=True)




