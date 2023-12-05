import math
import time

import cv2
import wcx.utils1.move_action as ur_ma
import wcx.utils1.realsense as rs
import wcx.utils1.rotate_matrix as rot_m
import numpy as np
import keyboard as kb
import copy


class PickPlaceTast():
    def __init__(self, rot_ma, part_4_pos):
        self.__rot_ma = rot_ma
        self.__part_4_pos = part_4_pos

    def return_start_pos(self):
        target_rot_mat = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        target_pos = copy.deepcopy(self.__part_4_pos)
        target_pos[2] += 200
        self.__rot_ma.specific_move(target_pos=target_pos, target_rot_mat=target_rot_mat)

    def show_pos(self, pos):

        target_rot_mat = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        ee_pos, ee_rot_mat, jnts = rot_lft.get_pos_rot_jnts_right_now(data_need=True, trigger=True)
        target_pos = copy.deepcopy(pos)
        # target_pos[2] += 200 # this is for cam searching
        target_pos[2] += 150
        print(target_pos)
        self.__rot_ma.specific_move(target_pos=target_pos, target_rot_mat=target_rot_mat)

    def place_part_7(self):
        self.return_start_pos()
        ee_pos, ee_rot_mat, jnts = rot_lft.get_pos_rot_jnts_right_now(data_need=True, trigger=True)
        part4 = self.__part_4_pos
        part4[2] += 200
        self.__rot_ma.specific_move(target_rot_mat=ee_rot_mat, target_pos=part4)
        self.__rot_ma.axis_line_move(-80, 'z')
        self.__rot_ma.axis_line_move(200, 'y')
        dis = 25
        p7_pos = [609.61760108, 295.29138598, 789.844038 +dis]
        self.__rot_ma.specific_move(target_rot_mat=ee_rot_mat, target_pos=p7_pos)
        self.__rot_ma.rotate_move(angle=45, axis='z')
        self.__rot_ma.axis_line_move(-25, 'z')
        self.__rot_ma.gripper(action='open')
        self.__rot_ma.axis_line_move(80, 'z')
        self.return_start_pos()

    def catch_part_7(self, part_4_initial_pose):
        self.return_start_pos()
        ee_pos, ee_rot_mat, jnts = rot_lft.get_pos_rot_jnts_right_now(data_need=True, trigger=True)
        p7_pos = [609.61760108 ,295.29138598 ,789.844038]
        z_dis = 25
        p7_pos[2] += z_dis
        self.__rot_ma.specific_move(target_rot_mat=ee_rot_mat, target_pos=p7_pos)
        self.__rot_ma.gripper(action='open')
        self.__rot_ma.rotate_move(angle=45, axis='z')
        self.__rot_ma.axis_line_move(-z_dis, 'z')
        self.__rot_ma.gripper(action='close')
        self.__rot_ma.axis_line_move(+100, 'z')
        ee_pos, ee_rot_mat, jnts = self.__rot_ma.get_pos_rot_jnts_right_now(data_need=True)
        part_4_top_pose = [part_4_initial_pose[0], part_4_initial_pose[1], 860]
        self.__rot_ma.specific_move(target_rot_mat=ee_rot_mat, target_pos=part_4_top_pose)

    def rotate(self):
        for i in range(10):
            self.__rot_ma.rotate_move(10, 'z')
            time.sleep(0.2)
        time.sleep(2.5)
        self.__rot_ma.rotate_move(-50, 'z')
        time.sleep(0.5)
        self.__rot_ma.rotate_move(-5, 'z')
        time.sleep(1)
        ee_pos, ee_rot_mat, jnts = self.__rot_ma.get_pos_rot_jnts_right_now(data_need=True)
        ee_pos[2] = 807
        self.__rot_ma.specific_move(target_rot_mat=ee_rot_mat, target_pos=ee_pos)
        self.__rot_ma.gripper(action='open')
        self.__rot_ma.gripper(action='close')
        while True:
            gain = input("Wait for screwing, if completed, please press q.")
            if gain == 'q':
                break
        self.__rot_ma.gripper(action='open')
        self.__rot_ma.axis_line_move(60, 'z')
        self.__rot_ma.move_to_xx_pos()

    def catch_part_6(self, p_6_back_left_pose):
        self.__rot_ma.gripper(action='open', open_range=50)
        self.__rot_ma.move_to_xx_pos()


        ee_pos, ee_rot_mat, jnts = self.__rot_ma.get_pos_rot_jnts_right_now(data_need=True, trigger=True)
        print('rotmat',ee_rot_mat)
        print('pos:', p_6_back_left_pose)
        targetPos = copy.deepcopy(p_6_back_left_pose)
        targetPos[2] += 30
        self.__rot_ma.specific_move(target_rot_mat=ee_rot_mat, target_pos=targetPos)
        self.__rot_ma.axis_line_move(axis='z',distance=-30)
        self.__rot_ma.gripper(action='close')

        dis = 50
        ee_pos, ee_rot_mat, jnts = self.__rot_ma.get_pos_rot_jnts_right_now(data_need=True)
        target_pos = ee_pos
        for i in range(11):
            step = dis/10
            target_pos = [target_pos[0] - math.cos(45) * step, target_pos[1], target_pos[2] + math.sin(45) * step]
            self.__rot_ma.specific_move(target_rot_mat=ee_rot_mat, target_pos=target_pos)
        targetMat = np.dot(rot_m.matrix_generate(45, 'z'), ee_rot_mat)
        targetPos = [725.07555282,  124.50018612, 1096.60178284]

        self.__rot_ma.specific_move(target_rot_mat=targetMat, target_pos=targetPos)

    def search_and_place_part_6_1(self):
        ee_rot_mat = np.array([[ 0.70660513, -0.70754621, -0.00935654],
 [-0.70752393, -0.70666325,  0.00607767],
 [-0.01091216,  0.00232546, -0.99993776]])
        rot_mat = rot_m.matrix_generate(angle=180, axis='z')
        ee_rot_mat = np.dot(ee_rot_mat, rot_mat)
        z_dis = 50
        diff = 210
        ee_pos = [self.__part_4_pos[0] - math.cos(45) * diff, self.__part_4_pos[1] - math.cos(45) * diff, 947]
        ee_pos[2] += z_dis

        # self.__rot_ma.axis_line_move(100, 'y')
        ee_rot_mat = np.dot(rot_mat, ee_rot_mat)
        self.__rot_ma.specific_move(target_rot_mat=ee_rot_mat, target_pos=ee_pos)

        for i in range(12):
            self.__rot_ma.axis_line_move(distance=-5, axis='z')
            time.sleep(0.5)
        time.sleep(3)
        dis = 30
        target_pos = [ee_pos[0] + math.cos(45) * dis, ee_pos[1] + math.cos(45) * dis, ee_pos[2]-z_dis]
        ee_pos, ee_rot_mat, jnts = rot_lft.get_pos_rot_jnts_right_now(data_need=True)
        self.__rot_ma.specific_move(ee_rot_mat, target_pos)
        time.sleep(3)

    def search_and_place_part_6_2(self):
        ee_rot_mat = np.array([[0.70660513, -0.70754621, -0.00935654],
                               [-0.70752393, -0.70666325, 0.00607767],
                               [-0.01091216, 0.00232546, -0.99993776]])
        # rot_mat = rot_m.matrix_generate(angle=180, axis='z')
        # ee_rot_mat = np.dot(ee_rot_mat, rot_mat)
        z_dis = 150
        diff = 210

        ee_pos = [self.__part_4_pos[0] - math.cos(45) * diff, self.__part_4_pos[1] - math.cos(45) * diff, 947]
        ee_pos[2] += z_dis
        self.__rot_ma.specific_move(target_rot_mat=ee_rot_mat, target_pos=ee_pos)

        self.__rot_ma.axis_line_move(-80, 'z')
        self.__rot_ma.axis_line_move(-z_dis+80, 'z')

        dis = 30
        target_pos = [ee_pos[0] + math.cos(45) * dis, ee_pos[1] + math.cos(45) * dis, ee_pos[2]-z_dis]
        ee_pos, ee_rot_mat, jnts = rot_lft.get_pos_rot_jnts_right_now(data_need=True)
        self.__rot_ma.specific_move(ee_rot_mat, target_pos)
        time.sleep(3)


        '''
        target pos ee pos: [558.99034114  41.81002793 961.45272337]
        ee rot_mat: [[-5.99498106e-01  7.97499262e-01 -6.78008015e-02]
        '''

    def openGripperWait(self):
        self.__rot_ma.gripper(action='open')
        self.__rot_ma.axis_line_move(axis='z', distance=150)
        catch_pos = copy.deepcopy(self.__part_4_pos)
        catch_pos[2] = 807+50

        target_rot_mat = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        self.__rot_ma.specific_move(target_rot_mat=target_rot_mat, target_pos=catch_pos)
        self.__rot_ma.axis_line_move(axis='z', distance=-50)
        self.__rot_ma.gripper(action='close')

        while True:
            key = input("Press a key: ")
            if key == 'q':
                break

        self.__rot_ma.axis_line_move(axis='z', distance=50)
        self.__rot_ma.rotate_move(axis='z',angle=90)
        self.__rot_ma.axis_line_move(axis='z', distance=-50)
        self.__rot_ma.gripper(action='open')
        self.__rot_ma.axis_line_move(axis='z', distance=200)

    def closeGripperWait(self):
        self.__rot_ma.gripper(action='close')

        while True:
            key = input("Press a key: ")
            if key == 'q':
                break

    def CatchRotateP7ToAssP6(self):
        self.__rot_ma.gripper(action='open')
        target_rot_mat = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        target_pos = copy.deepcopy(self.__part_4_pos)
        z_dis = 200
        target_pos[2] += z_dis
        self.__rot_ma.specific_move(target_rot_mat=target_rot_mat, target_pos=target_pos)
        self.__rot_ma.axis_line_move(axis='z', distance=50-z_dis)
        self.__rot_ma.axis_line_move(axis='z', distance=-22)
        self.__rot_ma.gripper(action='close')
        self.__rot_ma.axis_line_move(axis='z', distance=20)
        self.__rot_ma.rotate_move(axis='z', angle=90)
        self.__rot_ma.axis_line_move(axis='z', distance=-20)
        self.__rot_ma.gripper(action='open')
        self.__rot_ma.axis_line_move(axis='z', distance=100)

    def Step5(self, new_p4, changed_p4):
        '''
        searching height for p8
        '''
        p8_initial_pos = [670.72625595,   24.95576536, 1056.11889105]
        target_rot_mat = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        self.__rot_ma.gripper(action='open')
        self.__rot_ma.specific_move(target_rot_mat=target_rot_mat, target_pos=p8_initial_pos)
        time.sleep(2)
        self.__rot_ma.gripper(action='close')
        ee_pos, ee_rot_mat, jnts = rot_lft.get_pos_rot_jnts_right_now(data_need=True)
        ee_pos[0] -= 120
        ee_pos[1] -= 10
        self.__rot_ma.specific_move(target_rot_mat= ee_rot_mat ,target_pos= ee_pos)

        for i in range(10):
            self.__rot_ma.axis_line_move(distance=-10, axis='z')
            time.sleep(0.5)
        time.sleep(2.5)
        self.__rot_ma.axis_line_move(distance=30, axis='z')
        time.sleep(2)
        self.__rot_ma.specific_move(target_rot_mat=target_rot_mat, target_pos=p8_initial_pos)

        self.__rot_ma.gripper(action='open')

        while True:
            key = input("Press a key: ")
            if key == 'q':
                break

        cat_p4 = copy.deepcopy(new_p4)
        z_diff = 100
        cat_p4[2] += z_diff
        self.__rot_ma.specific_move(target_rot_mat=np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]), target_pos=cat_p4)

        self.__rot_ma.axis_line_move(axis='z', distance=-100)
        self.__rot_ma.gripper(action='close')
        self.__rot_ma.axis_line_move(axis='z', distance=50)
        ee_pos, ee_rot_mat, jnts = rot_lft.get_pos_rot_jnts_right_now(data_need=True)
        p4_changed_pos = changed_p4
        p4_changed_pos[2] += 50
        target_rot_mat = np.dot(rot_m.matrix_generate(angle=90, axis='z'), ee_rot_mat)
        self.__rot_ma.specific_move(target_pos=p4_changed_pos, target_rot_mat=target_rot_mat)
        self.__rot_ma.axis_line_move(axis='z', distance=-50)
        self.__rot_ma.gripper(action='open')
        self.__rot_ma.axis_line_move(axis='z', distance=100)

    def Step5Return(self, new_p4):
        ee_pos, ee_rot_mat, jnts = rot_lft.get_pos_rot_jnts_right_now(data_need=True)
        target_pos = copy.deepcopy(new_p4)
        target_pos[2] += 50
        self.__rot_ma.specific_move(target_rot_mat=ee_rot_mat, target_pos=target_pos)
        self.__rot_ma.axis_line_move(axis='z', distance=-50)
        self.__rot_ma.gripper(action='open')
        self.__rot_ma.axis_line_move(axis='z', distance=100)

    def place_p4(self):
        target_rot_mat = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        target_pos = copy.deepcopy(self.__part_4_pos)
        target_pos[2] = 100 + 807
        self.__rot_ma.specific_move(target_rot_mat=target_rot_mat, target_pos=target_pos)
        self.__rot_ma.axis_line_move(axis='z', distance=-100)
        self.__rot_ma.gripper(action='open')
        self.__rot_ma.axis_line_move(axis='z', distance=100)

    def catP7ForStp5(self, new_p4):
        self.__rot_ma.gripper(action='open')
        target_rot_mat = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        target_pos = copy.deepcopy(new_p4)
        z_dis = 100
        target_pos[2] += z_dis
        self.__rot_ma.specific_move(target_rot_mat=target_rot_mat, target_pos=target_pos)
        self.__rot_ma.axis_line_move(axis='z', distance=50-z_dis)
        self.__rot_ma.axis_line_move(axis='z', distance=-22)
        self.__rot_ma.gripper(action='close')
        self.__rot_ma.axis_line_move(axis='z', distance=280)

    def place_new_p4(self, new_p4):
        target_pos = copy.deepcopy(self.__part_4_pos)
        z_dis = 50
        target_pos[2] = 807 + z_dis
        target_rot_mat = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        self.__rot_ma.specific_move(target_rot_mat=target_rot_mat, target_pos=target_pos)
        self.__rot_ma.axis_line_move(axis='z', distance=-z_dis)
        self.__rot_ma.gripper(action='close')
        self.__rot_ma.axis_line_move(axis='z', distance=260)

        while True:
            key = input("Press a key: ")
            if key == 'q':
                break

        ee_pos, ee_rot_mat, jnts = rot_lft.get_pos_rot_jnts_right_now(data_need=True)
        rot_mat = rot_m.matrix_generate(axis='z', angle=-90)
        tgt_rot_mat = np.dot(rot_mat, ee_rot_mat)
        self.__rot_ma.specific_move(target_rot_mat=tgt_rot_mat, target_pos=new_p4)
        self.__rot_ma.gripper(action='open')
        self.__rot_ma.axis_line_move(axis='z', distance=100)

    def final_assembly(self, changed_p4):
 #        mid_p = [831.78583266, 104.91156061, 978.56612745]
 #        Final_step_Place_Pos = [705.31521366, 102.21355847, 876.64894509]
 #        Catch_pos = [795.02225336, 102.60612726, 866.94173463]
 #        final_step_posture = [[-0.99799973,  0.01102311,  0.06224981],
 # [ 0.01149776,  0.99990746,  0.00727179],
 # [-0.06216389,  0.00797298, -0.99803411]]
 #        self.__rot_ma.gripper(action='open')
 #        self.__rot_ma.specific_move(target_rot_mat=final_step_posture, target_pos=mid_p)
 #        self.__rot_ma.specific_move(target_rot_mat=final_step_posture, target_pos=Catch_pos)


 # #        self.__rot_ma.gripper(action='close')
        # self.__rot_ma.specific_move(target_rot_mat=final_step_posture, target_pos=Final_step_Place_Pos)
        # time.sleep(2)
        # self.__rot_ma.gripper(action='open')
        # self.__rot_ma.axis_line_move(axis='z', distance=150)


        self.show_pos(changed_p4)
        self.__rot_ma.axis_line_move(axis='z', distance=-100)
        self.__rot_ma.axis_line_move(axis='z', distance=-50)
        self.__rot_ma.gripper(action='close')

        while True:
            key = input("Press a key: ")
            if key == 'q':
                break
        self.__rot_ma.gripper(action='open')
        self.__rot_ma.axis_line_move(axis='z', distance=100)

    def fourP6PosTest(self, rightFrontPos):
        self.__rot_ma.gripper(open_range=50)
        ee_pos, ee_rot_mat, jnts = self.__rot_ma.get_pos_rot_jnts_right_now(data_need=True)
        rightFrontPos[2] += 15
        self.__rot_ma.specific_move(target_rot_mat=ee_rot_mat, target_pos=rightFrontPos)
        rightBackPos = copy.deepcopy(rightFrontPos)
        rightBackPos[0] += 30
        self.__rot_ma.specific_move(target_rot_mat=ee_rot_mat, target_pos=rightBackPos)
        leftFrontPos = copy.deepcopy(rightFrontPos)
        leftFrontPos[1] -= 32
        self.__rot_ma.specific_move(target_rot_mat=ee_rot_mat, target_pos=leftFrontPos)
        leftBackPos = copy.deepcopy(rightFrontPos)
        leftBackPos[0] += 30
        leftBackPos[1] -= 32


        self.__rot_ma.specific_move(target_rot_mat=ee_rot_mat, target_pos=leftBackPos)


if __name__ == '__main__':

    '''
    Screw holes detection
    '''
    
    '''
    Initial pose recognition
    '''

    '''
    load rgt camera pose
    '''
    # start pose = [602.17301192  27.59945758 982.76396617]
    # p4_pos = [608.89888926, 95.84135886, 780.60438596]
    p4_pos = [608.89888926, 45.84135886, 780.60438596]
    p6_pos = [810.6386921,  246.90420044, 930]  # diff = 30 in x axis, diff = 35 in y axis

    p4_new_pos =[455.54319937,  22.96847528, 957.30472292]
    p4_change_pos  = [565.33147606, 103.55368602, 955.05840293]
    p7_pos = [609.61760108, 295.29138598, 789.844038]
    p4_new_pos = np.array(p4_new_pos)
    p4_pos = np.array(p4_pos)
    diff = p4_new_pos - p4_pos
    diff2 = p4_change_pos - p4_new_pos


    # camera pos setting
    # rot_rgt = ur_ma.ur3eusing_example(name='rgt')
    # rot_rgt.move_to_xx_pos(name='higher_cam')


    # rot_lft = ur_ma.ur3eusing_example(name='lft')
    # ee_pos, ee_rot_mat, jnts = rot_lft.get_pos_rot_jnts_right_now(data_need=True)
    # p4_new_pos[2]+= 100
    # rot_lft.specific_move(target_rot_mat=ee_rot_mat, target_pos=p4_new_pos)
    #
    rot_lft = ur_ma.ur3eusing_example(name='lft')
    #
    # rot_lft.dual_arm_manipulation()
    # rot_lft.returnState()


    # ee_pos, ee_rot_mat, jnts = rot_lft.get_pos_rot_jnts_right_now(data_need=True)
    # rot_lft.specific_move(target_rot_mat=ee_rot_mat,target_pos=[806.77896245, 185.62800867 910.80730584])
    # rot_lft.get_pos_rot_jnts_right_now(trigger=True)
    # rot_lft.arm_name = "rgt"
    rot_lft.get_pos_rot_jnts_right_now(trigger=True)
    pptk = PickPlaceTast(rot_lft, part_4_pos=p4_pos)
    pptk.fourP6PosTest(p6_pos)
    # pptk.show_pos(p4_change_pos)


    # pos: [685.65719887 184.93299234 929.44029515]
    #
    # pos:  [681.96385305  26.93307258 929.46138402]

    # [670.10562444 184.70140036 919.6067588]
    # [675.26292348  33.47560084 929.61347579]
    # rot_lft.get_pos_rot_jnts_right_now(trigger=True)
    #
    # pptk.show_pos(p4_change_pos)
    # rot_lft.rotate_move(axis='z', angle=-90)
    # ee_pos, ee_rot_mat, jnts = rot_lft.get_pos_rot_jnts_right_now(data_need=True)
    # ee_pos[2] -= 50
    # rot_lft.specific_move(target_pos=ee_pos, target_rot_mat=ee_rot_mat)
    # rot_lft.axis_line_move(axis='z', distance=-50)
    # pptk.fourP6PosTest(p6_pos)
    # pptk.return_start_pos()

    # pptk.return_start_pos()

    # start_jnts = np.load("data/start_jnts.npy")
    # rot_lft.jnts_based_move(goal_jnts=start_jnts)
    # ee_pos, ee_rot_mat, jnts = rot_lft.get_pos_rot_jnts_right_now(data_need=True, trigger=True)

    '''
    step_1
    '''
    # pptk.place_part_7()
    # pptk.catch_part_7(part_4_initial_pose=p4_pos)
    # pptk.rotate()
    '''
    step_2
    '''
    # p6_right_front = p6_pos
    #
    # p6_pos_left_front = copy.deepcopy(p6_pos)
    # p6_pos_left_front[1] -= 32
    #
    # p6_pos_right_back = copy.deepcopy(p6_pos)
    # p6_pos_right_back[0] += 30
    #
    # p6_pos_left_back = copy.deepcopy(p6_pos)
    # p6_pos_left_back[1] -= 32
    # p6_pos_left_back[0] += 30
    #
    # pptk.catch_part_6(p_6_back_left_pose=p6_pos_left_front)
    # pptk.search_and_place_part_6_1()
    # pptk.openGripperWait()
    #
    # pptk.catch_part_6(p_6_back_left_pose=p6_pos_left_back)
    # pptk.search_and_place_part_6_2()
    # pptk.openGripperWait()
    #
    #
    # pptk.catch_part_6(p_6_back_left_pose=p6_right_front)
    # pptk.search_and_place_part_6_2()
    # pptk.openGripperWait()
    #
    # pptk.catch_part_6(p_6_back_left_pose=p6_pos_right_back)
    # pptk.search_and_place_part_6_2()
    # pptk.openGripperWait()

    '''
    step_5
    '''
    # 5.1
    #cam pos change

    # rot_rgt = ur_ma.ur3eusing_example(name='rgt')
    # rot_rgt.move_to_xx_pos('final_cam')
    #
    # ee_pos, ee_rot_mat, jnts = rot_rgt.get_pos_rot_jnts_right_now(data_need=True)
    # ee_pos[0] += diff[0]
    # ee_pos[1] += diff[1]
    # ee_pos[2] += diff[2]-27
    # rot_rgt.specific_move(target_rot_mat=ee_rot_mat, target_pos=ee_pos)

    # rot_lft.move_to_xx_pos()

    # pptk.place_new_p4(new_p4)
    # pptk.catP7ForStp5(p4_new_pos)
    # pptk.place_p4_higher()

    # 5.2
    # pptk.Step5Return(p4_new_pos)
    # pptk.Step5(p4_new_pos, p4_change_pos)

    #5.3
    # pptk.final_assembly(p4_change_pos)

    #5.4
 #    cam_pos_new = [ 857.07725028, -422.96478684, 1123.94803055]
 #    cam_posture = [[ 0.763348,    0.50258667, -0.40584044],
 # [ 0.64511411, -0.56043382,  0.51936666],
 # [ 0.03358005, -0.6582709,  -0.75203179]]
 #    rot_rgt = ur_ma.ur3eusing_example(name='rgt')
 #    ee_pos, ee_rot_mat, jnts = rot_rgt.get_pos_rot_jnts_right_now(data_need=True, trigger=True)
 #    rot_rgt.specific_move(target_rot_mat=cam_posture, target_pos=cam_pos_new)






