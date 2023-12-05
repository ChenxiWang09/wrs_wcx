import copy
import math
import time

import numpy as np
import wcx.utils1.rotate_matrix as rot_m
import wcx.localenv.envloader as el
import wcx.utils1.sphere_coordinate as sphere_coordinate

class ur3eusing_example():
    def __init__(self, name='lft', simulation=False):
        '''
        create environment
        '''

        self.base, self.env = el.loadEnv_wrs()
        rbt, rbtmg, rbtball = el.loadUr3e(showrbt=True)
        if not simulation:
            rbtx = el.loadUr3ex(rbt)
        '''
        create robot
        '''

        self.robot_s = rbt
        if not simulation:
            self.robot_c =rbtx
        self.rbtmg=rbtmg
        self.arm_name = name
        self.simulation = simulation
        self.curLftJnts = self.robot_s.initlftjnts

    def load_obj(self, name):
        self.env.loadobj(name=name)

    def __simulation_movement(self, task):

        while self.planning_count >= 0:
            self.planning_count -= 1
            start_jnts = self.curLftJnts
            start_jnts = np.array(start_jnts)
            self.robot_s.movearmfk(armname=self.arm_name, armjnts=start_jnts)
            self.rbtmg.genmnp(self.robot_s, toggleendcoord=True).reparentTo(self.base.render)
            [pos, rot_mat] = self.robot_s.getee(armname=self.arm_name)
            goal_jnts = self.robot_s.numikmsc(self.planning_pos[self.planning_count], self.planning_rot_mats[self.planning_count], seedjntagls=start_jnts, armname=self.arm_name)
            if goal_jnts is not None:
                self.robot_s.movearmfk(goal_jnts, armname=self.arm_name)
                self.rbtmg.genmnp(self.robot_s, toggleendcoord=True).reparentTo(self.base.render)
            else:
                print("No feasible IK!")
        return task.cont

    def planning(self, tgt_poss, tgt_rot_mats) -> object:
        '''
        make simulation
        '''
        self.planning_count = len(tgt_poss)
        self.planning_pos = tgt_poss
        self.planning_rot_mats = tgt_rot_mats
        self.base.taskMgr.doMethodLater(.1, self.__simulation_movement, "Robot movement")
        self.base.run()


    def specific_move(self, target_rot_mat, target_pos) -> object:

        '''
        to place the robot in the ideal pos by human and read its positon
        '''

        start_jnts = self.robot_c.getjnts(armname=self.arm_name)
        start_jnts = np.array(start_jnts)
        self.robot_s.movearmfk(armname=self.arm_name, armjnts=start_jnts)
        self.rbtmg.genmnp(self.robot_s, toggleendcoord=True).reparentTo(self.base.render)
        [pos, rot_mat] = self.robot_s.getee(armname=self.arm_name)
        # print('pos: ', pos)
        # print("rot_mat: ", rot_mat)

        '''
        move to some place
        '''
        tgt_rot_mat = target_rot_mat
        tgt_pos = target_pos
        goal_jnts=self.robot_s.numikmsc(tgt_pos, tgt_rot_mat, seedjntagls = start_jnts, armname = self.arm_name)

        if goal_jnts is None:
            print("Can not find feasible IK!")
            return False

        self.robot_c.movejntssgl(goal_jnts, armname=self.arm_name)

        return True

    def spherical_move(self, target_rot_mat, target_pos, center, radius_range=[300, 600]):

        '''
        to place the robot in the ideal pos by human and read its positon
        '''

        start_jnts = self.robot_c.getjnts(armname=self.arm_name)
        start_jnts = np.array(start_jnts)
        self.robot_s.movearmfk(armname=self.arm_name, armjnts=start_jnts)
        self.rbtmg.genmnp(self.robot_s, toggleendcoord=True).reparentTo(self.base.render)
        [pos, rot_mat] = self.robot_s.getee(armname=self.arm_name)
        # print('pos: ', pos)
        # print("rot_mat: ", rot_mat)

        '''
        move to some place
        '''
        tgt_rot_mat = target_rot_mat
        radius = radius_range[0]
        while(radius <= radius_range[1]):
            tgt_pos = sphere_coordinate.modify_spherical_coordinates(target_pos, center, radius, 0, 0)
            goal_jnts = self.robot_s.numikmsc(tgt_pos, tgt_rot_mat, seedjntagls=start_jnts, armname=self.arm_name)
            radius += 1
            if goal_jnts is None:
                continue
            else:
                break
        if goal_jnts is None:
            print("Can not find feasible IK!")
            return False
        else:
            print('Found target_pos:', tgt_pos)

        self.robot_c.movejntssgl(goal_jnts, armname=self.arm_name)
        return True

    def axis_line_move(self, distance: object, axis: object, data_need: object = False) -> object:


        '''
        to place the robot in the ideal pos by human and read its positon
        '''
        start_jnts = self.robot_c.getjnts(armname=self.arm_name)
        start_jnts = np.array(start_jnts)
        self.robot_s.movearmfk(armname=self.arm_name, armjnts=start_jnts)
        self.rbtmg.genmnp(self.robot_s, toggleendcoord=True).reparentTo(self.base.render)
        [pos, rot_mat] = self.robot_s.getee(armname=self.arm_name)
        # print('pos: ', pos)
        # print("rot_mat: ", rot_mat)

        '''
        move to some place
        '''
        tgt_rot_mat = rot_mat
        if axis=='x':
            tgt_pos=np.array([pos[0]+distance,pos[1],pos[2]])
        elif axis=='y':
            tgt_pos = np.array([pos[0], pos[1]+distance, pos[2]])
        elif axis=='z':
            tgt_pos = np.array([pos[0], pos[1], pos[2]+distance])
        elif axis=='free_move':
            tgt_pos = distance[0]
            tgt_rot_mat = distance[1]
        else:
            print('please define the type of move!')
            return 0
        goal_jnts=self.robot_s.numikmsc(tgt_pos, tgt_rot_mat, seedjntagls = start_jnts, armname = self.arm_name)
        # self.robot_s.movearmfk(goal_jnts, armname=self.arm_name)
        # self.rbtmg.genmnp(self.robot_s, toggleendcoord=True).reparentTo(self.base.render)
        # self.base.run()
        self.robot_c.movejntssgl(goal_jnts, armname=self.arm_name)
        if data_need == True:
            return goal_jnts, tgt_pos, tgt_rot_mat

    def rotate_move(self, angle, axis , data_need=False):
        '''
         to place the robot in the ideal pos by human and read its positon
         '''

        start_jnts = self.robot_c.getjnts(armname=self.arm_name)
        start_jnts = np.array(start_jnts)
        self.robot_s.movearmfk(armname=self.arm_name, armjnts=start_jnts)
        self.rbtmg.genmnp(self.robot_s, toggleendcoord=True).reparentTo(self.base.render)
        [pos, rot_mat] = self.robot_s.getee(armname=self.arm_name)
        print('pos: ', pos)
        print("rot_mat: ", rot_mat)

        '''
        move to some place
        '''
        tgt_pos = pos
        rotate_matrix=rot_m.matrix_generate(angle, axis)
        tgt_rot_mat = np.dot(rotate_matrix,rot_mat)

        goal_jnts = self.robot_s.numikmsc(tgt_pos, tgt_rot_mat, seedjntagls = start_jnts, armname = self.arm_name)
        self.robot_s.movearmfk(goal_jnts, armname=self.arm_name)
        self.rbtmg.genmnp(self.robot_s, toggleendcoord=True).reparentTo(self.base.render)
        # self.base.run()


        self.robot_c.movejntssgl(goal_jnts, armname=self.arm_name)

        if data_need == True:
            return goal_jnts, tgt_pos, tgt_rot_mat

    def set_xx_pos(self, name='start'):
        '''
         to place the robot in the ideal pos by human and read its position
         '''
        '''
        [[-1, 0, 0], [0, 1, 0],[0, 0, -1]]
        '''

        start_jnts = self.robot_c.getjnts(armname=self.arm_name)
        start_jnts = np.array(start_jnts)
        self.robot_s.movearmfk(armname=self.arm_name, armjnts=start_jnts)
        self.rbtmg.genmnp(self.robot_s, toggleendcoord=True).reparentTo(self.base.render)
        [pos, rot_mat] = self.robot_s.getee(armname=self.arm_name)
        print('pos: ', pos)
        print("rot_mat: ", rot_mat)

        '''
        move to some place
        '''
        tgt_pos = pos
        tgt_rot_mat = rot_mat

        # tgt_rot_mat=rot_mat
        goal_jnts = self.robot_s.numikmsc(tgt_pos, tgt_rot_mat, seedjntagls = start_jnts, armname = self.arm_name)

        np.save("data/"+name+"_jnts.npy", goal_jnts)
        print("Successfully save the "+name+" pose!")

        self.robot_c.movejntssgl(goal_jnts, armname=self.arm_name)

        return goal_jnts

    def move_to_xx_pos(self, name: object = 'start', data_need: object = False, trigger: object = False) -> object:
        start_jnts = self.robot_c.getjnts(armname=self.arm_name)
        start_jnts = np.array(start_jnts)
        self.robot_s.movearmfk(armname=self.arm_name, armjnts=start_jnts)
        self.rbtmg.genmnp(self.robot_s, toggleendcoord=True).reparentTo(self.base.render)
        [pos, rot_mat] = self.robot_s.getee(armname=self.arm_name)
        if trigger:
            print('pos: ', pos)
            print("rot_mat: ", rot_mat)

        '''
        move to some place
        '''
        goal_jnts=np.load('data/'+name+'_jnts.npy')
        self.jnts_based_move(goal_jnts)
        if data_need:
            return goal_jnts

    def jnts_based_move(self,goal_jnts, data_need=False):
        self.robot_c.movejntssgl(goal_jnts, armname=self.arm_name)
        if data_need:
            self.robot_s.movearmfk(armname=self.arm_name, armjnts=goal_jnts)
            self.rbtmg.genmnp(self.robot_s, toggleendcoord=True).reparentTo(self.base.render)
            [pos, rot_mat] = self.robot_s.getee(armname=self.arm_name)
            return pos, rot_mat

    def get_pos_rot_jnts_right_now(self, data_need: object = False, trigger: object = False) -> object:
        jnts = self.robot_c.getjnts(armname=self.arm_name)
        jnts = np.array(jnts)
        self.robot_s.movearmfk(armname=self.arm_name, armjnts=jnts)
        self.rbtmg.genmnp(self.robot_s, toggleendcoord=True).reparentTo(self.base.render)
        [ee_pos, ee_rot_mat] = self.robot_s.getee(armname=self.arm_name)
        if trigger:
            print('right now ee pos:',ee_pos)
            print('right now ee rot_mat:',ee_rot_mat)
        return ee_pos, ee_rot_mat, jnts

    def gripper(self, action='open', open_range=0):
        if action == 'open':
            self.robot_c.opengripper(speedpercentage=100, forcepercentage=10, fingerdistance=open_range, armname=self.arm_name)

        elif action == 'close':
            self.robot_c. closegripper(speedpercentage=100, forcepercentage=85, armname=self.arm_name)

    def get_worldToEe_Mat(self):
        '''
        Compute the homogeneous matrix from world coordinate to ee
        :return: homogeneour matrix
        '''

        ee_pos, ee_rot_mat, jnts = self.get_pos_rot_jnts_right_now(data_need=True)
        ZeroRow = np.array([[0, 0, 0, 1]])
        trans_mid = np.hstack((ee_rot_mat, ee_pos))
        TransRG = np.vstack((trans_mid, ZeroRow))

        return TransRG

    def get_homography_matrix(self):
        '''
        Get gripper's homography matrix
        :return: homography matrix
        '''
        pos, rot_mat, jnts = self.get_pos_rot_jnts_right_now(data_need=True)
        x_diff = +20
        z_diff = -15
        y_diff = +110
        r_mat = rot_mat
        t_mat = np.array([[pos[0] - x_diff, pos[1] - y_diff, pos[2] - z_diff]]).T
        homo_mat = np.hstack((r_mat, t_mat))
        homo_mat = np.vstack((homo_mat, np.array([[0, 0, 0, 1]])))

        return homo_mat

    def dual_arm_manipulation(self):
        time.sleep(3)
        self.arm_name = "lft"
        self.gripper(action='open')
        self.arm_name = "rgt"
        self.gripper(action='open')
 #        l_target_pos = [693.22272684, 191.52820895, 907.72439298]
 #        l_rot_mat = [[-0.13070765, -0.98622911,  0.1013294 ],
 # [-0.03003566, -0.09822019, -0.99471134],
 # [ 0.99096588, -0.13305988, -0.01678391]]
 #
 #        r_target_pos = [686.45410676,  37.42157563, 905.35503621]
 #        r_rot_mat = [[ 0.97954587, -0.02807909, -0.19925226],
 # [ 0.19294289, -0.1500342,   0.96967148],
 # [-0.05712215, -0.988282,   -0.14154772]]



        # self.arm_name = "lft"
        # self.gripper(action='open')
        # ee_pos, ee_rot_mat, jnts = self.get_pos_rot_jnts_right_now(data_need=True)
        # ee_pos[2] += 50
        # ee_pos[1] += 65
        # self.PoseFloatMovement(target_rot_mat=ee_rot_mat, target_pos= ee_pos)

        # self.arm_name = "rgt"
        # self.gripper(action='open')
        # ee_pos, ee_rot_mat, jnts = self.get_pos_rot_jnts_right_now(data_need=True)
        # ee_pos[2] += 50
        # ee_pos[1] -= 100
        # self.specific_move(target_rot_mat=ee_rot_mat, target_pos= ee_pos)


        """
        Catch the part
        """
        # self.arm_name = "lft"
        # self.gripper(action='open')
        # ee_pos, ee_rot_mat, jnts = self.get_pos_rot_jnts_right_now(data_need=True)
        # ee_pos[2] -= 50
        # ee_pos[1] -= 65
        # self.PoseFloatMovement(target_pos=ee_pos, target_rot_mat=ee_rot_mat)
        # self.arm_name = "rgt"
        # self.gripper(action='open')
        # ee_pos, ee_rot_mat, jnts = self.get_pos_rot_jnts_right_now(data_need=True)
        # ee_pos[2] -= 50
        # ee_pos[1] += 100
        # self.PoseFloatMovement(target_pos=ee_pos, target_rot_mat=ee_rot_mat)
        #
        # self.arm_name = "lft"
        # self.gripper(action='close')
        # self.arm_name = "rgt"
        # self.gripper(action='close')
        # x_dis = 100
        # z_dis = 20
        # lrDiff = 0
        # l_ee_pos, l_ee_rot_mat, jnts = self.get_pos_rot_jnts_right_now(data_need=True)
        # r_ee_pos, r_ee_rot_mat, jnts = self.get_pos_rot_jnts_right_now(data_need=True)
        # l_tar_pos = l_ee_pos
        # l_tar_rot_mat = l_ee_rot_mat
        # r_tar_pos = r_ee_pos
        # r_tar_rot_mat = r_ee_rot_mat
        #
        # for times in range(100):
        #     self.arm_name = "lft"
        #     l_tar_pos[0] -= x_dis/100
        #     l_tar_pos[2] += z_dis/100
        #     self.PoseFloatMovement(target_rot_mat=l_tar_rot_mat,target_pos=l_tar_pos)
        #
        #     self.arm_name = "rgt"
        #     r_tar_pos[0] -= x_dis/100
        #     r_tar_pos[2] += z_dis/100
        #     self.PoseFloatMovement(target_rot_mat=r_tar_rot_mat, target_pos=r_tar_pos)
        #     print(times)
        #
        # print("finish!")

    def returnState(self):
        self.arm_name = "rgt"
        ee_pos, ee_rot_mat, jnts = self.get_pos_rot_jnts_right_now(data_need=True)
        x_dis = 100
        z_dis = 20
        times  = 20
        ee_pos[0] += x_dis * times / 100
        ee_pos[2] -= z_dis * times / 100
        self.specific_move(target_rot_mat=ee_rot_mat, target_pos=ee_pos)

    def PoseFloatMovement(self, target_pos, target_rot_mat):
        pos, rot_mat, jnts = self.get_pos_rot_jnts_right_now(data_need=True)
        goal_jnts = None
        for i in range(-2, 2, 1):
            if goal_jnts is not None:
                break
            for j in range(-2, 2, 1):
                if goal_jnts is not None:
                    break
                for k in range(-2, 2, 1):
                    rot_z = rot_m.matrix_generate(angle=i, axis='z')
                    rot_x = rot_m.matrix_generate(angle=j, axis='x')
                    rot_y = rot_m.matrix_generate(angle=k, axis='y')

                    test_rot_mat = np.dot(rot_y, np.dot(rot_x, np.dot(rot_z, target_rot_mat)))
                    goal_jnts = self.robot_s.numikmsc(target_pos, test_rot_mat, seedjntagls=jnts, armname=self.arm_name)
                    if goal_jnts is not None:
                        break
        if goal_jnts is None:
            return False
        self.robot_c.movejntssgl(goal_jnts, armname=self.arm_name)


    def rrtPlan(self, start_pos, target_pos, rot_mat, times = 100):
        x_dis = target_pos[0] - start_pos[0]
        y_dis = target_pos[1] - start_pos[1]
        z_dis = target_pos[2] - start_pos[2]
        target_pos = start_pos
        dis = math.sqrt(x_dis^2 + y_dis^2 + z_dis^2)/100
        for i in range(times):
            target_pos[0] += x_dis / times
            target_pos[1] += y_dis / times
            target_pos[2] += z_dis / times

            pos, rot_mat, jnts = self.get_pos_rot_jnts_right_now(data_need=True)
            goal_jnts = self.robot_s.numikmsc(target_pos, rot_mat, seedjntagls=jnts, armname=self.arm_name)
            if goal_jnts is None:
                test_pos = copy.deepcopy(pos)
                for theta in range(90):
                    test_pos[0] = pos[0] + math.sin(theta)*dis
                    test_pos[2] = pos[2] + math.cos(theta)*dis
                    goal_jnts = self.robot_s.numikmsc(test_pos, rot_mat, seedjntagls=jnts, armname=self.arm_name)
                    if goal_jnts is not None:
                        break

        if goal_jnts is None:
            print("No feasible pos!")
            return
        self.robot_c.movejntssgl(goal_jnts, armname=self.arm_name)









if __name__ == '__main__':
    ma_ex=ur3eusing_example()
    ma_ex.arm_name = "lft"
    ma_ex.get_pos_rot_jnts_right_now(trigger=True)
    ma_ex.arm_name = "rgt"
    ma_ex.get_pos_rot_jnts_right_now(trigger=True)
    # ma_ex.gripper(action='open')
