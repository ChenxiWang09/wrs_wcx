import localenv.envloader as el
import cv2
class simulation_of_ideal_pos_camera():
    def __init__(self, arm_name):
        '''
        create environment
        '''
        filename = "/home/hlabbunri/Desktop/chenxi wang/wrs-wcx/wcx/Assemble_task/data/simulation_image/screenshot_" + str(
            1 - 1) + ".png"
        self.base, env = el.loadEnv_wrs(camp=[800, -800, 1600], lookatpos=[820, 120, 820])
        self.base.points_draw([])

        '''
        create robot
        '''
        # self.robot_s = rbt
        # self.rbtmg = rbtmg
        # self.arm_name = arm_name

    def simulation_testing(self):
        self.base.run()




if __name__ == '__main__':
    simulation = simulation_of_ideal_pos_camera(arm_name='rgt')
    simulation.simulation_testing()
