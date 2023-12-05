import pickle

import numpy as np

import config
from localenv import envloader as el

from wcx.utils1 import featureextraction_ShiTomas
import realsense
import utiltools.robotmath as rm

'''
compute homogeneous transformation matrix
'''
# notice: while using different size or different number's aruco maker , plz change the parameter in the realsense file.
def coordinate_transformation(realsense_client, camera_coordinate, save=False):
    # camera coordinate system to tcp coordinate system
    extrinsic_matrix=realsense_client.get_extrinsic_matrix(aruco_size=100,trigger=False) # unit:mm
    # tcp_coordinate=np.dot(np.linalg.inv(extrinsic_matrix),camera_coordinate)
    # print(tcp_coordinate)
    # tcp coordinate system to world coordinate system
    # rotate vector and transformation matrix based on the aruco maker's position and pose
    rotz=np.array([(0, -1, 0), (1, 0, 0), (0, 0, 1)])
    roty = np.array([(0, 0, 1), (0, 1, 0), (-1, 0, 0)])
    rot=np.dot(rotz,roty)
    trans=np.array([(550, -80, 880)]).T
    transformation_matrix=np.hstack((rot,trans))
    transformation_matrix=np.vstack((transformation_matrix,np.array([0,0,0,1])))
    homo_transformation_matrix=np.dot(transformation_matrix,np.linalg.inv(extrinsic_matrix))

    world_coordinate = np.dot(homo_transformation_matrix,camera_coordinate)
    print(world_coordinate)
    if save==True:
        np.save("extrinsic_matrix.npy",extrinsic_matrix)
        np.save("homo_transformation_matrix.npy",homo_transformation_matrix)
        print("successfully save the homogeneous matrix!")


if __name__ == '__main__':
    realsense_client: object = realsense.RealSense()
    '''
    compute homogeneous transformation matrix
    '''

    # camera_coordinate=np.hstack((realsense_client.getcenter(),np.array([1]))).T
    # coordinate_transformation(realsense_client,camera_coordinate,save=True)

    '''
    move action
    '''
    # base, env = el.loadEnv_wrs()
    # rbt, rbtmg, rbtball = el.loadUr3e()
    # rbtx = el.loadUr3ex(rbt)
    # rbt.opengripper(armname="rgt")
    # rbt.opengripper(armname="lft")
    #
    # mp_lft = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="lft")
    # mp_rgt = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="rgt")
    # mp_x_lft = m_plannerx.MotionPlannerRbtX(env, rbt, rbtmg, rbtball, rbtx, armname="lft")
    # mp_x_rgt = m_plannerx.MotionPlannerRbtX(env, rbt, rbtmg, rbtball, rbtx, armname="rgt")
    # #
    '''
    set param
    '''
    # rot_y = np.array([(0, 0, 1), (0, 1, 0), (-1, 0, 0)])
    # objrot = np.dot(rot_y, np.array([(-1, 0, 0), (0, -1, 0), (0, 0, 1)]).T)
    # objpos = np.array([550, 100, 880])
    # # init_tmat = rm.homobuild(objpos,objrot)
    # # init_jnt = mp_lft.get_numik(objpos, objrot)
    # # path_toinit = mp_lft.plan_start2end(end = init_jnt)
    # # mp_x_lft.movepath(path_toinit)
    # # base.run()
    # objcm = el.loadObj("calibboard.stl")
    # grasp = pickle.load(open(config.ROOT + "/graspplanner/pregrasp/calibboard_pregrasps.pkl", "rb"))[0]
    # for x in range(500, 700, 20):
    #     objrelpos, objrelrot = mp_lft.get_rel_posrot(grasp, objpos=(x, 200, 1150), objrot=objrot)
    #     if objrelrot is not None:
    #         break
    # print("objrelpos:", objrelpos, "objrelrot:", objrelrot)
    # success = mp_x_lft.goto_objmat4_goal_x(grasp, objrelpos, objrelrot, rm.homobuild(objpos, objrot), objcm)
    #
    # camera_coordinate = np.hstack((realsense_client.getcenter(), np.array([1]))).T

    '''
    use transformation matrix
    '''
    homo_matrix= np.load('homo_transformation_matrix.npy')
    camera_coordinate= featureextraction_ShiTomas.main()
    world_coordinate=np.dot(homo_matrix,camera_coordinate)
    rot_y = np.array([(0, 0, 1), (0, 1, 0), (-1, 0, 0)])
    objrot = np.dot(rot_y, np.array([(-1, 0, 0), (0, -1, 0), (0, 0, 1)]).T)
    objpos = world_coordinate
    objcm = el.loadObj("calibboard.stl")
    grasp = pickle.load(open(config.ROOT + "/graspplanner/pregrasp/calibboard_pregrasps.pkl", "rb"))[0]
    for x in range(500, 700, 20):
        objrelpos, objrelrot = mp_lft.get_rel_posrot(grasp, objpos=(x, 200, 1150), objrot=objrot)
        if objrelrot is not None:
            break
    print("objrelpos:", objrelpos, "objrelrot:", objrelrot)
    success = mp_x_lft.goto_objmat4_goal_x(grasp, objrelpos, objrelrot, rm.homobuild(objpos, objrot), objcm)
    print(world_coordinate)
