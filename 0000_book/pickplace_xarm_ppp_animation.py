import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import math
import numpy as np
import basis.robot_math as rm
import robot_sim.robots.xarm7_shuidi_mobile.xarm7_shuidi_mobile as xsm
import robot_sim.end_effectors.grippers.xarm_gripper.xarm_gripper as xag
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc

base = wd.World(cam_pos=[2, -2, 2], lookat_pos=[.0, 0, .3])
gm.gen_frame().attach_to(base)

ground = cm.gen_box(extent=[5, 5, 1], rgba=[.57, .57, .5, .7])
ground.set_pos(np.array([0, 0, -.51]))
ground.attach_to(base)

object_box = cm.gen_box(extent=[.02, .06, .7], rgba=[.7, .5, .3, .7])
object_box_gl_pos = np.array([.5, -.3, .35])
object_box_gl_rotmat = np.eye(3)
obgl_start_homomat = rm.homomat_from_posrot(object_box_gl_pos, object_box_gl_rotmat)
object_box.set_pos(object_box_gl_pos)
object_box.set_rotmat(object_box_gl_rotmat)
gm.gen_frame().attach_to(object_box)

object_box_gl_goal_pos = np.array([.3, -.4, .25])
object_box_gl_goal_rotmat = rm.rotmat_from_euler(0, math.pi / 2, 0)
obgl_goal_homomat = rm.homomat_from_posrot(object_box_gl_goal_pos, object_box_gl_goal_rotmat)

robot_s = xsm.XArm7YunjiMobile()
rrtc_s = rrtc.RRTConnect(robot_s)
ppp_s = ppp.PickPlacePlanner(robot_s)

original_grasp_info_list = gpa.load_pickle_file('box', './', 'xarm_long_box.pickle')
manipulator_name = "arm"
hand_name = "hnd"
start_conf = robot_s.get_jnt_values(manipulator_name)
conf_list, jawwidth_list, objpose_list = \
    ppp_s.gen_pick_and_place_motion(manipulator_name, hand_name, object_box, original_grasp_info_list,
                                    start_conf, [obgl_start_homomat, obgl_goal_homomat])

robot_attached_list = []
object_attached_list = []
counter = [0]
def update(robot_s, object_box, robot_path, obj_path, robot_attached_list, object_attached_list, counter, task):
    if counter[0] >= len(robot_path):
        counter[0] = 0
    if len(robot_attached_list) != 0:
        for robot_attached in robot_attached_list:
            robot_attached.detach()
        for object_attached in object_attached_list:
            object_attached.detach()
    pose = robot_path[counter[0]]
    robot_s.fk(manipulator_name, pose)
    robot_meshmodel = robot_s.gen_meshmodel()
    robot_meshmodel.attach_to(base)
    robot_attached_list.append(robot_meshmodel)
    obj_pose = obj_path[counter[0]]
    objb_copy = object_box.copy()
    objb_copy.set_homomat(obj_pose)
    objb_copy.attach_to(base)
    object_attached_list.append(objb_copy)
    counter[0]+=1
    return task.again

taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[robot_s, object_box, conf_list, objpose_list, robot_attached_list, object_attached_list, counter],
                      appendTask=True)

# for jvp, objp in zip(conf_list, objpose_list):
#     robot_s.fk(manipulator_name, jvp)
#     robot_s.gen_meshmodel().attach_to(base)
#     objb_copy = object_box.copy()
#     objb_copy.set_homomat(objp)
#     objb_copy.attach_to(base)
base.run()