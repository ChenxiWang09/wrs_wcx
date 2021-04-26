import os
import math
import copy
import pickle
import numpy as np
import basis.data_adapter as da
import modeling.collision_model as cm
import motion.optimization_based.incremental_nik as inik
import motion.probabilistic.rrt_connect as rrtc


class ADPlanner(object):  # AD = Approach_Depart

    def __init__(self, robot_s):
        """
        :param robot_s:
        author: weiwei, hao
        date: 20191122, 20210113
        """
        self.rbt = robot_s
        self.inik_slvr = inik.IncrementalNIK(self.rbt)
        self.rrtc_plnr = rrtc.RRTConnect(self.rbt)

    def gen_jawwidth_motion(self, conf_list, jawwidth):
        jawwidth_list = []
        for _ in conf_list:
            jawwidth_list.append(jawwidth)
        return jawwidth_list

    def gen_approach_linear(self,
                            component_name,
                            goal_hnd_pos,
                            goal_hnd_rotmat,
                            approach_direction=np.array([0, 0, -1]),
                            approach_distance=.1,
                            approach_jawwidth=.05,
                            granularity=0.03,
                            seed_jnt_values=None,
                            toggle_end_grasp='False',
                            end_jawwidth=.0):
        """

        :param component_name:
        :param goal_hnd_pos:
        :param goal_hnd_rotmat:
        :param approach_direction:
        :param approach_distance:
        :param approach_jawwidth:
        :param granularity:
        :param toggle_end_grasp:
        :param end_jawwidth: only used when toggle_end_grasp is True
        :return:
        author: weiwei
        date: 20210125
        """
        conf_list = self.inik_slvr.gen_rel_linear_motion(component_name,
                                                         goal_hnd_pos,
                                                         goal_hnd_rotmat,
                                                         approach_direction,
                                                         approach_distance,
                                                         obstacle_list=[],
                                                         granularity=granularity,
                                                         type='sink',
                                                         seed_jnt_values=seed_jnt_values)
        if len(conf_list) == 0:
            print('Cannot perform approach action!')
            return [], []
        else:
            if toggle_end_grasp:
                jawwidth_list = self.gen_jawwidth_motion(conf_list, approach_jawwidth)
                conf_list += [conf_list[-1]]
                jawwidth_list += [end_jawwidth]
                return conf_list, jawwidth_list
            else:
                return conf_list, self.gen_jawwidth_motion(conf_list, approach_jawwidth)

    def gen_depart_linear(self,
                          component_name,
                          start_hnd_pos,
                          start_hnd_rotmat,
                          depart_direction=np.array([0, 0, 1]),
                          depart_distance=.1,
                          depart_jawwidth=.05,
                          granularity=0.03,
                          seed_jnt_values=None,
                          toggle_begin_grasp=False,
                          begin_jawwidth=.0):
        """

        :param component_name:
        :param goal_hnd_pos:
        :param goal_hnd_rotmat:
        :param depart_direction:
        :param depart_distance:
        :param depart_jawwidth:
        :param granularity:
        :param seed_jnt_values:
        :param toggle_begin_grasp:
        :param begin_jawwidth: only used when toggle_end_grasp is True
        :return: conf_list, jawwidth_list, objhomomat_list_list
        author: weiwei
        date: 20210125
        """
        conf_list = self.inik_slvr.gen_rel_linear_motion(component_name,
                                                         start_hnd_pos,
                                                         start_hnd_rotmat,
                                                         depart_direction,
                                                         depart_distance,
                                                         obstacle_list=[],
                                                         granularity=granularity,
                                                         type='source',
                                                         seed_jnt_values=seed_jnt_values)
        if len(conf_list) == 0:
            print('Cannot perform depart action!')
            return [], []
        else:
            if toggle_begin_grasp:
                jawwidth_list = self.gen_jawwidth_motion(conf_list, depart_jawwidth)
                conf_list = [conf_list[0]] + conf_list
                jawwidth_list = [begin_jawwidth] + jawwidth_list
                return conf_list, jawwidth_list
            else:
                return conf_list, self.gen_jawwidth_motion(conf_list, depart_jawwidth)

    def gen_approach_and_depart_linear(self,
                                       component_name,
                                       goal_hnd_pos,
                                       goal_hnd_rotmat,
                                       approach_direction=np.array([0, 0, -1]),
                                       approach_distance=.1,
                                       approach_jawwidth=.05,
                                       depart_direction=np.array([0, 0, 1]),
                                       depart_distance=.1,
                                       depart_jawwidth=0,
                                       granularity=.03,
                                       seed_jnt_values=None):
        """
        :param component_name:
        :param goal_hnd_pos:
        :param goal_hnd_rotmat:
        :param component_name:
        :param approach_direction:
        :param approach_distance:
        :param approach_jawwidth:
        :param depart_direction:
        :param depart_distance:
        :param depart_jawwidth:
        :param granularity:
        :return: approach_conf_list, depart_jawwidth_list
        author: weiwei, hao
        date: 20191122, 20200105, 20210113, 20210125
        """
        approach_conf_list = self.inik_slvr.gen_rel_linear_motion(component_name,
                                                                  goal_hnd_pos,
                                                                  goal_hnd_rotmat,
                                                                  approach_direction,
                                                                  approach_distance,
                                                                  obstacle_list=[],
                                                                  granularity=granularity,
                                                                  type='sink',
                                                                  seed_jnt_values=seed_jnt_values)
        if approach_distance != 0 and len(approach_conf_list) == 0:
            print('Cannot perform approach action!')
        else:
            depart_conf_list = self.inik_slvr.gen_rel_linear_motion(component_name,
                                                                    goal_hnd_pos,
                                                                    goal_hnd_rotmat,
                                                                    depart_direction,
                                                                    depart_distance,
                                                                    obstacle_list=[],
                                                                    granularity=granularity,
                                                                    type='source',
                                                                    seed_jnt_values=approach_conf_list[-1])
            if depart_distance != 0 and len(depart_conf_list) == 0:
                print('Cannot perform depart action!')
            else:
                approach_jawwidth_list = self.gen_jawwidth_motion(approach_conf_list, approach_jawwidth)
                depart_jawwidth_list = self.gen_jawwidth_motion(depart_conf_list, depart_jawwidth)
                return approach_conf_list + depart_conf_list, approach_jawwidth_list + depart_jawwidth_list
        return [], []

    def gen_approach_motion(self,
                            component_name,
                            goal_hnd_pos,
                            goal_hnd_rotmat,
                            start_conf=None,
                            approach_direction=np.array([0, 0, -1]),
                            approach_distance=.1,
                            approach_jawwidth=.05,
                            granularity=.03,
                            obstacle_list=[],
                            seed_jnt_values=None,
                            toggle_end_grasp='False',
                            end_jawwidth=.0):
        if seed_jnt_values is None:
            seed_jnt_values = start_conf
        conf_list, jawwidth_list = self.gen_approach_linear(component_name,
                                                            goal_hnd_pos,
                                                            goal_hnd_rotmat,
                                                            approach_direction,
                                                            approach_distance,
                                                            approach_jawwidth,
                                                            granularity,
                                                            seed_jnt_values,
                                                            toggle_end_grasp,
                                                            end_jawwidth)
        if start_conf is not None:
            start2approach_conf_list = self.rrtc_plnr.plan(component_name=component_name,
                                                           start_conf=start_conf,
                                                           goal_conf=conf_list[0],
                                                           obstacle_list=obstacle_list,
                                                           ext_dist=.05,
                                                           rand_rate=70,
                                                           maxtime=300)
            start2approach_jawwidth_list = self.gen_jawwidth_motion(start2approach_conf_list, approach_jawwidth)
        return start2approach_conf_list + conf_list, start2approach_jawwidth_list + jawwidth_list

    def gen_depart_motion(self,
                          component_name,
                          start_hnd_pos,
                          start_hnd_rotmat,
                          goal_conf=None,
                          depart_direction=np.array([0, 0, 1]),
                          depart_distance=.1,
                          depart_jawwidth=.05,
                          granularity=.03,
                          obstacle_list=[],
                          seed_jnt_values=None,
                          toggle_begin_grasp=False,
                          begin_jawwidth=.0):
        if seed_jnt_values is None:
            seed_jnt_values = goal_conf
        conf_list, jawwidth_list = self.gen_depart_linear(component_name,
                                                          start_hnd_pos,
                                                          start_hnd_rotmat,
                                                          depart_direction,
                                                          depart_distance,
                                                          depart_jawwidth,
                                                          granularity,
                                                          seed_jnt_values,
                                                          toggle_begin_grasp,
                                                          begin_jawwidth)
        if goal_conf is not None:
            depart2goal_conf_list = self.rrtc_plnr.plan(component_name=component_name,
                                                        start_conf=conf_list[-1],
                                                        goal_conf=goal_conf,
                                                        obstacle_list=obstacle_list,
                                                        ext_dist=.05,
                                                        rand_rate=70,
                                                        maxtime=300)
            depart2goal_jawwidth_list = self.gen_jawwidth_motion(depart2goal_conf_list, depart_jawwidth)
        else:
            depart2goal_conf_list = []
            depart2goal_jawwidth_list = []
        return conf_list + depart2goal_conf_list, jawwidth_list + depart2goal_jawwidth_list

    def gen_approach_and_depart_motion(self,
                                       component_name,
                                       goal_hnd_pos,
                                       goal_hnd_rotmat,
                                       start_conf=None,
                                       goal_conf=None,
                                       approach_direction=np.array([0, 0, -1]),
                                       approach_distance=.1,
                                       approach_jawwidth=.05,
                                       depart_direction=np.array([0, 0, 1]),
                                       depart_distance=.1,
                                       depart_jawwidth=0,
                                       granularity=.03,
                                       seed_jnt_values=None,
                                       obstacle_list=[]):
        """
        degenerate into gen_ad_primitive if both seed_jnt_values and goal_conf are None
        :param component_name:
        :param goal_hnd_pos:
        :param goal_hnd_rotmat:
        :param start_conf:
        :param goal_conf:
        :param approach_direction:
        :param approach_distance:
        :param approach_jawwidth:
        :param depart_direction:
        :param depart_distance:
        :param depart_jawwidth:
        :param granularity:
        :param seed_jnt_values:
        :param obstacle_list:
        :return:
        author: weiwei
        date: 20210113, 20210125
        """
        if seed_jnt_values is None:
            seed_jnt_values = start_conf
        ad_conf_list, ad_jawwidth_list = self.gen_approach_and_depart_linear(component_name,
                                                                             goal_hnd_pos,
                                                                             goal_hnd_rotmat,
                                                                             approach_direction,
                                                                             approach_distance,
                                                                             approach_jawwidth,
                                                                             depart_direction,
                                                                             depart_distance,
                                                                             depart_jawwidth,
                                                                             granularity,
                                                                             seed_jnt_values)
        if start_conf is not None:
            start2approach_conf_list = self.rrtc_plnr.plan(component_name=component_name,
                                                           start_conf=start_conf,
                                                           goal_conf=ad_conf_list[0],
                                                           obstacle_list=obstacle_list,
                                                           ext_dist=.05,
                                                           rand_rate=70,
                                                           maxtime=300)
            start2approach_jawwidth_list = self.gen_jawwidth_motion(start2approach_conf_list, approach_jawwidth)
        if goal_conf is not None:
            depart2goal_conf_list = self.rrtc_plnr.plan(component_name=component_name,
                                                        start_conf=ad_conf_list[-1],
                                                        goal_conf=goal_conf,
                                                        obstacle_list=obstacle_list,
                                                        ext_dist=.05,
                                                        rand_rate=70,
                                                        maxtime=300)
            depart2goal_jawwidth_list = self.gen_jawwidth_motion(depart2goal_conf_list, depart_jawwidth)
        return start2approach_conf_list + ad_conf_list + depart2goal_conf_list, \
               start2approach_jawwidth_list + ad_jawwidth_list + depart2goal_jawwidth_list


if __name__ == '__main__':
    import time
    import basis.robot_math as rm
    import robot_sim.robots.yumi.yumi as ym
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    gm.gen_frame().attach_to(base)
    yumi_instance = ym.Yumi(enable_cc=True)
    manipulator_name = 'rgt_arm'
    hnd_name = 'rgt_hnd'
    goal_pos = np.array([.55, -.1, .3])
    goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    gm.gen_frame(pos=goal_pos, rotmat=goal_rotmat).attach_to(base)

    adp = ADPlanner(yumi_instance)
    tic = time.time()
    # conf_list, jawwidth_list = adp.gen_ad_primitive(component_name,
    #                                                 goal_pos,
    #                                                 goal_rotmat,
    #                                                 approach_direction=np.array([0, 0, -1]),
    #                                                 approach_distance=.1,
    #                                                 depart_direction=np.array([0, 1, 0]),
    #                                                 depart_distance=.0,
    #                                                 depart_jawwidth=0)
    conf_list, jawwidth_list = adp.gen_approach_and_depart_motion(manipulator_name,
                                                                  goal_pos,
                                                                  goal_rotmat,
                                                                  start_conf=yumi_instance.get_jnt_values(manipulator_name),
                                                                  goal_conf=yumi_instance.get_jnt_values(manipulator_name),
                                                                  approach_direction=np.array([0, 0, -1]),
                                                                  approach_distance=.1,
                                                                  depart_direction=np.array([0, -1, 0]),
                                                                  depart_distance=.0,
                                                                  depart_jawwidth=0)
    # conf_list, jawwidth_list = adp.gen_approach_motion(component_name,
    #                                                    goal_pos,
    #                                                    goal_rotmat,
    #                                                    seed_jnt_values=yumi_s.get_jnt_values(component_name),
    #                                                    approach_direction=np.array([0, 0, -1]),
    #                                                    approach_distance=.1)
    # conf_list, jawwidth_list = adp.gen_depart_motion(component_name,
    #                                                  goal_pos,
    #                                                  goal_rotmat,
    #                                                  goal_conf=yumi_s.get_jnt_values(component_name),
    #                                                  depart_direction=np.array([0, 0, 1]),
    #                                                  depart_distance=.1)
    toc = time.time()
    print(toc - tic)
    for i, conf_value in enumerate(conf_list):
        yumi_instance.fk(manipulator_name, conf_value)
        yumi_instance.jaw_to(hnd_name, jawwidth_list[i])
        yumi_meshmodel = yumi_instance.gen_meshmodel()
        yumi_meshmodel.attach_to(base)
        yumi_instance.show_cdprimit()
    base.run()
