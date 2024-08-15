import math
import os
import re
import sys

import numpy as np
import traci
from sumolib import checkBinary

from utils import xls_read, json_read

# np.set_printoptions(threshold=np.inf)
# np.random.seed(3407)  # 设置随机种子(没用到随机)
sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))


class Environment:
    def __init__(self, config, single_flag):
        self.net_name = 'four'  # 手动设置
        cfg = config['single'] if single_flag else config[self.net_name]
        self.single_flag = single_flag

        self.time_step = cfg['time_step']
        self.sumocfg_path = cfg['sumocfg_path']
        self.net_path = cfg['net_path']
        self.rou_path = cfg['rou_path']
        self.base_lane_length = cfg['base_lane_length']
        self.base_cycle_length = cfg['base_cycle_length']
        self.max_speed = cfg['max_speed']
        self.max_acc = cfg['max_acc']
        self.car_length = cfg['car_length']

        self.max_green = config['max_green']
        self.yellow = config['yellow']
        self.red = config['red']

        self.collision_count = 0

        if not self.single_flag:
            self.ctrl_lane_path = cfg['ctrl_lane_path']
            self.del_intersection = []
            if self.net_name == 'Chengdu':
                self.all_edges = xls_read(cfg['add_lane_path'])  # np.ndarray
                self.del_intersection = cfg['del_intersection']
                # self.edges_1 = self.all_edges[:, 0]
                # self.edges_2 = self.all_edges[:, 1]
                # self.intersection = self.all_edges[:, 2]
                # # self.intersection_adj = self.get_intersection_relationship()
                # #     """找到路口间的相邻关系，输入当前路口名，返回相邻路口名"""
                # #     data = xls_read('intersection_relationship.xlsx')
                # #     data_dict = {}
                # #     for _ in range(len(data)):
                # #         save_data = list(data[_][1:])
                # #         while np.nan in save_data:
                # #             save_data.remove(np.nan)
                # #
                # #         data_dict[data[_][0]] = save_data
                # #     return data_dict

    """
                                        仿真环境的启动配置
    """

    def start_env(self, turn_on_gui, n_file=1):
        """更新设置仿真环境的路网、车流文件和仿真步长，并开启仿真环境。n_file:车流文件标号"""
        n_file = [str(n_file) if n_file >= 10 else '0' + str(n_file)][0]  # int to str
        sumoBinary = checkBinary('sumo-gui') if turn_on_gui else checkBinary('sumo')

        f = open(self.sumocfg_path, 'r')
        new = [line for line in f]
        f.close()
        new[5] = '        <net-file value="' + self.net_path + '"/>\n'
        new[6] = '        <route-files value="' + self.rou_path + 'rou.rou' + n_file + '.xml"/>\n'
        new[12] = '        <step-length value="' + str(self.time_step) + '"/>\n'
        f = open(self.sumocfg_path, 'w')
        for n in new:
            f.write(n)
        f.close()
        traci.start([sumoBinary, "-c", self.sumocfg_path])

        self.collision_count = 0

    def step_env(self):
        """仿真步进一个步长"""
        self.collision_count += traci.simulation.getCollidingVehiclesNumber() // 2
        traci.simulationStep()

    @staticmethod
    def end_env():
        """关闭仿真环境"""
        traci.close()

    """
                                        读取环境信息——道路结构
    """

    def get_light_id(self):
        """获取路网中所有要控制的信号灯路口的id, 三岔路口没有设置信号灯"""
        if self.single_flag:
            return ['gneJ1']

        if not traci.isLoaded():    # 用于给主函数提供路口列表（在加载agent时路口列表的获取必须在仿真开始前）
            self.start_env(turn_on_gui=False, n_file=1)
            light_id_list = traci.trafficlight.getIDList()
            self.end_env()
        else:                       # 用于给别的函数提供路口列表
            light_id_list = traci.trafficlight.getIDList()

        if self.net_name == 'Chengdu':
            light_id_list = sorted(['n_' + _.split('_')[1] for _ in light_id_list])

        for _ in self.del_intersection:  # 删除三岔路口
            light_id_list.remove(_)
        return light_id_list

    def light_get_lane(self, intersection: str) -> list:
        """获取该信号灯所对应的入口车道编号"""
        if self.single_flag:
            return ["edge_1_1", "edge_1_2", "edge_2_1", "edge_2_2", "edge_3_1", "edge_3_2", "edge_0_1", "edge_0_2"]

        # 读取信号灯与所控制车道字典(若要异构，修改control_links.json，三岔路口某方向的车道是None)
        json_data = json_read(self.ctrl_lane_path)
        return json_data[intersection][0]  # 返回对用车道编号所构成的列表

    def lane_get_con_lane(self, lane):
        """输入一个车道id，输出与之相连的车道id和“输入车道是否为路口的入口车道”(-1:3legs F:add T:lane)"""
        if self.single_flag or (not self.single_flag and self.net_name == 'four'):
            return None, True

        edge = traci.lane.getEdgeID(lane)
        r, c = np.where(self.all_edges[:, :2] == edge)
        if len(r) == 0:  # 查无此道，即该lane不在4-legs的路口
            return None, -1
        else:
            flag = int(c) == 0  # 在第一列找到则说明给的lane直接是信号灯的入口车道
            add_edge = self.all_edges[r, 1 - c][0]
            if add_edge == '0':
                return None, flag
            else:
                order = lane.split('_')[-1]
                return str(add_edge) + '_' + str(order), flag  # edge_lane

    """
                                        获取环境信息——交通状态
    """

    def light_get_ctrl_lane(self, intersection, next_phase, curr_phase=False):
        """获取该路口目前绿灯或即将绿灯的车道名。curr_phase:只获取当前相位车道的头CAV，还是路口所有车道的头CAV"""
        lanes = self.light_get_lane(intersection)
        if curr_phase:
            phase = traci.trafficlight.getPhase(intersection)
            if phase % 3 == 0:  # green light
                lanes = [lanes[phase // 3], lanes[phase // 3 + 4]]
            else:  # yellow or red, control cav on next phase lane
                lanes = [lanes[next_phase], lanes[next_phase + 4]]
        return lanes

    @staticmethod
    def cav_get_lane(cav_id):
        """获取cav所在车道名"""
        return traci.vehicle.getLaneID(cav_id)

    @staticmethod
    def lane_get_all_car(lane_id):
        """获取车道上所有车辆id"""
        return [_ for _ in traci.lane.getLastStepVehicleIDs(lane_id)]

    def lane_get_cav(self, lane_id, head_only=True):
        """获取车道上的CAV。head_only:只获取头CAV还是所有CAV"""
        def find_cav(lane):  # 获取车道上最前一辆CAV的编号。若lane不是None，则返回最前车id，否则返回None
            if lane is not None:
                carID_type_dict = dict([(_, traci.vehicle.getTypeID(_)) for _ in traci.lane.getLastStepVehicleIDs(lane)])
                if head_only:
                    return next((vid for vid, vtype in reversed(list(carID_type_dict.items())) if vtype == 'CAV' and
                                 self.get_head_cav_obs(vid)[1] * self.max_speed > 1), None)     # 仅控制车速大于1m/s的CAV
                    # return next((vid for vid, vtype in reversed(list(carID_type_dict.items())) if
                    #              vtype == 'CAV' and (self.get_head_cav_obs(vid)[3] * self.base_lane_length > 10 or
                    #              self.get_head_cav_obs(vid)[3] == -1)), None)  # 跟路口或前车距离太短的不控，且只控最前面的那辆。
                else:
                    return [vid for vid, vtype in reversed(list(carID_type_dict.items())) if vtype == 'CAV']
            else:  # lane is None
                return None

        cav_list = find_cav(lane_id)
        add_lane_id = self.lane_get_con_lane(lane_id)[0]
        if head_only:
            return cav_list if cav_list else find_cav(add_lane_id)
        else:
            add_list = find_cav(add_lane_id) if add_lane_id else []
            return cav_list + add_list

    def light_get_head_cav(self, intersection, next_phase, curr_phase):
        """获取该路口目前最靠近绿灯的CAV的标号。curr_phase:只获取当前相位车道的头CAV，还是路口所有车道的头CAV"""
        lanes = self.light_get_ctrl_lane(intersection, next_phase, curr_phase)  # 保证了cav顺序和车道顺序一致
        head_cav = []
        for lane in lanes:  # 若green_lane上没有CAV，往后在add_lane上找找看。single则不必
            cav_on_lane = self.lane_get_cav(lane, head_only=True)
            # 当该条道路上没有车时，找到该条道路的上游道路，看道路上是否有车
            if cav_on_lane:
                head_cav.append(cav_on_lane)
            else:
                add_lane = self.lane_get_con_lane(lane)[0]
                head_cav.append(self.lane_get_cav(add_lane, head_only=True))
        return head_cav

    def lane_get_speed(self, lane_id):
        """获取车道平均速度。（路网设置的车道限速若大于env_cfg的13.89，则截断使最大速度保持一致）"""
        return min(traci.lane.getLastStepMeanSpeed(lane_id), self.max_speed)

    def get_light_obs(self, intersection):
        """获取信号灯状态：当前相位,各相排队数（暂不考虑异构路口放信号灯，即目前只能是四相位）"""
        def get_lane_upcoming_veh_num(lane):  # 获取车道状态：车辆数（归一化因子：道路占用率）
            return 0 if lane is None else traci.lane.getLastStepVehicleNumber(lane) * 10 / traci.lane.getLength(lane)

        def get_lane_halting_veh_num(lane):  # 获取车道状态：车辆数（归一化因子：道路占用率）
            return 0 if lane is None else traci.lane.getLastStepHaltingNumber(lane) * 10 / traci.lane.getLength(lane)

        lanes = self.light_get_lane(intersection)  # 获取该信号灯所控制的道路编号
        veh_num, halt = [], []  # 存储该信号灯状态
        for lane in lanes:  # 遍历道路
            add_lane = self.lane_get_con_lane(lane)[0]
            veh_num.append(get_lane_upcoming_veh_num(lane) + get_lane_upcoming_veh_num(add_lane))
            halt.append(get_lane_halting_veh_num(lane) + get_lane_halting_veh_num(add_lane))

        veh_state = [veh_num[i] + veh_num[i + 4] for i in range(4)]
        halt_state = [halt[i] + halt[i + 4] for i in range(4)]
        # curr_phase = traci.trafficlight.getPhase(intersection) // 3
        # one_hot_encoding = np.eye(4)[curr_phase].tolist()

        # return one_hot_encoding + veh_state  # 返回状态(4+4=8)
        # return one_hot_encoding + veh_state + halt_state  # 返回状态(4+4+4=12)
        return veh_state + halt_state  # 返回状态(4+4=8)

    def get_goal_obs(self, intersection):
        """返回路口各车道当前平均车速，各车道头车xva"""
        lanes = self.light_get_lane(intersection)
        ave_v, head_xva = [], []
        for lane in lanes:
            ave_v.append(self.lane_get_speed(lane) / self.max_speed)
            head_cav_id = self.lane_get_cav(lane, head_only=True)
            if head_cav_id is not None:
                # loc=该车到停车线的距离=道路总长度-当前位置(与起始线距离)=lane长度-到lane起点距离
                loc = traci.lane.getLength(lane) - traci.vehicle.getLanePosition(head_cav_id)

                a_lane, flag = self.lane_get_con_lane(lane)
                if not flag:  # 若本lane前面找不到lane(3leg或自己就是lane)则loc不需修改，否则自己是add_lane，就需要加上前面lane的长度
                    loc += traci.lane.getLength(a_lane)

                speed = traci.vehicle.getSpeed(head_cav_id)
                acc = traci.vehicle.getAcceleration(head_cav_id)
                head_xva += [loc / self.base_lane_length, speed / self.max_speed, acc / self.max_acc]
            else:  # vehicle_id is None
                head_xva += [-1, -1, -1]
        return ave_v + head_xva

    def get_head_cav_obs(self, cav_id):
        """获取该车状态xva及其与前一辆车的dx、dv,还有它前方信号灯的通行状态及状态转换的倒计时"""

        def find_front_car(vehicle_id):  # 获取该车前车的id，没有则返回None
            if vehicle_id is not None:
                lane = traci.vehicle.getLaneID(vehicle_id)
                other_car = list(traci.lane.getLastStepVehicleIDs(lane))  # 因为本函数仅用于找路口被控CAV的前车，故lane不会为None

                a_lane, flag = self.lane_get_con_lane(lane)
                if not flag:  # 本lane不直接与前方路口相连,即本车在add_lane上,则加上前lane上的车,反之则为相连或3leg情形,只看本lane车即可
                    other_car = other_car + list(traci.lane.getLastStepVehicleIDs(a_lane))
                    # other_car = other_car_1 + other_car_2 # 师兄这里错了？！[add最后...add最前]+[lane最后...lane最前]才对吧 #####

                if other_car.index(vehicle_id) + 1 >= len(other_car):
                    return None
                else:
                    front_car = other_car[other_car.index(vehicle_id) + 1]
                    return front_car
            else:  # 当cav is None, 则front也是None
                return None

        def get_vehicle_state(vehicle_id):
            """获取指定车辆的自身运行状态(xva)"""
            if vehicle_id is not None:
                lane = traci.vehicle.getLaneID(vehicle_id)
                # loc=该车到停车线的距离=道路总长度-当前位置(与起始线距离)=lane长度-到lane起点距离
                loc = traci.lane.getLength(lane) - traci.vehicle.getLanePosition(vehicle_id)

                a_lane, flag = self.lane_get_con_lane(lane)
                if not flag:  # 若本lane前面找不到lane(3leg或自己就是lane)则loc不需修改，否则自己是add_lane，就需要加上前面lane的长度
                    loc += traci.lane.getLength(a_lane)

                speed = traci.vehicle.getSpeed(vehicle_id)
                acc = traci.vehicle.getAcceleration(vehicle_id)
                return [loc / self.base_lane_length, speed / self.max_speed, acc / self.max_acc]
            else:  # vehicle_id is None
                return [-1, -1, -1]

        def get_light_state_to_car(vehicle_id, use_left_time):
            # 找到当前相位是否为当前车辆的通行相位，若是（1），则输出当前相位剩余时间；若不是（0），则输出还有多久开启该车通行相位
            if vehicle_id is not None:
                if self.single_flag or (not self.single_flag and self.net_name == 'four'):
                    intersection = traci.vehicle.getNextTLS(vehicle_id)[0][0]
                else:
                    edges_1 = self.all_edges[:, 0]
                    edges_2 = self.all_edges[:, 1]
                    intersecs = self.all_edges[:, 2]
                    lane = traci.vehicle.getLaneID(vehicle_id)  # 该车所在车道
                    if lane.split('_')[0] not in edges_1 and lane.split('_')[0] not in edges_2:
                        return -1, -1  # 当该车处于不受红绿灯控制的道路时，返回-1
                    else:
                        if lane.split('_')[0] in edges_2:
                            lane = edges_1[np.where(edges_2 == lane.split('_')[0])][0] + '_' + lane.split('_')[1]
                        intersection = intersecs[np.where(edges_1 == lane.split('_')[0])][0]  # 该车对应的信号灯
                        if lane not in self.light_get_lane(intersection) or intersection in self.del_intersection:
                            return -1, -1

                # print(intersection)
                current_phase = traci.trafficlight.getPhase(intersection)  # 获取当前相位编号
                lane = traci.vehicle.getLaneID(vehicle_id)
                lane_index = self.light_get_lane(intersection).index(lane)
                phase_index = (lane_index % 4) * 3

                curr_phase_left_time = []
                if use_left_time:
                    current_phase_left_time = traci.trafficlight.getNextSwitch(intersection) - traci.simulation.getTime()
                    if current_phase == phase_index:  # 当当前相位为该车辆的通行相位
                        curr_phase_left_time = [current_phase_left_time / self.base_cycle_length]  # 返回通行相位符(1)、当前相位剩余时间
                    else:  # 当当前相位不为该车辆的通行时间
                        all_state_intersection = str(traci.trafficlight.getAllProgramLogics(intersection))
                        duration = re.findall('duration.....', all_state_intersection)
                        duration_list = [int(s.split('=')[1].split('.')[0]) for s in duration]  # 用以存储当前周期整个的配时方案
                        if current_phase < phase_index:  # 下一通行相位在本周期内，返回通行相位剩余开启时间
                            start_time = current_phase_left_time + sum(duration_list[current_phase + 1: phase_index])
                        else:  # 下一通行相位在下一周期
                            start_time = (current_phase_left_time + sum(duration_list[current_phase + 1:]) +
                                          (phase_index % 3) * (self.base_cycle_length / 4))
                        curr_phase_left_time = [start_time / self.base_cycle_length]
                curr_phase_flag = [1] if current_phase == phase_index else [0]
                return curr_phase_flag + curr_phase_left_time
            else:
                return [-1, -1] if use_left_time else [-1]

        ego_state = get_vehicle_state(cav_id)
        front_state = get_vehicle_state(find_front_car(cav_id))
        dx = ego_state[0] - front_state[0] if front_state[0] != -1 else -1
        dv = ego_state[1] - front_state[1] if front_state[1] != -1 else -1

        return ego_state + [dx, dv] + get_light_state_to_car(cav_id, use_left_time=True)

    # def get_cav_od(self, cav_id):    # 获取指定车辆的出发和终到位置
    # def set_head_cav_next_node(self, cav_id, destiny):    # 获取指定车辆的出发和终到位置
    #     traci.vehicle.changeLane(cav_id, )

    """
                                        获取环境信息——RL奖励及评价指标
    """

    def get_light_reward(self, intersection):
        """获取路口的压力值奖励"""
        # w1, w2 = 1, 0

        def get_pressure(e):  # 获取edge入、出道的（备选：压力值）占用率
            r_e = e[1:] if e[0] == '-' else '-' + e
            # return traci.edge.getLastStepVehicleNumber(e), traci.edge.getLastStepVehicleNumber(r_e)
            cap = traci.lane.getLength(e + '_1') / 10  # edge没有getLength方法被迫为之; 车长5m 前后车距2.5m 故设置每车占用10m
            return traci.edge.getLastStepVehicleNumber(e) / cap, traci.edge.getLastStepVehicleNumber(r_e) / cap

        # def get_halting(e):
        #     cap = traci.lane.getLength(e + '_1') / 10  # edge没有getLength方法被迫为之; 车长5m 前后车距2.5m 故设置每车占用10m
        #     return traci.edge.getLastStepHaltingNumber(e) / cap
        #     # return traci.edge.getLastStepHaltingNumber(e) / 10

        lanes = self.light_get_lane(intersection)  # 获取该信号灯所控制的道路编号
        edges = [traci.lane.getEdgeID(lane) for lane in lanes][::2]  # 获取信号灯控制的4个入口edge

        if self.single_flag or (not self.single_flag and self.net_name == 'four'):
            income_count, outcome_count, halt = 0, 0, 0
            for edge in edges:
                i, o = get_pressure(edge)
                income_count, outcome_count = income_count + i, outcome_count + o
                # halt = halt + get_halting(edge)
            # print('pressure:', income_count - outcome_count)
            # print('halting:', halt)
            # return - w1 * (income_count - outcome_count) - w2 * halt  # 奖励是让车尽量多地放行到下游去,且总排队数尽量少
            return outcome_count - income_count  # 奖励是让车尽量多地放行到下游去,且总排队数尽量少

        # for 108 net
        income_count, outcome_count = 0, 0
        for edge in edges:
            r, c = np.where(self.all_edges[:, :2] == edge)
            add_edge = self.all_edges[r, 1 - c][0]
            if len(r) == 0 or add_edge == '0':  # 无add_edge
                i, o = get_pressure(edge)
                income_count, outcome_count = income_count + i, outcome_count + o
            else:  # 有add_edge
                i1, o1 = get_pressure(edge)
                i2, o2 = get_pressure(add_edge)
                income_count, outcome_count = income_count + i1 + i2, outcome_count + o1 + o2
        return - (income_count - outcome_count)  # 奖励是让车尽量多地放行到下游去

    def get_cav_reward(self, obs, last_real_a, last_act):
        """obs即当前时刻车辆观测，last_real_a即上一时刻车辆观测中的加速度，[-3,3]"""
        w1, w2, w3 = 1, 1, 1        # 安全、舒适度和效率的平衡因子
        last_real_a, last_act = last_real_a * self.max_acc, last_act * self.max_acc
        x, v, a = obs[0] * self.base_lane_length, obs[1] * self.max_speed, obs[2] * self.max_acc
        dx, dv = (obs[3] * self.base_lane_length, obs[4] * self.max_speed) if obs[3] != -1 else (-1, -1)

        TTC = (dx - self.car_length) / (dv + 1e-5) if dx != -1 else 10   # 碰撞时间，若无前车则该值很大，不过取大于4即可
        TTC = 1e-5 if dx - self.car_length <= 0 and dx != -1 else TTC    # 在本仿真环境中可能出现已经撞了的情况
        safe = math.log(TTC / 4) if 0 < TTC <= 4 else 0    # 如果排除距离为负导致TTC为负后，TTC仍为负，则说明是前车快于本车，安全。

        jerk = -(((a - last_real_a) / self.max_acc) ** 2)    # 急动度平方，归一化。

        efficiency = v / self.max_speed

        reward = w1 * safe + w2 * jerk + w3 * efficiency
        if obs[5] == 1:     # 绿灯
            min_cross_v = x / (obs[6] * self.base_cycle_length)     # 必须比他大才能冲过去
            if min_cross_v <= min(self.max_speed, v + self.max_acc):     # 能冲过去
                loaf = (v - max(8.33, min_cross_v)) / self.max_speed   # 不用加速度判据，而是期望车辆速度别太低就行
                if loaf < 0:  # 车在磨蹭
                    reward -= TTC * abs(loaf) if 0 < TTC <= 4 else 5 * abs(loaf)    # 如果安全却磨洋工，罚
                    # print('\t\t磨蹭！v=', v * 3.6, 'km/h, loaf=', loaf)
            else:
                if a > 0:   # 过不去还加速
                    reward -= max(1, 5 * abs(a) / self.max_acc)
                    # print('\t\t瞎冲！')
        else:   # 前方红灯
            if dx != -1 and dv < 0 and a < 0:   # 前车快于本车却减速了
                reward -= max(1, 5 * abs(a) / self.max_acc)
                # print('\t\t该走啦')

        mis_match = abs(last_act * self.max_acc - a)  # 两个都是[-3,3]
        if mis_match > 1e-3:
            # print('\t不一致', last_act * self.max_acc, a)
            reward -= mis_match
        return reward

    def get_manager_CoTV_reward(self, lane_id):
        car_in_same_lane = list(traci.lane.getLastStepVehicleIDs(lane_id))  # 仅用于找路口被控CAV的同列车，故lane不会为None
        CoTV1, CoTV2 = 0, 0
        for veh in car_in_same_lane:
            v_a = traci.vehicle.getAcceleration(veh)
            CoTV1 += 1 - traci.vehicle.getSpeed(veh) / self.max_speed
            CoTV2 += (v_a / self.max_acc) ** 2 if v_a >= 0 else 0
        return -(CoTV1 / len(car_in_same_lane) + (CoTV2 / (len(car_in_same_lane) ** 2)) ** 0.5) if len(car_in_same_lane) != 0 else 0

    def get_performance(self):
        """获取路网总的等待时间、排队长度、碳排放、燃油消耗、平均速度作为评价指标"""
        light_id = self.get_light_id()
        w, h, e, f, v, t = 0, 0, 0, 0, [], 0
        for light in light_id:
            lanes = self.light_get_lane(light)
            lanes += [self.lane_get_con_lane(lane)[0] for lane in lanes]
            for lane in lanes:
                if lane:  # lane is not None
                    w += traci.lane.getWaitingTime(lane)
                    h += traci.lane.getLastStepHaltingNumber(lane)
                    e += (traci.lane.getCOEmission(lane) + traci.lane.getCO2Emission(lane))
                    f += traci.lane.getFuelConsumption(lane)
                    lane_v = self.lane_get_speed(lane)
                    v.append(lane_v)
                    car_in_ctrl_lane = [_ for _ in traci.lane.getLastStepVehicleIDs(lane)]
                    t += sum([traci.vehicle.getTimeLoss(_) for _ in car_in_ctrl_lane]) / len(car_in_ctrl_lane) if len(
                        car_in_ctrl_lane) != 0 else 0
        v = sum(v) / len(v)
        return w, h, e / 1e6, f / 1e6, v, t  # e&f in kg/s

    """
                                            智能体动作与环境交互
    """

    @staticmethod
    def set_light_action(intersection, index, duration):
        """改变信号灯配时方案"""
        traci.trafficlight.setPhase(intersection, index)
        traci.trafficlight.setPhaseDuration(intersection, duration)

    def set_head_cav_action(self, cav_id, exact_v, exact_a):
        """将cav的加速度设置为给定值"""
        exact_speed = exact_v * self.max_speed
        exact_acc = exact_a * self.max_acc
        # traci.vehicle.setSpeedMode(cav_id, 0)
        next_speed = max(0, min(exact_speed + exact_acc, self.max_speed))
        traci.vehicle.setSpeed(cav_id, next_speed)

    @staticmethod
    def reset_head_cav(cav_id):
        """控制完成后，恢复默认的跟车模型"""
        traci.vehicle.setSpeed(cav_id, -1)

    @staticmethod
    def set_lane_act_speed(cav_id, next_speed):
        """将cav的加速度设置为给定值, 仅供Loyal使用"""
        # exact_speed = delta_v * self.max_speed + self.lane_get_speed(traci.vehicle.getLaneID(cav_id))
        # next_speed = max(0, min(exact_speed, self.max_speed))
        # traci.vehicle.setSpeedMode(cav_id, 1110)
        traci.vehicle.setSpeed(cav_id, next_speed)

    def get_lane_obs(self, lane_id):
        """获取 lane 上的所有车的信息"""
        cars = traci.lane.getLastStepVehicleIDs(lane_id)
        obs = []
        for car in cars:
            x = 1 - (traci.vehicle.getLanePosition(car) / self.base_lane_length)    # 车到路口的距离
            v = traci.vehicle.getSpeed(car) / self.max_speed
            obs.append([x, v])
        return obs

    @staticmethod
    def get_lane_halt(lane):
        return -traci.lane.getLastStepHaltingNumber(lane) / 10

    # @staticmethod
    # def tmp_get_phase(intersection):
    #     return traci.trafficlight.getPhase(intersection) // 3
    # @staticmethod
    # def get_lane_wait(lane):
    #     return -traci.lane.getWaitingTime(lane) / 60

    def get_adj_light_obs(self, intersection):
        res = {'n_0': ['n_1', 'n_2'],
               'n_1': ['n_0', 'n_3'],
               'n_2': ['n_0', 'n_3'],
               'n_3': ['n_1', 'n_2']}
        l1, l2 = res[intersection][0], res[intersection][1]
        o1, o2 = self.get_light_obs(l1), self.get_light_obs(l2)
        return o1 + o2  # list

    def get_adj_light_reward(self, intersection):
        res = {'n_0': ['n_1', 'n_2'],
               'n_1': ['n_0', 'n_3'],
               'n_2': ['n_0', 'n_3'],
               'n_3': ['n_1', 'n_2']}
        l1, l2 = res[intersection][0], res[intersection][1]
        r1, r2 = self.get_light_reward(l1), self.get_light_reward(l2)
        return r1 + r2  # list


if __name__ == '__main__':
    from configs import env_configs

    env_configs['single']['rou_path'] = 'single/rou/'
    env = Environment(env_configs, False)
    sumoBinary = checkBinary('sumo-gui')
    # time_episode = TIME
    # car_co2_analysis = [['car_' + str(i + 53278) + '.0'] for i in range(7500)]    # 用以存储每辆车的二氧化碳排放情况
    for i in range(30):
        nf = i+1
        env.start_env(sumoBinary, nf)
        for time in range(3000):
            lane = env.light_get_lane(env.get_light_id()[0])[0]
            print(lane, env.lane_get_speed(lane))
            env.step_env()
            # env.ym_try('car_53278.0')

    # for time in range(200):
    #     env.step()
    #     # print(env.get_light_state_to_car('car_54786.0'))
    #     # print(env.get_car_loc_spe_acc('gneE198_0'))
    #     env.get_macro_agent_state_adj('n_11')
    #     # cn = env.get_car_name()
    #     # print(cn)
    #     env.ym_try('car_53278.0')
        env.end_env()