import copy
import json
import os

import numpy as np
import pandas as pd

np.random.seed(1)  # 设置随机种子


def change_dict(old_dict, change):
    new_dict = copy.deepcopy(old_dict)
    for key, value in change.items():
        if isinstance(value, dict) and key in new_dict and isinstance(new_dict[key], dict):
            new_dict[key] = change_dict(new_dict[key], value)
        else:
            new_dict[key] = value
    return new_dict


def xls_read(filename: str) -> np.array:
    return np.array(pd.read_excel(filename))


def json_read(filename: str):
    with open(filename, 'r', encoding='utf8') as f:
        json_data = json.load(f)
    return json_data


def mkdir(dir_path):
    # make sure the directory exists
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def txt_save(filename, data):
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        raise TypeError

    if filename[-4:] != '.txt':
        filename += '.txt'
    json_data = json.dumps(data, default=convert_numpy, indent=4)
    with open(filename, "w") as file:
        file.write(json_data)


def choose_in_and_out(total_node, del_turn_right):
    """选择车辆出入口，排除调头和无效右转车流。仅支持要么单路口，要么2*2"""
    assert total_node in [4, 8]
    four_turn_right = {1: 0, 3: 2, 5: 4, 7: 6}
    edge_in = np.random.choice(total_node)  # 按概率分布随机选择驶入口，返回int
    remain_edge = [_ for _ in range(total_node)]
    remain_edge.remove(edge_in)  # 去掉调头
    if del_turn_right:
        if total_node == 4:
            remain_edge.remove((edge_in + 3) % 4)  # 去掉右转
        else:
            if edge_in in four_turn_right:
                remain_edge.remove(four_turn_right[edge_in])  # 去掉右转
    edge_out = np.random.choice(remain_edge)  # 按概率分布随机选择驶入口
    return edge_in, edge_out


def generate_flow(net_mode: str, rou_path: str, total_car_num: list, space_dist_len: int,
                  penetration: float, bias_ratio: float, total_time=3000):
    """
    single:
        根据入口和出口决定驶入车道,右、直、左分别为0,1,2
                        | 1 |
                    ____|   |____
                    0     △     2
                    ````|   |````
                        | 3 |
    four:
        注意不能用强行禁止变道的net，因为现在按flow写没有制定正确的上路位置，如果不准变道可能到不了destnation。右、直、左分别为0,1,2
                                | 1 |        | 2 |
                            ____|   |________|   |____
                            0    n_0          n_1    3
                            ````|   |````````|   |````
                                |   |        |   |
                            ____|   |________|   |____
                            7    n_2          n_3    4
                            ````|   |````````|   |````
                                | 6 |        | 5 |
    """
    if net_mode == 'single':
        num_io = 4
    elif net_mode == 'four':
        num_io = 8
    else:
        raise ValueError("can generate single or four only!")
    single_dict = {'0,3': 0, '0,2': 1, '0,1': 2,
                   '1,0': 0, '1,3': 1, '1,2': 2,
                   '2,1': 0, '2,0': 1, '2,3': 2,
                   '3,2': 0, '3,1': 1, '3,0': 2}
    four_dict = {0: 'edge_0', 1: 'edge_1', 2: 'E0', 3: 'E1', 4: 'E6', 5: 'E7', 6: 'E5', 7: 'E3'}

    points = np.linspace(0, total_time, len(total_car_num) + 1)
    time_period = [[points[i], points[i + 1]] for i in range(len(points) - 1)]

    with open(rou_path, 'w', encoding='utf-8') as f:
        # 按格式写入车辆类型的最大加速度，最大减速度，车长度，最大速度，未装备自动路由设备
        f.write('<routes>\n' +
                ' <vType id="CAV" accel="3" decel="8" length="4" maxSpeed="15" reroute="false"' +
                ' color="1,0,0" carFollowModel="CACC" probability="' + str(penetration) + '" />\n' +
                ' <vType id="HDV" accel="3" decel="8" length="4" maxSpeed="15" reroute="false"' +
                ' probability="' + str(1 - penetration) + '" />\n' +
                ' <vTypeDistribution id="typedist1" vTypes="CAV HDV" />\n\n')

        space_dist = [choose_in_and_out(num_io, del_turn_right=True) for _ in range(space_dist_len)]
        space_imbalance_start = 0 if len(total_car_num) < 10 else total_time // 2  # 看要不要启动两段式空间分布

        count = 0  # 记录已写入的车辆数
        for time in range(len(time_period)):
            depart_time = np.random.randint(time_period[time][0], time_period[time][1], total_car_num[time])  # 驶入时间
            depart_time.sort()
            for time_car in range(total_car_num[time]):
                count += 1

                flag = False  # 用于标记有没有bias
                if space_dist_len > 0:  # 启用空间不均，按一定概率（bias_ratio）分配车辆到优势车流
                    if depart_time[time_car] >= space_imbalance_start:  # 在开始空间不均时刻之前，按均匀走
                        if np.random.random() < bias_ratio:  # 被选中为bias部分的车流
                            segment_length = (total_time - space_imbalance_start) // space_dist_len  # 每段优势车流时长
                            segment_index = (depart_time[time_car] - space_imbalance_start) // segment_length
                            if depart_time[time_car] == total_time:
                                segment_index = space_dist_len - 1
                            edge_in, edge_out = space_dist[segment_index]
                            flag = True
                if not flag:
                    edge_in, edge_out = choose_in_and_out(num_io, del_turn_right=False)

                if net_mode == 'single':
                    lane_in = single_dict[str(edge_in) + ',' + str(edge_out)]  # 由于不考虑变道模型，故需要限制驶入车道
                    f.write('  <vehicle id="car_' + str(count) + '" depart="' + str(depart_time[time_car]) +
                            '" departLane="' + str(lane_in) + '" arrivalLane="' + str(np.random.randint(3)) +
                            '" departSpeed="max" type="typedist1">\n' +
                            '    <route edges="edge_' + str(edge_in) + ' -edge_' + str(edge_out) + '"/>\n' +
                            '  </vehicle>\n\n')
                else:
                    f.write('  <flow id="car_' + str(count) +
                            '" begin="' + str(depart_time[time_car]) +
                            '" end="' + str(depart_time[time_car]) +
                            '" from="' + four_dict[edge_in] +
                            '" to="' + '-' + four_dict[edge_out] +
                            '" number="1" departSpeed="max" type="typedist1">\n' +
                            '  </flow>\n\n')
        f.write('</routes>\n')
    return space_dist


if __name__ == "__main__":
    np.random.seed(3407)  # 设置随机种子

    # rou_dir = 'single/rou_test'
    rou_dir = 'four/rou_half'
    mkdir('../sumo_sim_env/' + rou_dir)

    cav_penetration = 0.2
    space_imbalance_ratio = 0.16 if 'single' in rou_dir else 0.1

    # time-low_balance/high_balance/imbalance,  space-balance/static_imbalance/dynamic_imbalance
    feature_config = {
        'time_low_balance__space_balance': {  # note:稀疏车流
            'time_dist': [30] * 6,
            'space_dist_len': 0,
        },
        'time_high_balance__space_dynamic_imbalance': {  # note:稠密车流 动态空间不均
            'time_dist': [300] * 6,
            'space_dist_len': 3,
        },
        'time_imbalance__space_balance': {  # note:时间不均 空间均
            'time_dist': [50, 300, 650, 200, 450, 150],
            'space_dist_len': 0,
        },
        'time_imbalance__space_static_imbalance': {  # note:时间不均 静态空间不均
            'time_dist': [50, 300, 650, 200, 450, 150],
            'space_dist_len': 1,
        },
        'time_imbalance__space_dynamic_imbalance': {  # note:时间不均 动态空间不均
            'time_dist': [50, 300, 650, 200, 450, 150],
            'space_dist_len': 3,
        },
        'time_imbalance__space_balance_to_dynamic_imbalance': {  # note:时间双峰 空间半均半静态不均
            'time_dist': [40, 60, 100, 200, 500, 200, 100, 150, 210, 120, 90, 30],  # 1800
            'space_dist_len': 1,  # 目前是根据time_dist长度判断要不要采用空间先均后不均的策略
        },
    }
    if 'four' in rou_dir:  # 2*2入口数翻一倍，路口数翻四倍，大概估计把车流量*2(试过2.5)来从single转到2*2
        for _, ft in feature_config.items():
            ft['time_dist'] = [int(_ * 1) for _ in ft['time_dist']]

    total_n_file = 0
    for feature in feature_config:
        space_dict_in_each_file = {}
        for i in range(5):
            total_n_file += 1
            rou_file_num = str(total_n_file) if total_n_file >= 10 else '0' + str(total_n_file)
            random_space_dict = generate_flow(net_mode=rou_dir.split('/')[0],
                                              rou_path='../sumo_sim_env/' + rou_dir + '/rou.rou' + rou_file_num + '.xml',
                                              total_car_num=feature_config[feature]['time_dist'],
                                              space_dist_len=feature_config[feature]['space_dist_len'],
                                              penetration=cav_penetration,
                                              bias_ratio=space_imbalance_ratio)
            space_dict_in_each_file[rou_file_num] = random_space_dict
        feature_config[feature]['space_dict'] = space_dict_in_each_file
        feature_config[feature]['penetration'] = cav_penetration
        feature_config[feature]['bias_ratio'] = None if feature_config[feature][
                                                            'space_dist_len'] == 0 else space_imbalance_ratio
    txt_save('../sumo_sim_env/' + rou_dir + '/flow_feature.txt', feature_config)
