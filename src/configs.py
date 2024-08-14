
from utils import change_dict


def get_agent_configs(modify_dict):
    light, cav = modify_dict['light'], modify_dict['cav']
    new_light_cfg = change_dict(light_configs, light)
    new_cav_cfg = change_dict(CAV_configs, cav)
    return new_light_cfg, new_cav_cfg


env_configs = {
    # 信号灯时长设置
    'yellow': 3,  # 全红灯时长可以设为0，但黄灯不能为0
    'red': 2,
    'min_green': 5,
    'max_green': 35,

    # 针对单路口的配置
    'single': {
        'base_cycle_length': 100,  # 基准周期时长，用以进行标准化处理
        'base_lane_length': 300,  # 基准道路长度，用以进行标准化处理
        'max_speed': 13.89,  # 道路允许最大车速, m/s
        'max_acc': 3,  # 最大加速度, m/s^2
        'car_length': 4,    # 车辆长度均为4m
        'time_step': 1,  # 仿真步长为1s

        # 文件路径设置
        'sumocfg_path': '../sumo_sim_env/collision_env.sumocfg',    # 从代码到env.sumocfg的路径
        'rou_path': 'single/rou/',
        'net_path': 'single/no_lane_change.net.xml',  # 路网文件只会有一个,故写全
    },

    # 针对2*2路网的配置
    'four': {
        'base_cycle_length': 100,  # 基准周期时长，用以进行标准化处理
        'base_lane_length': 300,  # 基准道路长度，用以进行标准化处理
        'max_speed': 13.89,  # 道路允许最大车速, m/s
        'max_acc': 3,  # 最大加速度, m/s^2
        'time_step': 1,  # 仿真步长为1s
        'car_length': 4,  # 车辆长度均为4m

        # 文件路径设置
        'sumocfg_path': '../sumo_sim_env/collision_env.sumocfg',  # 从代码到env.sumocfg的路径
        'rou_path': 'four/rou_half/',
        'net_path': 'four/no_lane_change_2_2.net.xml',
        'ctrl_lane_path': '../sumo_sim_env/four/control_links.json',  # node-incoming_lane
    },

    # 针对108路网的配置
    'Chengdu': {
        'base_cycle_length': 100,  # 基准周期时长，用以进行标准化处理
        'base_lane_length': 100,    # 基准道路长度，用以进行标准化处理
        'max_speed': 13.89,         # 道路允许最大车速, m/s
        'max_acc': 3,               # 最大加速度, m/s^2
        'time_step': 1,             # 仿真步长为1s
        'car_length': 5,    # 车辆长度均为4m

        # 文件路径设置
        'sumocfg_path': '../sumo_sim_env/collision_env.sumocfg',    # 从代码到env.sumocfg的路径
        'holon_dir': '../sumo_sim_env/Chengdu/',
        'rou_path': 'Chengdu/rou_high/',
        'net_path': 'Chengdu/net.net.xml',
        'add_lane_path': '../sumo_sim_env/Chengdu/intersection_lane.xlsx',  # lane-add_lane
        'ctrl_lane_path': '../sumo_sim_env/Chengdu/control_links.json',     # node-incoming_lane
        'del_intersection': ['n_168', 'n_208', 'n_210', 'n_214', 'n_226', 'n_250', 'n_276', 'n_285', 'n_299', 'n_303',
                             'n_304', 'n_307', 'n_310', 'n_311', 'n_334', 'n_336', 'n_344', 'n_345', 'n_326'],  # 3-legs
    }
}


light_configs = {
    'use_opc': False,   # 注意，目前还没实现tg/pg/tpg的opc，只有G能用        # NOTE: 注意检查是否启用OPC，以及所用的方案是否能正确OPC
    'use_time': True,   # 不用HATD3则用TD3
    'use_phase': True,
    'phase_agent': 'DQN',   # 'DQN'/'TD3'   # here,暂时不实现DQN
    'train_model': True,
    'load_model_name': None,
    'load_model_ep': None,
    'lstm_observe_every_step': True,        # here，基本废弃该参数

    'use_adj': False,       # 若为True，每个路口的信号灯obs和reward会考虑相邻路口的
    'lane_agent': True,     # 若为True，light每次决策一个车道的goal，否则决策所有车道的goal
    'goal_only_indicates_state_mean': True,     # 若为True，上层的goal只表示状态均值delta，不指示状态方差要怎么变

    'time': {
        'obs_dim': 13,   # 路口智能体的状态维度 [下一相位one-hot, 各相位车辆数, 各相位排队数]
        'state_dim': 128,        # RNN层的输出维度
        'act_dim': 1,           # 路口智能体的动作空间 [下个相位持续时间]
        'T': env_configs['yellow'] + env_configs['red'] + env_configs['min_green'],
        'hidden_dim': [400, 300],    # actor和critic网络隐藏层维度一样
    },

    'phase': {
        'obs_dim': 13,  # 路口智能体的状态维度 [各相位车辆数, 各相位排队数]
        'state_dim': 128,  # RNN层的输出维度
        'act_dim': 4,  # 路口智能体的动作空间 [下个相位]
        'T': env_configs['yellow'] + env_configs['red'] + env_configs['min_green'],
        'hidden_dim': [400, 300],  # actor和critic网络隐藏层维度一样。
    },

    'vehicle': {
        'obs_dim': 13,  # 路口智能体的状态维度 [下一相位，各车道当前平均车速，各车道头车xva，时间（即另外两actor动作）]
        'state_dim': 128,  # RNN层的输出维度
        'act_dim': 16,       # 速度建议智能体的动作空间 [路口控制车道数]
        'T': env_configs['yellow'] + env_configs['red'] + env_configs['min_green'],     # here, T后面可以看情况修改
        'hidden_dim': [400, 300],  # actor和critic网络隐藏层维度一样。
    },

    # 信号灯时长设置
    'yellow': env_configs['yellow'],  # 全红灯时长可以设为0，但黄灯不能为0
    'red': env_configs['red'],
    'min_green': env_configs['min_green'],
    'max_green': env_configs['max_green'],

    'encoder_load_path': 'VAE/newEnc_bigger_0814/dataset_2',

    'var': .6,
    'tau': 0.005,  # 软更新参数
    'gamma': 0.95,  # .95  20步
    'batch_size': 64,  # 批大小
    'memory_capacity': 20000,                                           # NOTE: 注意检查开始学习的时机
    # 'learn_start_ratio': 0.1,
    'learn_start_ratio': 0.05,
    # 'learn_start_ratio': 0.15,
    'actor_hidden_dim': [400, 300],
    'critic_hidden_dim': [512, 256],
    'actor_learning_rate': 0.0001,
    'critic_learning_rate': 0.001,
    'actor_scheduler_step': 300,  # 200
    'critic_scheduler_step': 300,  # 400
}


CAV_configs = {
    'use_CAV': True,
    'train_model': True,
    'load_model_name': None,
    'load_model_ep': None,
    'only_ctrl_curr_phase': False,  # 每时刻是只控制当前相位车道(T)还是控制所有车道(F)       # NOTE: 注意检查控制哪些车道
    'only_ctrl_head_cav': False,    # 每个车道上只控制头车(T)还是所有CAV(F)               # NOTE: 注意检查控制哪些CAV

    'cav': {
        'obs_dim': 5 + 2 + 1,   # 车辆智能体的状态维度 [与前车距离、与前车速度差、与路口距离、当前车速、当前加速度、信号灯指示符、倒计时、平均车速]
        'state_dim': 32,   # LSTM输出维度   # !16!
        'act_dim': 1,    # 车辆智能体的动作空间 [决定车辆加速度的大小]
        'T': 1,    # 不宜设置过大，因为要攒够这么多步的obs才能开始决策和学习  # Here, 测试时T基本上都设为1
        'hidden_dim': [128, 128],  # actor和critic网络隐藏层维度一样。
    },
    'high_goal_dim': light_configs['vehicle']['act_dim'],                   # NOTE: 如果要改成一次定所有车道目标，这里改成/8
    'goal_dim': 1,         # 目标维度——建议速度

    'encoder_load_path': 'VAE/newEnc_bigger_0814/dataset_2',
    'hidden_dim': [128, 128],   # !

    'batch_size': 64,       # 批大小
    'memory_capacity': 40000,    # fixed goal 7
    'learn_start_ratio': 0.2,    # fixed goal                                           # NOTE: 注意检查开始学习的时机
    # 'memory_capacity': 120000,    # fixed goal 14
    # 'learn_start_ratio': 0.4,    # fixed goal
    'gamma': 0.9,           # 比路灯短视! 10步
    'tau': 0.005,           # 软更新参数
    # 'alpha': 0.5,             # 内在奖励权重,为0则不考虑上层，为1则忠心耿耿
    # 'alpha': 0.2,             # 内在奖励权重,为0则不考虑上层，为1则忠心耿耿
    'alpha': 0.95,             # 内在奖励权重,为0则不考虑上层，为1则忠心耿耿                    # NOTE: 注意检查alpha
    'actor_learning_rate': 0.0001,
    'critic_learning_rate': 0.001,
    'actor_scheduler_step': 2000,   # !
    'critic_scheduler_step': 1500,  # !
    'var': .6,
}
