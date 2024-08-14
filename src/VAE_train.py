
import numpy as np
from algorithm import Encoder
from environment import Environment
from configs import env_configs
import utils

import random
from torch.utils.tensorboard import SummaryWriter
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.manual_seed(3407)
torch.manual_seed(3407)
np.random.seed(3407)  # 设置随机种子
torch.backends.cudnn.deterministic = True


# MODE = 'sample'  # 'sample' or 'train'
MODE = 'train'  # 'sample' or 'train'

utils.mkdir('../model/VAE/')
if MODE == 'sample':
    # 收集数据
    data_save_name = '0814_sample_2'
    env_configs['four']['rou_path'] = 'four/rou_half/'    # 这版改成了可以禁止变道
    env = Environment(env_configs, False)
    order = np.random.permutation(30)
    dataset = {_: [] for _ in range(60)}
    for i in order:
        nf = i + 1
        env.start_env(True, nf)
        print(nf)
        for time in range(3000):
            if time % 2 == 0:
                for light in env.get_light_id():
                    # 更新encoder参数
                    for lane in env.light_get_lane(light):
                        lane_obs_data = env.get_lane_obs(lane)
                        if len(lane_obs_data) > 0:
                            dataset[len(lane_obs_data)].append(lane_obs_data)
            env.step_env()
        env.end_env()
    utils.txt_save('../model/VAE/' + data_save_name + '.txt', dataset)

if MODE == 'train':
    # 训练
    experiment_name = 'VAE/newEnc_bigger_0814/dataset_2'
    utils.mkdir('../model/' + experiment_name + '/')
    writer = SummaryWriter('../log/' + experiment_name)  # './log/240226light_only'

    my_dataset = utils.json_read('../model/VAE/0814_sample_2.txt')
    batch_size = 64

    dataset = {k: [v[i: i + batch_size] for i in range(0, len(v), batch_size)] for k, v in my_dataset.items() if len(v) > 0}
    shuffled_dataset = [item for sublist in dataset.values() for item in sublist]
    random.shuffle(shuffled_dataset)

    random_keys = np.array(list(dataset.keys()))

    for goal_dim in [2, 4, 8, 16]:
        print(goal_dim, 'running')
        cnt, loss = 0, 0
        encoder = Encoder(2, goal_dim).to(device)
        for data in shuffled_dataset:
            loss = encoder.learn(data)
            writer.add_scalar(str(goal_dim) + '_loss', loss, cnt)
            cnt += 1
        encoder.save('../model/' + experiment_name + '/encoder_dim_' + str(goal_dim))
