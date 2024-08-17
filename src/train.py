
# -*- coding:utf-8 -*-
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import utils
from agent import IndependentLightAgent, ManagerLightAgent, IndependentCavAgent, WorkerCavAgent, LoyalCavAgent
from configs import env_configs, get_agent_configs
from environment import Environment

np.random.seed(3407)  # 设置随机种子


def setting(base_key, change):
    experience_cfg_base = {
        'baseline': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': False, },
                            'cav': {'use_CAV': False, }}},
        'T': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': False, },
                            'cav': {'use_CAV': False, }}},
        'P': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': True, },
                            'cav': {'use_CAV': False, }}},
        'V': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': False, },
                            'cav': {'use_CAV': True, }}},
        'TV': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': False, },
                            'cav': {'use_CAV': True, }}},
        'PV': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': True, },
                            'cav': {'use_CAV': True, }}},
        'tp': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': True, },
                            'cav': {'use_CAV': False, }}},
        'tpV': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': True, },
                            'cav': {'use_CAV': True, }}},
        'Gv': {
            'use_HRL': True,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': False, },
                            'cav': {'use_CAV': True, }}},
        'tgv': {
            'use_HRL': True,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': False, },
                            'cav': {'use_CAV': True, }}},
        'pgv': {
            'use_HRL': True,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': True, },
                            'cav': {'use_CAV': True, }}},
        'tpgv': {
            'use_HRL': True,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': True, },
                            'cav': {'use_CAV': True, }}},
    }
    return utils.change_dict(experience_cfg_base[base_key], {'modify_dict': change})


experience_cfg = {
    # # Note：Done

    # # Note：Doing
    # # # 公司
    # 'tpgv_g4': setting('tpgv', {
    #     'light': {'vehicle': {'act_dim': 8}},
    #     'cav': {'high_goal_dim': 8}}),
    # 'tpgv_g4_mean_only': setting('tpgv', {
    #     'light': {'goal_only_indicates_state_mean': True},
    #     'cav': {'goal_only_indicates_state_mean': True}}),

    # single
    # 'tpgv_g4': setting('tpgv', {
    #     'light': {'vehicle': {'act_dim': 8}},
    #     'cav': {'high_goal_dim': 8}}),

    # # # 宿舍
    # 'tpgv_loyal_g4': setting('tpgv', {
    #     'light': {'vehicle': {'act_dim': 8}},
    #     'cav': {'high_goal_dim': 8}}),

    # single
    # 'tpgv_g4_noCavVReward': setting('tpgv', {
    #     'light': {'vehicle': {'act_dim': 8}},
    #     'cav': {'high_goal_dim': 8}}),

    # # # 服务器
    # 'no_ctrl': setting('baseline', {}),
    # 'T': setting('T', {}),
    # 'P': setting('P', {}),
    # 'V': setting('V', {}),

    # 'TV': setting('TV', {}),
    # 'PV': setting('PV', {}),

    # 'P_store_oriAct': setting('P', {}),

    # # Note: To do
}

SINGLE_FLAG = True
series_name = '0815_VAE'
MAX_EPISODES = 50  # 训练轮数
SUMO_GUI = False
# flow_feat_id_list = [0, 1, 2, 3, 4, 5]
# flow_feat_id_list = None  # 表示使用所有flow_feat
# flow_feat_id_list = [0, 4]
flow_feat_id_list = [4, 0]

series_name += '_single' if SINGLE_FLAG else ''


def launch_experiment(exp_cfg, save_model=True, single_flag=True, flow_feat_id=None):
    global MAX_EPISODES, SUMO_GUI

    exp_cfg['turn_on_gui_after_learn_start'] = True
    light_configs, cav_configs = get_agent_configs(exp_cfg['modify_dict'])

    experiment_name = exp_cfg['experiment_name']
    writer = SummaryWriter('../log/' + experiment_name)
    model_dir = '../model/' + experiment_name + '/'
    env = Environment(env_configs, single_flag)
    light_id_list = env.get_light_id()
    holon_light_list = [light_id_list]  # four&single都是用一个agent控所有，因此可以这么写。108需要修改这里。独立路口[[n_0], [n_1]...]

    if exp_cfg['use_HRL']:
        if 'loyal' in exp_cfg['experiment_name']:   # 方便起见，以检索实验名中有无loyal字段来判断cav是否使用loyal
            cav_agent = [LoyalCavAgent(light_idl, cav_configs) for light_idl in holon_light_list]
        else:
            cav_agent = [WorkerCavAgent(light_idl, cav_configs) for light_idl in holon_light_list]
        light_agent = [ManagerLightAgent(light_idl, light_configs, cav_agent[hid].get_oa, cav_agent[hid].network.policy)
                       for hid, light_idl in enumerate(holon_light_list)]
    else:
        light_agent = [IndependentLightAgent(light_idl, light_configs) for light_idl in holon_light_list]
        cav_agent = [IndependentCavAgent(light_idl, cav_configs) for light_idl in holon_light_list]

    utils.txt_save('../log/' + str(experiment_name) + '/configs',
                   {'env': env_configs, 'light': light_configs, 'cav': cav_configs})

    for episode in range(MAX_EPISODES):
        if flow_feat_id is None:
            rou_file_num = np.random.randint(1, 31)  # 随机选取一个训练环境
        else:
            rou_file_num = np.random.randint(flow_feat_id * 5 + 1, flow_feat_id * 5 + 6)  # 随机选取一个训练环境
        print("Ep:", episode, "File:", env.rou_path, rou_file_num, '\t', time.strftime("%Y-%m-%d %H:%M:%S"))
        if light_agent[0].pointer > light_agent[0].learn_begin and cav_agent[0].pointer > cav_agent[0].learn_begin:
            SUMO_GUI = exp_cfg['turn_on_gui_after_learn_start']
        env.start_env(SUMO_GUI, n_file=rou_file_num)

        waiting_time, halting_num, emission, fuel_consumption, mean_speed, time_loss = [], [], [], [], [], []

        for t in range(3000):
            for hid in range(len(holon_light_list)):
                if light_agent[hid].__class__.__name__ == 'ManagerLightAgent':
                    l_t, l_p, goal = light_agent[hid].step(env)
                else:   # 'IndependentLightAgent'
                    l_t, l_p = light_agent[hid].step(env)
                    goal = [None] * len(holon_light_list[hid])   # dim=路口数
                real_a, v_a = cav_agent[hid].step(env, goal, l_p)

                # tensorboard只显示每个区第一个路口的动作
                if l_t[0] is not None:
                    writer.add_scalar('green time/' + str(episode), l_t[0], t)
                if l_p[0] is not None:
                    writer.add_scalar('next phase/' + str(episode), l_p[0], t)
                # if goal[0] is not None:
                #     writer.add_scalar('advice speed lane0/' + str(episode), goal[0][0] * env.max_speed, t)
                #     # print(goal * env.max_speed)
                if v_a[0] is not None:
                    writer.add_scalar('head CAV action/' + str(episode), v_a[0], t)
                    writer.add_scalar('head CAV acc_real/' + str(episode), real_a[0], t)

            env.step_env()

            if t % 10 == 0:  # episode % 10 == 9 and
                w, h, e, f, v, timeLoss = env.get_performance()
                waiting_time.append(w)
                halting_num.append(h)
                emission.append(e)
                fuel_consumption.append(f)
                mean_speed.append(v)
                time_loss.append(timeLoss)

            print('\r', t, '\t', light_agent[0].pointer, cav_agent[0].pointer, flush=True, end='')

        ep_light_r = sum(sum(light_agent[_].reward_list) for _ in range(len(holon_light_list)))
        ep_cav_r = sum(sum(sum(sublist) for sublist in cav_agent[_].reward_list) for _ in range(len(holon_light_list)))
        ep_wait = sum(waiting_time)
        ep_halt = sum(halting_num)
        ep_emission = sum(emission)
        ep_fuel = sum(fuel_consumption)
        ep_speed = sum(mean_speed) / len(mean_speed)
        ep_timeloss = sum(time_loss)

        writer.add_scalar('light reward', ep_light_r, episode)
        writer.add_scalar('cav reward', ep_cav_r, episode)
        writer.add_scalar('waiting time', ep_wait, episode)
        writer.add_scalar('halting count', ep_halt, episode)
        writer.add_scalar('carbon emission', ep_emission, episode)
        writer.add_scalar('fuel consumption', ep_fuel, episode)
        writer.add_scalar('average speed', ep_speed, episode)
        writer.add_scalar('time loss', ep_timeloss, episode)
        writer.add_scalar('collision', env.collision_count, episode)

        print('\n', episode,
              '\n\tlight:\tpointer=', light_agent[0].pointer, '\tvar=', light_agent[0].var, '\treward=', ep_light_r,
              '\n\tcav:\tpointer=', cav_agent[0].pointer, '\tvar=', cav_agent[0].var, '\treward=', ep_cav_r,
              '\n\twait=', ep_wait, '\thalt=', ep_halt,
              '\tspeed=', ep_speed, '\tcollision=', env.collision_count,
              '\temission=', ep_emission, '\tfuel_consumption=', ep_fuel, '\ttime_loss=', ep_timeloss)

        # 重置智能体内暂存的列表, 顺便实现每10轮存储一次模型参数
        for hid in range(len(holon_light_list)):
            light, cav = light_agent[hid], cav_agent[hid]
            if save_model:
                utils.mkdir(model_dir)
                if episode % 10 == 9:
                    light.save(model_dir, episode)
                    cav.save(model_dir, episode)
            light.reset()
            cav.reset()
        env.end_env()


if __name__ == "__main__":
    for ffi in flow_feat_id_list:
        for key in experience_cfg:
            series_name = series_name + '/' if series_name[-1] != '/' else series_name
            experience_cfg[key]['experiment_name'] = series_name + key + '_' + str(ffi)
            print(experience_cfg[key]['experiment_name'], 'start running')
            launch_experiment(experience_cfg[key], save_model=True, single_flag=SINGLE_FLAG, flow_feat_id=ffi)

