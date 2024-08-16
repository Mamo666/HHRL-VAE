"""
这版的Light无论如何只有一个网络(把TP去掉)
[T/P/tp/TP] [Gv/tgv/pgv/tpgv] [V/TV/PV/tpV]    # tpg:HATD3,v:worker,TPGV:TD3
"""

from collections import deque

import numpy as np

from algorithm import HATD3Triple, HATD3Double, HATD3, TD3Single, WorkerTD3, ManagerTD3, Encoder

np.random.seed(3407)  # 设置随机种子


class IndependentLightAgent:
    def __init__(self, light_id, config):
        if isinstance(light_id, str):
            self.holon_name = light_id
            self.light_id = [light_id]
        elif isinstance(light_id, (list, tuple)):
            self.holon_name = 'h_' + light_id[0]
            self.light_id = list(light_id)

        self.use_adj = config['use_adj']
        self.use_time = config['use_time']
        self.use_phase = config['use_phase']
        self.train_model = config['train_model']
        self.load_model = config['load_model_name'] is not None
        self.lstm_observe_every_step = config['lstm_observe_every_step']

        config['memory_capacity'] = config['memory_capacity'] * len(self.light_id)  # 控制多路口会导致存速翻倍，故扩大容量以匹配

        if self.use_time and self.use_phase:
            self.network = HATD3(config)
        elif self.use_time:
            self.network = TD3Single(config, 'time')
        else:   # only phase or neither
            self.network = TD3Single(config, 'phase')
            # self.network = DQN(config, 'phase')
        self.save = lambda path, ep: self.network.save(path + 'light_agent_' + self.holon_name + '_ep_' + str(ep))
        if self.load_model:
            load_ep = str(config['load_model_ep']) if config['load_model_ep'] else '99'
            self.network.load('../model/' + config['load_model_name'] + '/light_agent_' + self.holon_name + '_ep_' + load_ep)

        self.var = config['var']
        self.o_t = config['time']['obs_dim']
        self.T_t = config['time']['T']
        self.o_p = config['phase']['obs_dim']
        self.T_p = config['phase']['T']

        self.min_green = config['min_green']
        self.max_green = config['max_green']
        self.yellow = config['yellow']
        self.red = config['red']

        self.time_index = {light: 0 for light in self.light_id}
        self.green = {light: config['min_green'] for light in self.light_id}
        self.color = {light: 'g' for light in self.light_id}
        self.phase_list = {light: deque([0], maxlen=2) for light in self.light_id}

        self.step_time_obs = {light: deque([[0] * self.o_t for _ in range(self.T_t)], maxlen=self.T_t) for light in self.light_id}
        self.o_t_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_t_list = {light: deque(maxlen=2) for light in self.light_id}
        self.step_phase_obs = {light: deque([[0] * self.o_p for _ in range(self.T_p)], maxlen=self.T_p) for light in self.light_id}
        self.o_p_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_p_list = {light: deque(maxlen=2) for light in self.light_id}
        self.reward_list = []

    @property
    def pointer(self):
        return self.network.pointer

    @property
    def learn_begin(self):
        return self.network.learn_begin

    def step(self, env):
        tl, pl = [], []
        for light in self.light_id:
            t, p = self._step(env, light)
            tl.append(t)
            pl.append(p)
        return tl, pl
        # return tl[0], pl[0]  # 只向外展示第一个路口的动作

    def _step(self, env, light):
        my_obs = env.get_light_obs(light)   # 8
        add_phase_id = np.eye(4)[int(self.phase_list[light][-1])].tolist()               # 4
        remain_green = (self.green[light] if self.color[light] in 'yr' else self.time_index[light]) / (env.base_cycle_length / 4)   # 1
        if self.use_adj and len(self.light_id) > 1:
            adj_obs = [_ * 0.3 for _ in env.get_adj_light_obs(light)]  # 16
            add_light_id = np.eye(len(self.light_id))[int(self.light_id.index(light))].tolist()  # 4
            obs = my_obs + adj_obs + add_light_id + add_phase_id + [remain_green]
        else:
            obs = my_obs + add_phase_id + [remain_green]
        if self.lstm_observe_every_step:
            self.step_time_obs[light].append(obs)
            self.step_phase_obs[light].append(obs)

        next_green, next_phase = None, None
        if self.time_index[light] == 0:
            if self.color[light] == 'y' and self.red != 0:  # 黄灯结束切红灯
                env.set_light_action(light, self.phase_list[light][-2] * 3 + 2, self.red)
                self.time_index[light], self.color[light] = self.red, 'r'
            elif self.color[light] == 'r' or (self.color[light] == 'y' and self.red == 0):  # 红灯结束或（黄灯结束且无全红相位）切绿灯
                env.set_light_action(light, self.phase_list[light][-1] * 3, self.green[light])
                self.time_index[light], self.color[light] = self.green[light], 'g'
            elif self.color[light] == 'g':
                if not self.lstm_observe_every_step:
                    assert self.network.t == 1, 'LSTM should not be used when lstm_observe_every_step is False'
                    self.step_time_obs[light] = obs
                    self.step_phase_obs[light] = obs

                # Choose next phase
                if not self.use_phase or (not self.train_model and not self.load_model):
                    a_p = (self.phase_list[light][-1] + 1) % 4     # 不控制时默认动作
                else:
                    o_p = np.array(self.step_phase_obs[light]).flatten().tolist()
                    self.o_p_list[light].append(o_p)       # 存最近两次决策obs(2, T*o_dim)

                    if self.network.__class__.__name__ in ('TD3Single', 'HATD3'):
                        if self.pointer < self.learn_begin and not self.load_model:  # 随机填充
                            a_p = np.random.random(self.network.a2_dim) * 2 - 1
                        else:
                            a_p = self.network.choose_phase_action(o_p)
                        if self.train_model:    # 加噪声
                            a_p = np.clip(np.random.normal(0, self.var, size=a_p.shape) + a_p, -1, 1)
                        result = np.zeros_like(a_p)
                        result[np.argmax(a_p)] = 1.
                        self.a_p_list[light].append(result)
                        a_p = np.argmax(a_p)    # 4个里面值最大的
                    else:   # DQN
                        a_p = self.network.choose_phase_action(o_p)
                        self.a_p_list[light].append(a_p)

                next_phase = int(a_p)
                self.phase_list[light].append(next_phase)

                # Decide next green time
                if not self.use_time or (not self.train_model and not self.load_model):
                    a_t = np.array([0.])     # 经过下面的处理最终会得到20s
                else:
                    o_t = np.array(self.step_time_obs[light]).flatten().tolist()
                    self.o_t_list[light].append(o_t)       # 存最近两次决策obs(2, T*o_dim)
                    if self.pointer < self.learn_begin and not self.load_model:  # 随机填充
                        a_t = np.random.random(self.network.a1_dim) * 2 - 1
                    else:
                        a_t = self.network.choose_time_action(o_t)
                    if self.train_model:    # 加噪声
                        a_t = np.clip(np.random.normal(0, self.var, size=a_t.shape) + a_t, -1, 1)

                next_green = round((a_t[0] + 1) / 2 * (self.max_green - self.min_green) + self.min_green)
                self.green[light] = next_green
                self.a_t_list[light].append(a_t)

                if self.train_model:
                    reward = env.get_light_reward(light)
                    if self.use_adj and len(self.light_id) > 1:
                        reward = reward + 0.3 * env.get_adj_light_reward(light)
                    self.reward_list.append(reward)
                    if self.use_time and self.use_phase:
                        if len(self.o_t_list[light]) >= 2 and len(self.o_p_list[light]) >= 2:
                            self.network.store_transition(self.o_t_list[light][-2], self.o_p_list[light][-2],
                                                          self.a_t_list[light][-2], self.a_p_list[light][-2], reward,
                                                          self.o_t_list[light][-1], self.o_p_list[light][-1])
                    elif self.use_time:
                        if len(self.o_t_list[light]) >= 2:
                            self.network.store_transition(self.o_t_list[light][-2], self.a_t_list[light][-2], reward,
                                                          self.o_t_list[light][-1])
                    else:  # only phase or neither
                        if len(self.o_p_list[light]) >= 2:
                            self.network.store_transition(self.o_p_list[light][-2], self.a_p_list[light][-2], reward,
                                                          self.o_p_list[light][-1])
                    if self.train_model and self.pointer >= self.learn_begin:
                        self.var = max(0.01, self.var * 0.99)  # 0.9-40 0.99-400 0.999-4000
                        self.network.learn()

                if self.phase_list[light][-2] == self.phase_list[light][-1]:
                    skip_yr = self.yellow + self.red + self.green[light]
                    env.set_light_action(light, self.phase_list[light][-1] * 3, skip_yr)   # 本该亮黄灯的继续亮绿灯
                    self.time_index[light], self.color[light] = skip_yr, 'g'
                else:
                    env.set_light_action(light, self.phase_list[light][-2] * 3 + 1, self.yellow)
                    self.time_index[light], self.color[light] = self.yellow, 'y'

        self.time_index[light] -= 1
        return next_green, next_phase

    def reset(self):
        self.time_index = {light: 0 for light in self.light_id}
        self.green = {light: self.min_green for light in self.light_id}
        self.color = {light: 'g' for light in self.light_id}
        self.phase_list = {light: deque([0], maxlen=2) for light in self.light_id}

        self.step_time_obs = {light: deque([[0] * self.o_t for _ in range(self.T_t)], maxlen=self.T_t) for light in self.light_id}
        self.o_t_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_t_list = {light: deque(maxlen=2) for light in self.light_id}
        self.step_phase_obs = {light: deque([[0] * self.o_p for _ in range(self.T_p)], maxlen=self.T_p) for light in self.light_id}
        self.o_p_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_p_list = {light: deque(maxlen=2) for light in self.light_id}
        self.reward_list = []


class ManagerLightAgent:
    def __init__(self, light_id, config, get_worker_oa, worker_choose_action):
        if isinstance(light_id, str):
            self.holon_name = light_id
            self.light_id = [light_id]
        elif isinstance(light_id, (list, tuple)):
            self.holon_name = 'h_' + light_id[0]
            self.light_id = list(light_id)
        self.get_worker_oa = get_worker_oa
        self.worker_choose_action = worker_choose_action

        self.use_time = config['use_time']
        self.use_phase = config['use_phase']
        self.train_model = config['train_model']
        self.load_model = config['load_model_name'] is not None
        self.lstm_observe_every_step = config['lstm_observe_every_step']

        self.use_adj = config['use_adj']
        self.lane_agent = config['lane_agent']
        self.half_goal = config['goal_only_indicates_state_mean']

        config['memory_capacity'] = config['memory_capacity'] * len(self.light_id)  # 控制多路口会导致存速翻倍，故扩大容量以匹配
        if self.lane_agent:
            config['memory_capacity'] = config['memory_capacity'] * 8   # 每次只给一个车道制定目标也会导致存速翻倍
            self.lane_state_dim = config['vehicle']['act_dim']
        else:
            assert config['vehicle']['act_dim'] % 8 == 0, '一次定8个车道的目标，则要求vehicle:act_dim能被8整除'
            self.lane_state_dim = config['vehicle']['act_dim'] / 8    # 表示每条车道的状态维度
        if self.half_goal:
            self.lane_state_dim *= 2

        self.o_g = config['vehicle']['obs_dim']     # 因为要临时改掉使得网络init时用原obs+目标编码，但step_goal_obs的size仍是原obs
        self.a_g = config['vehicle']['act_dim']     # 网络输出的动作大小。
        config['vehicle']['obs_dim'] = config['vehicle']['obs_dim'] + self.lane_state_dim + 8  # note:临时修改当前的obs：加上状态编码目标

        self.encoder = Encoder(2, self.lane_state_dim // 2)    # 例如Encoder用的dim=4，那么实际出来的状态是8，因为是mean+logvar
        self.encoder.load('../model/' + config['encoder_load_path'] + '/encoder_dim_' + str(self.lane_state_dim // 2))

        if self.use_time and self.use_phase:
            self.light_opt = 'both'
            self.network = HATD3Triple(config)  # tpg
        elif not self.use_time and not self.use_phase:
            self.light_opt = 'neither'
            self.network = ManagerTD3(config)
        else:
            self.light_opt = 'phase' if self.use_phase else 'time'
            self.network = HATD3Double(config, self.light_opt)

        self.save = lambda path, ep: self.network.save(path + 'light_agent_' + self.holon_name + '_ep_' + str(ep))
        if self.load_model:
            load_ep = str(config['load_model_ep']) if config['load_model_ep'] else '99'
            self.network.load('../model/' + config['load_model_name'] + '/light_agent_' + self.holon_name + '_ep_' + load_ep)

        self.var = config['var']
        self.o_t = config['time']['obs_dim']
        self.a_t = config['time']['act_dim']
        self.T_t = config['time']['T']
        self.o_p = config['phase']['obs_dim']
        self.a_p = config['phase']['act_dim']
        self.T_p = config['phase']['T']
        self.T_g = config['vehicle']['T']

        self.min_green = config['min_green']
        self.max_green = config['max_green']
        self.yellow = config['yellow']
        self.red = config['red']

        self.time_index = {light: 0 for light in self.light_id}
        self.green = {light: config['min_green'] for light in self.light_id}
        self.color = {light: 'g' for light in self.light_id}
        self.phase_list = {light: deque([0], maxlen=2) for light in self.light_id}

        self.step_time_obs = {light: deque([[0] * self.o_t for _ in range(self.T_t)], maxlen=self.T_t) for light in self.light_id}
        self.o_t_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_t_list = {light: deque(maxlen=2) for light in self.light_id}
        self.step_phase_obs = {light: deque([[0] * self.o_p for _ in range(self.T_p)], maxlen=self.T_p) for light in self.light_id}
        self.o_p_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_p_list = {light: deque(maxlen=2) for light in self.light_id}
        self.step_goal_obs = {light: deque([[0] * self.o_g for _ in range(self.T_g)], maxlen=self.T_g) for light in self.light_id}
        self.o_g_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_g_list = {light: deque(maxlen=2) for light in self.light_id}
        self.reward_list = []
        self.accumulate_reward_manager = {light: [] for light in self.light_id}
        self.step_encode_obs = {light: deque([[[0.] * self.lane_state_dim for l in range(8)] for t in range(self.T_g)], maxlen=self.T_g) for light in self.light_id}

    @property
    def pointer(self):
        return self.network.pointer

    @property
    def learn_begin(self):
        return self.network.learn_begin

    def step(self, env):
        tl, pl, gl = [], [], []
        for light in self.light_id:
            t, p, g = self._step(env, light)
            tl.append(t)
            pl.append(p)
            gl.append(g)
        return tl, pl, gl

    def _step(self, env, light):
        my_obs = env.get_light_obs(light)   # 8
        add_phase_id = np.eye(4)[int(self.phase_list[light][-1])].tolist()               # 4
        remain_green = (self.green[light] if self.color[light] in 'yr' else self.time_index[light]) / (env.base_cycle_length / 4)   # 1
        if self.use_adj and len(self.light_id) > 1:
            adj_obs = [_ * 0.3 for _ in env.get_adj_light_obs(light)]  # 16
            add_light_id = np.eye(len(self.light_id))[int(self.light_id.index(light))].tolist()  # 4
            obs = my_obs + adj_obs + add_light_id + add_phase_id + [remain_green]
        else:
            obs = my_obs + add_phase_id + [remain_green]

        if self.lstm_observe_every_step:
            self.step_time_obs[light].append(obs)
            self.step_phase_obs[light].append(obs)
        # obs += env.get_goal_obs(light)        # 这里goal_obs是车道均速和头cav，因此弃用
        self.step_goal_obs[light].append(obs)   # goal只看变灯时刻状态不合理，应该每秒都看

        encoded = np.array([self.encoder.encode(env.get_lane_obs(lane)) for lane in env.light_get_lane(light)]).tolist()
        self.step_encode_obs[light].append(encoded)

        self.accumulate_reward_manager[light].append([env.get_lane_halt(lane) for lane in env.light_get_lane(light)])

        next_green, next_phase, vehicle_goal = None, None, None
        if self.time_index[light] == 0:
            if self.color[light] == 'y' and self.red != 0:  # 黄灯结束切红灯
                env.set_light_action(light, self.phase_list[light][-2] * 3 + 2, self.red)
                self.time_index[light], self.color[light] = self.red, 'r'
            elif self.color[light] == 'r' or (self.color[light] == 'y' and self.red == 0):  # 红灯结束或（黄灯结束且无全红相位）切绿灯
                env.set_light_action(light, self.phase_list[light][-1] * 3, self.green[light])
                self.time_index[light], self.color[light] = self.green[light], 'g'
            elif self.color[light] == 'g':
                if not self.lstm_observe_every_step:
                    assert self.network.t1 == 1, 'LSTM should not be used when lstm_observe_every_step is False'
                    self.step_time_obs[light] = obs
                    self.step_phase_obs[light] = obs

                # Choose next phase
                if not self.use_phase or (not self.train_model and not self.load_model):
                    a_p = (self.phase_list[light][-1] + 1) % 4     # 不控制时默认动作
                else:
                    o_p = np.array(self.step_phase_obs[light]).flatten().tolist()
                    self.o_p_list[light].append(o_p)       # 存最近两次决策obs(2, T*o_dim)

                    if self.train_model:  # 加噪声
                        if self.pointer < self.learn_begin and not self.load_model:  # 随机填充
                            a_p = np.random.random(self.a_p) * 2 - 1
                        else:
                            a_p = self.network.choose_phase_action(o_p)
                        a_p = np.clip(np.random.normal(0, self.var, size=a_p.shape) + a_p, -1, 1)
                    else:
                        a_p = self.network.choose_phase_action(o_p)
                    result = np.zeros_like(a_p)
                    result[np.argmax(a_p)] = 1.
                    # result = a_p      # note 最后要确定代码之前记得试一下，如果存phase的原始动作是什么效果
                    self.a_p_list[light].append(result)
                    a_p = np.argmax(a_p)    # 4个里面最大值的索引
                next_phase = int(a_p)
                self.phase_list[light].append(next_phase)

                # Decide next green time
                if not self.use_time or (not self.train_model and not self.load_model):
                    a_t = np.array([0.])     # 经过下面的处理最终会得到20s
                else:
                    o_t = np.array(self.step_time_obs[light]).flatten().tolist()
                    self.o_t_list[light].append(o_t)       # 存最近两次决策obs(2, T*o_dim)

                    if self.train_model:    # 加噪声
                        if self.pointer < self.learn_begin and not self.load_model:  # 随机填充
                            a_t = np.random.random(self.a_t) * 2 - 1
                        else:
                            a_t = self.network.choose_time_action(o_t)
                        a_t = np.clip(np.random.normal(0, self.var, size=a_t.shape) + a_t, -1, 1)
                    else:
                        a_t = self.network.choose_time_action(o_t)
                    self.a_t_list[light].append(a_t)
                next_green = round((a_t[0] + 1) / 2 * (self.max_green - self.min_green) + self.min_green)
                self.green[light] = next_green

                # Decide next vehicle goal
                g_obs = np.array(self.step_goal_obs[light])       # T, o_dim
                enc = np.array(self.step_encode_obs[light])     # T, 8, g_dim

                reward = env.get_light_reward(light)
                if self.use_adj and len(self.light_id) > 1:
                    reward = reward + 0.3 * env.get_adj_light_reward(light)
                halt = np.array(self.accumulate_reward_manager[light])  # T, 8
                length = halt.shape[0]
                g_reward_8lane = np.sum(halt, axis=0) / length if length > 0 else 0
                # wait = np.array([env.get_lane_wait(lane) for lane in env.light_get_lane(light)])
                # g_reward_8lane += wait   # 除2/30/60？   # here，暂时不加wait，不然导致边界动作
                self.accumulate_reward_manager[light] = []

                worker_obs, worker_act = self.get_worker_oa(light)

                if self.lane_agent:
                    curr_o_g_list, curr_a_g_list, curr_r_list = [], [], []
                    for i in range(len(env.light_get_lane(light))):
                        one_hot_lane_id = np.eye(len(env.light_get_lane(light)))[i].tolist()   # T, 8
                        add_lane_id = np.array([one_hot_lane_id for _ in range(self.T_g)])
                        o_g = np.concatenate((g_obs, enc[:, i, :], add_lane_id), axis=1).flatten().tolist()
                        curr_o_g_list.append(o_g)

                        if self.train_model:    # 加噪声
                            if self.pointer < self.learn_begin and not self.load_model:  # 随机填充
                                a_g = np.random.random(self.a_g) * 2 - 1
                            else:
                                a_g = self.network.choose_goal(o_g)
                            a_g = np.clip(np.random.normal(0, self.var, size=a_g.shape) + a_g, -1, 1)
                        else:
                            a_g = self.network.choose_goal(o_g)
                        curr_a_g_list.append(a_g)   # [-1,1]

                        g_reward = g_reward_8lane[i]
                        if self.light_opt == 'both':    # 3Actor
                            reward = (2 * reward + g_reward) / 3
                            if len(self.o_t_list[light]) >= 2 and len(self.o_p_list[light]) >= 2 and len(self.o_g_list[light]) >= 2:
                                self.network.store_transition(self.o_t_list[light][-2], self.o_p_list[light][-2], self.o_g_list[light][-2][i],
                                                              self.a_t_list[light][-2], self.a_p_list[light][-2], self.a_g_list[light][-2][i], reward,
                                                              self.o_t_list[light][-1], self.o_p_list[light][-1], o_g)
                        elif self.light_opt == 'time':
                            reward = (reward + g_reward) / 2
                            if len(self.o_t_list[light]) >= 2 and len(self.o_g_list[light]) >= 2:
                                self.network.store_transition(self.o_t_list[light][-2], self.o_g_list[light][-2][i],
                                                              self.a_t_list[light][-2], self.a_g_list[light][-2][i], reward,
                                                              self.o_t_list[light][-1], o_g)
                        elif self.light_opt == 'phase':
                            reward = (reward + g_reward) / 2
                            if len(self.o_p_list[light]) >= 2 and len(self.o_g_list[light]) >= 2:
                                self.network.store_transition(self.o_p_list[light][-2], self.o_g_list[light][-2][i],
                                                              self.a_p_list[light][-2], self.a_g_list[light][-2][i], reward,
                                                              self.o_p_list[light][-1], o_g)
                        else:   # self.light_opt == 'neither':   # only goalTD3
                            reward = g_reward
                            if len(self.o_g_list[light]) >= 2 and len(worker_obs) > 0:
                                self.network.store_transition(self.o_g_list[light][-2][i], self.a_g_list[light][-2][i],
                                                              reward, o_g, worker_obs, worker_act)
                        curr_r_list.append(reward)
                    self.o_g_list[light].append(curr_o_g_list)  # 各lane收集完了再统一更新这个时步的
                    self.a_g_list[light].append(curr_a_g_list)
                    self.reward_list.append(sum(curr_r_list) / len(curr_r_list))
                    vehicle_goal = np.array(curr_a_g_list)
                else:
                    enc_flatten = np.reshape(enc, (len(g_obs), -1))
                    o_g = np.concatenate((g_obs, enc_flatten), axis=1).flatten().tolist()   # o_dim+8*g_dim
                    self.o_g_list[light].append(o_g)

                    if self.train_model:  # 加噪声
                        if self.pointer < self.learn_begin and not self.load_model:  # 随机填充
                            a_g = np.random.random(self.a_g) * 2 - 1
                        else:
                            a_g = self.network.choose_goal(o_g)
                        a_g = np.clip(np.random.normal(0, self.var, size=a_g.shape) + a_g, -1, 1)
                    else:
                        a_g = self.network.choose_goal(o_g)
                    self.a_g_list[light].append(a_g)  # [-1,1]

                    g_reward = sum(g_reward_8lane)
                    if self.light_opt == 'both':  # 3Actor
                        reward = (2 * reward + g_reward) / 3
                        if len(self.o_t_list[light]) >= 2 and len(self.o_p_list[light]) >= 2 and len(
                                self.o_g_list[light]) >= 2:
                            self.network.store_transition(self.o_t_list[light][-2], self.o_p_list[light][-2], self.o_g_list[light][-2],
                                                          self.a_t_list[light][-2], self.a_p_list[light][-2], self.a_g_list[light][-2],
                                                          reward, self.o_t_list[light][-1], self.o_p_list[light][-1], o_g)
                    elif self.light_opt == 'time':
                        reward = (reward + g_reward) / 2
                        if len(self.o_t_list[light]) >= 2 and len(self.o_g_list[light]) >= 2:
                            self.network.store_transition(self.o_t_list[light][-2], self.o_g_list[light][-2],
                                                          self.a_t_list[light][-2], self.a_g_list[light][-2],
                                                          reward, self.o_t_list[light][-1], o_g)
                    elif self.light_opt == 'phase':
                        reward = (reward + g_reward) / 2
                        if len(self.o_p_list[light]) >= 2 and len(self.o_g_list[light]) >= 2:
                            self.network.store_transition(self.o_p_list[light][-2], self.o_g_list[light][-2],
                                                          self.a_p_list[light][-2], self.a_g_list[light][-2],
                                                          reward, self.o_p_list[light][-1], o_g)
                    else:  # self.light_opt == 'neither':   # only goalTD3
                        reward = g_reward
                        if len(self.o_g_list[light]) >= 2 and len(worker_obs) > 0:
                            self.network.store_transition(self.o_g_list[light][-2], self.a_g_list[light][-2],
                                                          reward, o_g, worker_obs, worker_act)
                    self.reward_list.append(reward)

                if self.train_model and self.pointer >= self.learn_begin:
                    self.var = max(0.01, self.var * 0.99)  # 0.9-40 0.99-400 0.999-4000
                    self.network.learn(self.worker_choose_action)

                if self.phase_list[light][-2] == self.phase_list[light][-1]:
                    skip_yr = self.yellow + self.red + self.green[light]
                    env.set_light_action(light, self.phase_list[light][-1] * 3, skip_yr)   # 本该亮黄灯的继续亮绿灯
                    self.time_index[light], self.color[light] = skip_yr, 'g'
                else:
                    env.set_light_action(light, self.phase_list[light][-2] * 3 + 1, self.yellow)
                    self.time_index[light], self.color[light] = self.yellow, 'y'

        self.time_index[light] -= 1
        return next_green, next_phase, vehicle_goal

    def reset(self):
        self.time_index = {light: 0 for light in self.light_id}
        self.green = {light: self.min_green for light in self.light_id}
        self.color = {light: 'g' for light in self.light_id}
        self.phase_list = {light: deque([0], maxlen=2) for light in self.light_id}

        self.step_time_obs = {light: deque([[0] * self.o_t for _ in range(self.T_t)], maxlen=self.T_t) for light in self.light_id}
        self.o_t_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_t_list = {light: deque(maxlen=2) for light in self.light_id}
        self.step_phase_obs = {light: deque([[0] * self.o_p for _ in range(self.T_p)], maxlen=self.T_p) for light in self.light_id}
        self.o_p_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_p_list = {light: deque(maxlen=2) for light in self.light_id}
        self.step_goal_obs = {light: deque([[0] * self.o_g for _ in range(self.T_g)], maxlen=self.T_g) for light in self.light_id}
        self.o_g_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_g_list = {light: deque(maxlen=2) for light in self.light_id}
        self.reward_list = []
        self.accumulate_reward_manager = {light: [] for light in self.light_id}
        self.step_encode_obs = {light: deque([[[0.] * self.lane_state_dim for l in range(8)] for t in range(self.T_g)], maxlen=self.T_g) for light in self.light_id}


"""
    车辆智能体
"""


class IndependentCavAgent:
    """理论上整个路网只用一个cav智能体，但时间所限不改这个了，目前是一个分区一个cav智能体。"""
    def __init__(self, light_id, config):
        if isinstance(light_id, str):
            self.holon_name = light_id
            self.light_id = [light_id]
        elif isinstance(light_id, (list, tuple)):
            self.holon_name = 'h_' + light_id[0]
            self.light_id = list(light_id)

        self.ctrl_all_lane = not config['only_ctrl_curr_phase']
        self.ctrl_lane_num = 8 if self.ctrl_all_lane else 2  # 每个时刻控制的入口车道数。每一时刻都控制所有方向的车道
        self.ctrl_all_cav = not config['only_ctrl_head_cav']

        self.use_CAV = config['use_CAV']
        self.train_model = config['train_model']
        self.load_model = config['load_model_name'] is not None

        config['memory_capacity'] = config['memory_capacity'] * len(self.light_id)  # 控制多路口会导致存速翻倍，故扩大容量以匹配

        self.network = TD3Single(config, 'cav')
        self.save = lambda path, ep: self.network.save(path + 'cav_agent_' + self.holon_name + '_ep_' + str(ep))
        if self.load_model:
            load_ep = str(config['load_model_ep']) if config['load_model_ep'] else '99'
            self.network.load('../model/' + config['load_model_name'] + '/cav_agent_' + self.holon_name + '_ep_' + load_ep)

        self.var = config['var']
        self.T = config['cav']['T']

        self.ctrl_cav = {light: deque([[None] * self.ctrl_lane_num], maxlen=2) for light in self.light_id}
        self.global_income_cav = deque([[], []], maxlen=2)
        self.next_phase = {light: 1 for light in self.light_id}

        self.trans_buffer = {}
        self.reward_list = []

    @property
    def pointer(self):
        return self.network.pointer

    @property
    def learn_begin(self):
        return self.network.learn_begin

    def step(self, env, goal, next_phase):
        real, next_a = [], []
        if self.use_CAV:
            global_income_cav = []
            for light in self.light_id:
                if self.ctrl_all_cav:
                    curr_cav = [car for lane in env.light_get_lane(light) for car in env.lane_get_cav(lane, head_only=False)]
                else:
                    curr_cav = env.light_get_head_cav(light, self.next_phase[light], curr_phase=not self.ctrl_all_lane)
                global_income_cav.extend(curr_cav)
                self.ctrl_cav[light].append(curr_cav)
            self.global_income_cav.append(global_income_cav)
        for light_idx, light in enumerate(self.light_id):
            r, n = self._step(env, goal[light_idx], next_phase[light_idx], light)
            real.append(r)
            next_a.append(n)
        return real, next_a

    def _step(self, env, goal, next_phase, light):
        next_acc, real_a = None, None

        if self.use_CAV:
            if next_phase is not None:    # 说明上层切相位了，接下来是一对新车道的yrg
                self.next_phase[light] = next_phase

            # 对比两时刻头CAV，上时刻还有现在没了(可能切相位或驶出)的要reset一下跟驰
            for cav_id in self.ctrl_cav[light][-2]:
                if cav_id is not None and cav_id not in self.global_income_cav[-1]:
                    env.reset_head_cav(cav_id)
                    self.reward_list.append(self.trans_buffer[cav_id]['reward'])

                    del self.trans_buffer[cav_id]

            for cav_id in self.ctrl_cav[light][-1]:
                if cav_id:  # cav is not None
                    o_v = env.get_head_cav_obs(cav_id)  # list

                    if cav_id not in self.trans_buffer:  # == 0
                        self.trans_buffer[cav_id] = {'obs': [o_v],  # 存储车辆每一步的obs
                                                     'action': [],  # 每一步的action
                                                     'real_acc': [],  # 每一步的action
                                                     'reward': []}
                    else:  # >=1
                        self.trans_buffer[cav_id]['obs'].append(o_v)

                    cav_obs = self.trans_buffer[cav_id]['obs']
                    if len(cav_obs) >= self.T:  # 没存满就先不控制
                        if self.train_model:  # 加噪声
                            if self.pointer < self.learn_begin and not self.load_model:  # 随机填充
                                a_v = np.random.random(self.network.a_dim) * 2 - 1
                            else:
                                a_v = self.network.choose_action(cav_obs[-self.T:])
                            a_v = np.clip(np.random.normal(0, self.var, size=a_v.shape) + a_v, -1, 1)
                        else:
                            a_v = self.network.choose_action(cav_obs[-self.T:])
                        self.trans_buffer[cav_id]['action'].append(a_v)
                        next_acc = a_v[0]    # [-1,1]
                        real_a = cav_obs[-1][2]    # [-?,1]
                        self.trans_buffer[cav_id]['real_acc'].append(real_a)   # 获取的是上一时步的实际acc

                        reward = env.get_cav_reward(cav_obs[-1], self.trans_buffer[cav_id]['real_acc'][-2],
                                                    self.trans_buffer[cav_id]['action'][-2]) if len(cav_obs) >= 1 + self.T else 0
                        self.trans_buffer[cav_id]['reward'].append(reward)

                        if self.train_model and len(cav_obs) >= self.T + 1:
                            self.network.store_transition(np.array(cav_obs[-self.T - 1: -1]).flatten(),
                                                          # self.trans_buffer[cav_id]['action'][-2],
                                                          self.trans_buffer[cav_id]['real_acc'][-1],    # 当前时刻的real_acc存的是上一时刻动作的真实效果
                                                          self.trans_buffer[cav_id]['reward'][-1],
                                                          np.array(cav_obs[-self.T:]).flatten())

                        env.set_head_cav_action(cav_id, cav_obs[-1][1], next_acc)
                        # print('cav obs:', cav_obs[-1], 'a:', next_acc, 'r:', self.trans_buffer[cav_id]['reward'][-1])
            if self.train_model and self.pointer >= self.learn_begin:
                self.var = max(0.01, self.var * 0.999)  # 0.9-40 0.99-400 0.999-4000
                self.network.learn()
        return (real_a, next_acc) if not real_a or not next_acc else (real_a * env.max_acc, next_acc * env.max_acc)

    def reset(self):
        self.ctrl_cav = {light: deque([[None] * self.ctrl_lane_num], maxlen=2) for light in self.light_id}
        self.global_income_cav = deque([[], []], maxlen=2)
        self.next_phase = {light: 1 for light in self.light_id}

        self.trans_buffer = {}
        self.reward_list = []


class WorkerCavAgent:
    def __init__(self, light_id, config):
        if isinstance(light_id, str):
            self.holon_name = light_id
            self.light_id = [light_id]
        elif isinstance(light_id, (list, tuple)):
            self.holon_name = 'h_' + light_id[0]
            self.light_id = list(light_id)

        self.ctrl_all_lane = not config['only_ctrl_curr_phase']
        self.ctrl_lane_num = 8 if self.ctrl_all_lane else 2  # 每个时刻控制的入口车道数。每一时刻都控制所有方向的车道
        self.ctrl_all_cav = not config['only_ctrl_head_cav']

        self.lane_agent = config['lane_agent']
        self.half_goal = config['goal_only_indicates_state_mean']

        self.use_CAV = config['use_CAV']
        self.train_model = config['train_model']
        self.load_model = config['load_model_name'] is not None

        config['memory_capacity'] = config['memory_capacity'] * len(self.light_id)  # 控制多路口会导致存速翻倍，故扩大容量以匹配

        self.lane_state_dim = config['high_goal_dim'] if self.lane_agent else config['high_goal_dim'] / 8     # 上层动作维度
        # 现在确定用VAE方案，否则普通AE这里需要改        # 上层动作：high_goal_dim，车道状态：self.lane_state_dim
        if self.half_goal:  # 若上层只管均值
            self.lane_state_dim *= 2     # 车道状态维度

        self.encoder = Encoder(2, self.lane_state_dim // 2)    # 例如Encoder用的dim=4，那么实际出来的状态是8，因为是mean+logvar
        self.encoder.load('../model/' + config['encoder_load_path'] + '/encoder_dim_' + str(self.lane_state_dim // 2))

        config['cav']['obs_dim'] = config['cav']['obs_dim'] + self.lane_state_dim  # note 临时加上目标
        config['goal_dim'] = self.lane_state_dim + 2    # 下层输入actor的“车辆goal”维度
        # config['goal_dim'] = 2    # here:可以一试

        self.network = WorkerTD3(config)
        self.save = lambda path, ep: self.network.save(path + 'cav_agent_' + self.holon_name + '_ep_' + str(ep))
        if self.load_model:
            load_ep = str(config['load_model_ep']) if config['load_model_ep'] else '99'
            self.network.load('../model/' + config['load_model_name'] + '/cav_agent_' + self.holon_name + '_ep_' + load_ep)

        self.var = config['var']
        self.T = config['cav']['T']
        self.alpha = config['alpha']

        self.ctrl_cav = {light: deque([[None] * self.ctrl_lane_num], maxlen=2) for light in self.light_id}
        self.global_income_cav = deque([[], []], maxlen=2)
        self.next_phase = {light: 1 for light in self.light_id}
        self.goal_state = {light: np.zeros((self.ctrl_lane_num, self.lane_state_dim)) for light in self.light_id}
        self.lane_state = {light: deque([np.zeros((self.ctrl_lane_num, self.lane_state_dim))], maxlen=2) for light in self.light_id}

        self.trans_buffer = {}
        self.ext_reward_list = []
        self.int_reward_list = []
        self.reward_list = []
        self.for_manager = {light: {'obs': [], 'act': []} for light in self.light_id}

    @property
    def pointer(self):
        return self.network.pointer

    @property
    def learn_begin(self):
        return self.network.learn_begin

    def get_oa(self, light):
        self.for_manager[light] = {'obs': [[[-1] * self.network.o_dim for _ in range(8)] for _ in range(25)],
                                   'act': [[[-1] for _ in range(8)] for _ in range(25)]}   # note: 注意，这里使得OPC始终无效！！！！！
        obs_seq = self.for_manager[light]['obs']
        act_seq = self.for_manager[light]['act']
        self.for_manager[light] = {'obs': [], 'act': []}
        return obs_seq, act_seq

    def step(self, env, goal, next_phase):
        real, next_a = [], []
        if self.use_CAV:
            global_income_cav = []
            for light in self.light_id:
                if self.ctrl_all_cav:
                    curr_cav = [car for lane in env.light_get_lane(light) for car in env.lane_get_cav(lane, head_only=False)]
                else:
                    curr_cav = env.light_get_head_cav(light, self.next_phase[light], curr_phase=not self.ctrl_all_lane)
                global_income_cav.extend(curr_cav)
                self.ctrl_cav[light].append(curr_cav)

                encoded = np.array([self.encoder.encode(env.get_lane_obs(lane)) for lane in env.light_get_lane(light)])
                self.lane_state[light].append(encoded)

            self.global_income_cav.append(global_income_cav)
        for light_idx, light in enumerate(self.light_id):
            r, n = self._step(env, goal[light_idx], next_phase[light_idx], light)
            real.append(r)
            next_a.append(n)
        return real, next_a

    def _step(self, env, goal: np.ndarray, next_phase, light):
        next_acc, real_a = None, None

        if self.use_CAV:
            curr_headcav = env.light_get_head_cav(light, self.next_phase[light], curr_phase=not self.ctrl_all_lane)

            curr_lane = env.light_get_ctrl_lane(light, self.next_phase[light], curr_phase=False)    # 等效light_get_lane
            if goal is not None:    # 说明上层切相位了，接下来是一对新车道的yrg
                goal = goal.reshape(8, -1)      # 注意改了这里，永久假设上层每次制定所有可受控车道的goal
                gs = self.lane_state[light][-1].copy()
                if self.half_goal:
                    gs[:, :goal.shape[1]] += goal
                else:
                    gs += goal
                self.goal_state[light] = np.clip(gs, 0, 1)  # clip
                self.next_phase[light] = next_phase

            # 对比两时刻头CAV，上时刻还有现在没了(可能切相位或驶出)的要reset一下跟驰
            for cav_id in self.ctrl_cav[light][-2]:
                if cav_id is not None and cav_id not in self.global_income_cav[-1]:
                    env.reset_head_cav(cav_id)
                    self.ext_reward_list.append(self.trans_buffer[cav_id]['ext_reward'])
                    self.int_reward_list.append(self.trans_buffer[cav_id]['int_reward'])
                    self.reward_list.append(self.trans_buffer[cav_id]['reward'])

                    del self.trans_buffer[cav_id]

            curr_delta = self.goal_state[light] - self.lane_state[light][-1]    # 当前的差分goal，用于计算内在奖励
            curr_all_lane_obs, curr_all_lane_act = [], []   # 用于保存8车道头车的o & a
            for cav_id in self.ctrl_cav[light][-1]:
                o_v = env.get_head_cav_obs(cav_id)  # list
                lane_id = curr_lane.index(env.cav_get_lane(cav_id))
                o_v = o_v + self.lane_state[light][-1][lane_id].tolist()  # note：是否有必要将编码形式的状态告诉CAV？值得对比

                a_v_for_manager = -1
                if cav_id:  # cav is not None
                    if cav_id not in self.trans_buffer:  # == 0
                        self.trans_buffer[cav_id] = {'obs': [o_v],  # 存储车辆每一步的obs
                                                     'action': [],  # 每一步的action
                                                     'real_acc': [],  # 每一步的action
                                                     'goal': deque(maxlen=2),
                                                     'ext_reward': [],
                                                     'int_reward': [],
                                                     'reward': []}
                    else:  # >=1
                        self.trans_buffer[cav_id]['obs'].append(o_v)

                    cav_obs = self.trans_buffer[cav_id]['obs']
                    if len(cav_obs) >= self.T:  # 没存满就先不控制
                        curr_adv_v = self.encoder.decode(self.goal_state[light][lane_id], [o_v[0]]).flatten().tolist()
                        next_loc = (o_v[0] * env.base_lane_length + o_v[1] * env.max_speed) / env.base_lane_length
                        next_adv_v = self.encoder.decode(self.goal_state[light][lane_id], [next_loc]).flatten().tolist()
                        g_v = self.goal_state[light][lane_id].flatten().tolist() + curr_adv_v + next_adv_v
                        self.trans_buffer[cav_id]['goal'].append(g_v)

                        if self.train_model:  # 加噪声
                            if self.pointer < self.learn_begin and not self.load_model:  # 随机填充
                                a_v = np.random.random(self.network.a_dim) * 2 - 1
                            else:
                                a_v = self.network.choose_action(cav_obs[-self.T:], g_v)
                            a_v = np.clip(np.random.normal(0, self.var, size=a_v.shape) + a_v, -1, 1)
                        else:
                            a_v = self.network.choose_action(cav_obs[-self.T:], g_v)
                        self.trans_buffer[cav_id]['action'].append(a_v)
                        a_v_for_manager = a_v[0]
                        next_acc = a_v[0]    # [-1,1]
                        real_a = cav_obs[-1][2]    # [-?,1]
                        self.trans_buffer[cav_id]['real_acc'].append(real_a)   # 获取的是上一时步的实际acc

                        int_reward = -np.sqrt(np.sum(curr_delta[lane_id] ** 2))     # 同一条车道上车辆的inner reward相同
                        int_reward += -abs(curr_adv_v[0] - o_v[1])                     # note:这里是新增的哦！！！！！！！！！！！！！！！
                        ext_reward = env.get_cav_reward(cav_obs[-1], self.trans_buffer[cav_id]['real_acc'][-2],
                                                        self.trans_buffer[cav_id]['action'][-2]) if len(cav_obs) >= 1 + self.T else 0
                        reward = (1 - self.alpha) * ext_reward + self.alpha * int_reward
                        self.trans_buffer[cav_id]['int_reward'].append(int_reward)
                        self.trans_buffer[cav_id]['ext_reward'].append(ext_reward)
                        self.trans_buffer[cav_id]['reward'].append(reward)

                        if self.train_model and len(cav_obs) >= self.T + 1:  # encoder稳定了才能存
                            self.network.store_transition(np.array(cav_obs[-self.T - 1: -1]).flatten(),
                                                          # self.trans_buffer[cav_id]['action'][-2],    # here,要对比吗？可以不用
                                                          self.trans_buffer[cav_id]['real_acc'][-1],    # 当前时刻的real_acc存的是上一时刻动作的真实效果
                                                          self.trans_buffer[cav_id]['goal'][-2],
                                                          self.trans_buffer[cav_id]['goal'][-1],    # next_goal
                                                          self.trans_buffer[cav_id]['reward'][-1],
                                                          np.array(cav_obs[-self.T:]).flatten())

                        env.set_head_cav_action(cav_id, cav_obs[-1][1], next_acc)
                        # print('cav obs:', cav_obs[-1], 'a:', next_acc, 'r:', self.trans_buffer[cav_id]['reward'][-1])

                if cav_id is not None and cav_id in curr_headcav:   # 若该车为头车，传出状态给opc
                    curr_all_lane_obs.append(o_v)
                    curr_all_lane_act.append([a_v_for_manager])
            for i in range(8 - len(curr_all_lane_act)):
                curr_all_lane_obs.append([-1] * self.network.o_dim)
                curr_all_lane_act.append([-1])
            self.for_manager[light]['obs'].append(curr_all_lane_obs)
            self.for_manager[light]['act'].append(curr_all_lane_act)

            if self.train_model and self.pointer >= self.learn_begin:
                self.var = max(0.01, self.var * 0.999)  # 0.9-40 0.99-400 0.999-4000
                self.network.learn()

        return (real_a, next_acc) if not real_a or not next_acc else (real_a * env.max_acc, next_acc * env.max_acc)

    def reset(self):
        self.ctrl_cav = {light: deque([[None] * self.ctrl_lane_num], maxlen=2) for light in self.light_id}
        self.global_income_cav = deque([[], []], maxlen=2)
        self.next_phase = {light: 1 for light in self.light_id}
        self.lane_state = {light: deque([np.zeros((self.ctrl_lane_num, self.lane_state_dim))], maxlen=2) for light in self.light_id}
        self.goal_state = {light: np.zeros((self.ctrl_lane_num, self.lane_state_dim)) for light in self.light_id}

        self.trans_buffer = {}
        self.ext_reward_list = []
        self.int_reward_list = []
        self.reward_list = []
        self.for_manager = {light: {'obs': [], 'act': []} for light in self.light_id}


class LoyalCavAgent:
    def __init__(self, light_id, config):
        if isinstance(light_id, str):
            self.holon_name = light_id
            self.light_id = [light_id]
        elif isinstance(light_id, (list, tuple)):
            self.holon_name = 'h_' + light_id[0]
            self.light_id = list(light_id)

        self.network = WorkerTD3(config)    # 实例化一个网络但没用于控制，只是对外接口保持一致罢了
        self.save = lambda path, ep: print('Loyal Agent no need to save')

        self.use_CAV = config['use_CAV']
        self.train_model = config['train_model']
        self.load_model = config['load_model_name'] is not None

        self.lane_agent = config['lane_agent']
        self.half_goal = config['goal_only_indicates_state_mean']

        self.lane_state_dim = config['high_goal_dim'] if self.lane_agent else config['high_goal_dim'] / 8     # 上层动作维度
        if self.half_goal:  # 若上层只管均值
            self.lane_state_dim *= 2

        self.encoder = Encoder(2, self.lane_state_dim // 2)    # 例如Encoder用的dim=4，那么实际出来的状态是8，因为是mean+logvar
        self.encoder.load('../model/' + config['encoder_load_path'] + '/encoder_dim_' + str(self.lane_state_dim // 2))

        self.var = config['var']
        self.T = config['cav']['T']

        self.last_car_list = {light: [] for light in self.light_id}
        self.goal_state = {light: np.zeros((8, self.lane_state_dim)) for light in self.light_id}
        self.reward_list = []
        self.for_manager = {'obs': [[[-1] * self.network.o_dim for _ in range(8)] for _ in range(25)],
                            'act': [[[-1] for _ in range(8)] for _ in range(25)]}

    @property
    def pointer(self):
        return self.network.pointer

    @property
    def learn_begin(self):
        return self.network.learn_begin

    def get_oa(self, light):
        obs_seq = self.for_manager['obs']
        act_seq = self.for_manager['act']
        # self.for_manager = {'obs': [], 'act': []}
        return obs_seq, act_seq

    def step(self, env, goal, next_phase):
        for light_idx, light in enumerate(self.light_id):
            self._step(env, goal[light_idx], next_phase[light_idx], light)
        return [None] * len(self.light_id), [None] * len(self.light_id)

    def _step(self, env, goal, next_phase, light):
        next_acc, real_a = None, None

        if self.use_CAV:
            # encoded = np.array([self.encoder.encode(env.get_lane_obs(lane)) for lane in env.light_get_lane(light)])
            # self.lane_state[light].append(encoded)
            if goal is not None:
                goal = goal.reshape(8, -1)      # 注意改了这里，永久假设上层每次制定所有可受控车道的goal
                gs = np.array([self.encoder.encode(env.get_lane_obs(lane)) for lane in env.light_get_lane(light)])
                if self.half_goal:
                    gs[:, :len(goal)] += goal
                else:
                    gs += goal
                self.goal_state[light] = np.clip(gs, 0, 1)  # clip

            curr_car = []
            for lid, lane in enumerate(env.light_get_lane(light)):  # 必控制所有车道所有车(无论是不是CAV)
                lane_car = env.lane_get_all_car(lane)
                curr_car.extend(lane_car)

                for car in lane_car:
                    o_v = env.get_head_cav_obs(car)
                    next_loc = (o_v[0] * env.base_lane_length + o_v[1] * env.max_speed) / env.base_lane_length
                    # adv_v = self.encoder.decode(self.goal_state[light][lid], [o_v[0]])[0]  # 弃用：当前位置转瞬即逝
                    adv_v = self.encoder.decode(self.goal_state[light][lid], [next_loc])[0]

                    if adv_v > 1 or adv_v < 0:
                        print('decode_v=', adv_v * env.max_speed)
                    curr_tar_v = max(min(adv_v * env.max_speed, 1), 0)
                    env.set_lane_act_speed(car, curr_tar_v)

            for vehicle in self.last_car_list[light]:
                if vehicle not in curr_car:
                    env.reset_head_cav(vehicle)
            self.last_car_list[light] = curr_car

        return real_a, next_acc

    def reset(self):
        self.last_car_list = {light: [] for light in self.light_id}
        self.goal_state = {light: np.zeros((8, self.lane_state_dim)) for light in self.light_id}
        self.reward_list = []
        self.for_manager = {'obs': [[[-1] * 8 for _ in range(8)] for _ in range(25)],
                            'act': [[[-1] for _ in range(8)] for _ in range(25)]}

