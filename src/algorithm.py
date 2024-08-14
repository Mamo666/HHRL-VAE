
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.cuda.manual_seed(3407)
torch.manual_seed(3407)
np.random.seed(3407)  # 设置随机种子
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.is_available())


class DeepSetEncoder(nn.Module):
    def __init__(self, car_info_dim, g_dim, rho_dim=None, phi_dim=16, hidden_dim=64):
        super(DeepSetEncoder, self).__init__()
        self.phi = nn.Sequential(
            nn.Linear(car_info_dim, 16),
            nn.ReLU(),
            nn.Linear(16, phi_dim)
        )
        if rho_dim is not None:
            self.rho = nn.Sequential(
                nn.Linear(phi_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, rho_dim),
            )
            mid_dim = rho_dim
            self.use_rho = True
        else:
            mid_dim = phi_dim
            self.use_rho = False
        self.to_mean = nn.Linear(mid_dim, g_dim)
        self.to_logvar = nn.Linear(mid_dim, g_dim)

    def forward(self, x):
        x = self.phi(x)  # shape: (batch_size, num_elements, phi_dim)
        x = torch.sum(x, dim=-2)  # 聚合
        if self.use_rho:
            x = self.rho(x)
        mean = self.to_mean(x)
        logvar = self.to_logvar(x)
        return mean, logvar


class DeepSetDecoder(nn.Module):
    def __init__(self, g_dim, hidden_dim=128):
        super(DeepSetDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(g_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32)
        )
        self.loc = nn.Linear(1, 32)
        self.func = nn.Sequential(
            nn.Linear(32 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 添加Sigmoid激活函数，将输出限制在0-1之间
        )

    def forward(self, g, loc):
        g = self.decoder(g)
        loc = self.loc(loc)
        x = torch.cat((g, loc), dim=-1)  # shape: (batch_size, goal_dim + 1)
        x = self.func(x)  # shape: (batch_size, goal_dim+loc_dim->velocity_dim)
        return x


class Encoder(nn.Module):
    def __init__(self, car_info_dim, goal_dim):
        super(Encoder, self).__init__()
        self.goal_dim = goal_dim
        self.encoder = DeepSetEncoder(car_info_dim, goal_dim, rho_dim=64, phi_dim=32, hidden_dim=128).to(device)  # rhonew
        # self.encoder = DeepSetEncoder(car_info_dim, goal_dim, rho_dim=32, phi_dim=16, hidden_dim=32).to(device)
        self.decoder = DeepSetDecoder(goal_dim).to(device)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def learn(self, data: list):
        data = torch.FloatTensor(data).to(device)
        loc = data[..., :1]
        true_v = data[..., 1:]

        # update
        self.opt.zero_grad()
        mean, logvar = self.encoder(data)
        z = self.reparameterize(mean, logvar)
        decoded = self.decoder(z.unsqueeze(-2).repeat(1, loc.shape[-2], 1), loc)
        recon_loss = self.criterion(decoded, true_v)
        kl_loss = -0.5 * torch.sum(1. + logvar - mean.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        loss.backward()
        self.opt.step()
        return loss.item()

    def encode(self, x):
        """由于同一路口每车道的车辆数会不同，所以这里每次只能输入一个lane的数据，形状为(n,2)，输出g"""
        if len(x) == 0:
            return np.zeros((self.goal_dim * 2))
        x = torch.FloatTensor(x).to(device)
        mean, logvar = self.encoder(x)
        # return mean.cpu().data.numpy(), logvar.cpu().data.numpy()
        return np.concatenate((mean.cpu().data.numpy(), logvar.cpu().data.numpy()), axis=-1)

    # def decode(self, mean, logvar, loc):
    #     mean = torch.FloatTensor(mean).to(device)
    #     logvar = torch.FloatTensor(logvar).to(device)
    def decode(self, goal, loc):
        """每次输入一个(g_dim)的encoded_state和(1)的loc，不存在batch和lane维度，输出1个v"""
        mean = torch.FloatTensor(goal[..., :self.goal_dim]).to(device)
        logvar = torch.FloatTensor(goal[..., self.goal_dim:]).to(device)
        z = self.reparameterize(mean, logvar)
        x = torch.FloatTensor(loc).to(device)
        return self.decoder(z, x).cpu().data.numpy()

    def save(self, filename):
        torch.save(self.encoder.state_dict(), filename + "_enc")
        torch.save(self.decoder.state_dict(), filename + "_dec")
        torch.save(self.opt.state_dict(), filename + "_encoder_optimizer")

    def load(self, filename):
        # 加载模型
        self.encoder.load_state_dict(torch.load(filename + "_enc"))
        self.decoder.load_state_dict(torch.load(filename + "_dec"))
        self.opt.load_state_dict(torch.load(filename + "_encoder_optimizer"))


class LSTM(nn.Module):
    def __init__(self, obs_dim, out_dim, t, rnn_num_layer=1, use_bilstm=True):
        super(LSTM, self).__init__()
        self.T = t

        rnn_dim = out_dim   # 乱设的，可能有问题？
        self.obs_dim = obs_dim
        self.lstm_layer = nn.LSTM(input_size=obs_dim, hidden_size=rnn_dim, num_layers=rnn_num_layer, batch_first=True,
                                  bidirectional=use_bilstm)
        self.fc = nn.Linear(rnn_dim * 2 if use_bilstm else rnn_dim, out_dim)

    def forward(self, x):
        x = x.view(-1, self.T, self.obs_dim)
        r_out, _ = self.lstm_layer(x)
        x = self.fc(r_out[:, -1, :])
        return x


class Actor(nn.Module):  # 定义 actor 网络结构
    def __init__(self, obs_dim, state_dim, action_dim, t, hidden_dim, max_action=1):
        super(Actor, self).__init__()
        self.T = t

        if t > 1:
            self.lstm = LSTM(obs_dim, state_dim, t)
        else:
            state_dim = obs_dim

        self.l1 = nn.Linear(state_dim, hidden_dim[0])
        self.l2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l3 = nn.Linear(hidden_dim[1], action_dim)
        self.max_action = max_action

    def forward(self, s):
        if self.T > 1:
            s = self.lstm(s)
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


"""
    TD3 算法
"""


class CriticSingle(nn.Module):  # 定义 critic 网络结构
    def __init__(self, o_dim, s_dim, a_dim, t, hidden_dim):
        super(CriticSingle, self).__init__()
        self.t = t

        if t > 1:
            self.lstm = LSTM(o_dim, s_dim, t)
        else:
            s_dim = o_dim

        # Q1 architecture   计算 Q1
        self.l1 = nn.Linear(s_dim + a_dim, hidden_dim[0])
        self.l2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l3 = nn.Linear(hidden_dim[1], 1)

        # Q2 architecture   计算 Q2
        self.l4 = nn.Linear(s_dim + a_dim, hidden_dim[0])
        self.l5 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l6 = nn.Linear(hidden_dim[1], 1)

    def forward(self, state, action):  # 注意此处，直接把两个网络写在一起，这样就可以只用一个梯度下降优化器
        if self.t > 1:
            state = self.lstm(state)

        sa = torch.cat([state, action], 1)  # 将s和a横着拼接在一起

        x1 = F.relu(self.l1(sa))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)    # 直接输出线性计算后的值作为Q值

        x2 = F.relu(self.l4(sa))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, state, action):  # 新增一个Q值输出的方法，只使用其中一个网络的结果作为输出，避免重复计算
        if self.t > 1:
            state = self.lstm(state)

        sa = torch.cat([state, action], 1)

        x1 = F.relu(self.l1(sa))
        x1 = F.relu(self.l2(x1))
        q = self.l3(x1)
        return q


class TD3Single:
    def __init__(self, cfg, task):
        self.o_dim = cfg[task]['obs_dim']    # 单步观测维度
        self.a_dim = cfg[task]['act_dim']    # 动作维度
        self.s_dim = cfg[task]['state_dim']  # LSTM输出维度
        self.t = cfg[task]['T']

        self.obs_dim = self.o_dim * self.t
        self.a1_dim = self.a2_dim = self.a3_dim = self.a_dim      # 与HATD3一致，方便外部调用
        self.choose_time_action = self.choose_phase_action = self.choose_goal = self.choose_action      # 方便外部调用

        self.gamma = cfg['gamma']
        self.tau = cfg['tau']  # 软更新系数
        self.batch_size = cfg['batch_size']

        self.memory_capacity = cfg['memory_capacity']  # 记忆库大小
        self.learn_begin = self.memory_capacity * cfg['learn_start_ratio']   # 存满一定比例的记忆库之后开始学习并用网络输出动作
        self.memory = np.zeros((self.memory_capacity, self.obs_dim * 2 + self.a_dim + 1))
        self.pointer = 0

        # 创建对应的四个网络
        self.actor = Actor(self.o_dim, self.s_dim, self.a_dim, self.t, cfg[task]['hidden_dim']).to(device)
        self.actor_target = Actor(self.o_dim, self.s_dim, self.a_dim, self.t, cfg[task]['hidden_dim']).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())  # 存储网络名字和对应参数
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg['actor_learning_rate'])
        self.scheduler_actor = torch.optim.lr_scheduler.StepLR(self.actor_optimizer,
                                                               step_size=cfg['actor_scheduler_step'], gamma=0.5)

        self.critic = CriticSingle(self.o_dim, self.s_dim, self.a_dim, self.t, cfg[task]['hidden_dim']).to(device)
        self.critic_target = CriticSingle(self.o_dim, self.s_dim, self.a_dim, self.t, cfg[task]['hidden_dim']).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg['critic_learning_rate'])
        self.scheduler_critic = torch.optim.lr_scheduler.StepLR(self.critic_optimizer,
                                                                step_size=cfg['critic_scheduler_step'], gamma=0.5)

        self.policy_freq = 2
        self.total_it = 0

    def store_transition(self, o, a, r, o_):
        transition = np.hstack((o, a, r, o_))
        index = self.pointer % self.memory_capacity
        self.memory[index, :] = transition
        self.pointer += 1

    def choose_action(self, o):
        obs = torch.FloatTensor(o).view(1, -1, self.o_dim).to(device)
        return self.actor(obs).cpu().data.numpy().flatten()

    def learn(self):
        self.total_it += 1
        # mini batch sample
        indices = np.random.choice(min(self.pointer, self.memory_capacity), size=self.batch_size)   # 注意，这里是默认有放回
        batch_trans = self.memory[indices, :]

        obs = torch.FloatTensor(batch_trans[:, :self.obs_dim]).to(device)
        action = torch.FloatTensor(batch_trans[:, self.obs_dim: self.obs_dim + self.a_dim]).to(device)
        reward = torch.FloatTensor(batch_trans[:, -self.obs_dim - 1: -self.obs_dim]).to(device)
        next_obs = torch.FloatTensor(batch_trans[:, -self.obs_dim:]).to(device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * 0.2).clamp(-0.5, 0.5)       # noise=0.2, clip=0.5
            next_action = (self.actor_target(next_obs) + noise).clamp(-1, 1)    # 默认动作空间[-1,1]

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (self.gamma * target_Q).detach()

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.scheduler_critic.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(obs, self.actor(obs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.scheduler_actor.step()

    def save(self, filename):
        # 保存模型
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

    def load(self, filename):
        # 加载模型
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)


"""
    t+p or t/p+g的HATD3，有2个Actor
"""


class CriticDouble(nn.Module):  # 定义 critic 网络结构
    def __init__(self, o1_dim, s1_dim, t1, o2_dim, s2_dim, t2, joint_action_dim, hidden_dim):
        super(CriticDouble, self).__init__()
        self.t1 = t1
        self.t2 = t2

        if t1 > 1:
            self.lstm1 = LSTM(o1_dim, s1_dim, t1)
        else:
            s1_dim = o1_dim
        if t2 > 1:
            self.lstm2 = LSTM(o2_dim, s2_dim, t2)
        else:
            s2_dim = o2_dim

        # Q1 architecture   计算 Q1
        self.l1 = nn.Linear(s1_dim + s2_dim + joint_action_dim, hidden_dim[0])
        self.l2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l3 = nn.Linear(hidden_dim[1], 1)

        # Q2 architecture   计算 Q2
        self.l4 = nn.Linear(s1_dim + s2_dim + joint_action_dim, hidden_dim[0])
        self.l5 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l6 = nn.Linear(hidden_dim[1], 1)

    def forward(self, s1, s2, u):
        if self.t1 > 1:
            s1 = self.lstm1(s1)
        if self.t2 > 1:
            s2 = self.lstm2(s2)

        xu = torch.cat([s1, s2, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, s1, s2, u):
        if self.t1 > 1:
            s1 = self.lstm1(s1)
        if self.t2 > 1:
            s2 = self.lstm2(s2)

        xu = torch.cat([s1, s2, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


# for tg/pg
class HATD3Double:
    def __init__(self, cfg, light_opt):    # l对应time，v对应phase，懒得改变量名了，就这吧
        self.use_opc = cfg['use_opc']
        self.o1_dim = cfg[light_opt]['obs_dim']       # 单步观测维度
        self.s1_dim = cfg[light_opt]['state_dim']     # LSTM输出维度
        self.a1_dim = cfg[light_opt]['act_dim']       # 动作维度
        self.t1 = cfg[light_opt]['T']
        self.obs1_dim = self.o1_dim * self.t1
        self.choose_time_action = self.choose_phase_action = self.choose_light_action

        self.o2_dim = cfg['vehicle']['obs_dim']
        self.s2_dim = cfg['vehicle']['state_dim']  # LSTM输出维度
        self.a2_dim = cfg['vehicle']['act_dim']
        self.t2 = cfg['vehicle']['T']
        self.obs2_dim = self.o2_dim * self.t2

        self.obs_dim = self.obs1_dim + self.obs2_dim
        self.act_dim = self.a1_dim + self.a2_dim
        # self.act_dim = self.a1_dim + 1

        self.gamma = cfg['gamma']
        self.tau = cfg['tau']  # 软更新系数
        self.batch_size = cfg['batch_size']

        self.memory_capacity = cfg['memory_capacity']  # 记忆库大小
        self.learn_begin = self.memory_capacity * cfg['learn_start_ratio']   # 存满一定比例的记忆库之后开始学习并用网络输出动作
        self.memory = np.zeros((self.memory_capacity, self.obs_dim * 2 + self.act_dim + 1))
        self.pointer = 0

        # 创建对应的四个网络
        self.actor1 = Actor(self.o1_dim, self.s1_dim, self.a1_dim, self.t1, cfg['actor_hidden_dim']).to(device)
        self.actor1_target = Actor(self.o1_dim, self.s1_dim, self.a1_dim, self.t1, cfg['actor_hidden_dim']).to(device)
        self.actor1_target.load_state_dict(self.actor1.state_dict())  # 存储网络名字和对应参数
        self.actor1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=cfg['actor_learning_rate'])
        self.scheduler_actor1 = torch.optim.lr_scheduler.StepLR(self.actor1_optimizer,
                                                                step_size=cfg['actor_scheduler_step'], gamma=0.5)

        self.actor2 = Actor(self.o2_dim, self.s2_dim, self.a2_dim, self.t2, cfg['actor_hidden_dim']).to(device)
        self.actor2_target = Actor(self.o2_dim, self.s2_dim, self.a2_dim, self.t2, cfg['actor_hidden_dim']).to(device)
        self.actor2_target.load_state_dict(self.actor2.state_dict())  # 存储网络名字和对应参数
        self.actor2_optimizer = torch.optim.Adam(self.actor2.parameters(), lr=cfg['actor_learning_rate'])
        self.scheduler_actor2 = torch.optim.lr_scheduler.StepLR(self.actor2_optimizer,
                                                                step_size=cfg['actor_scheduler_step'], gamma=0.5)

        self.critic = CriticDouble(self.o1_dim, self.s1_dim, self.t1, self.o2_dim, self.s2_dim, self.t2,
                                   self.act_dim, cfg['critic_hidden_dim']).to(device)
        self.critic_target = CriticDouble(self.o1_dim, self.s1_dim, self.t1, self.o2_dim, self.s2_dim, self.t2,
                                          self.act_dim, cfg['critic_hidden_dim']).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg['critic_learning_rate'])
        self.scheduler_critic = torch.optim.lr_scheduler.StepLR(self.critic_optimizer,
                                                                step_size=cfg['critic_scheduler_step'], gamma=0.5)

        self.policy_freq = 2
        self.total_it = 0

    def store_transition(self, o1, o2, a1, a2, r, o1_, o2_):
        transition = np.hstack((o1, o2, a1, a2, r, o1_, o2_))
        index = self.pointer % self.memory_capacity
        self.memory[index, :] = transition
        self.pointer += 1

    def choose_light_action(self, o1):
        obs = torch.FloatTensor(o1).view(1, -1, self.o1_dim).to(device)
        return self.actor1(obs).cpu().data.numpy().flatten()

    def choose_goal(self, o2):
        obs = torch.FloatTensor(o2).view(1, -1, self.o2_dim).to(device)
        return self.actor2(obs).cpu().data.numpy().flatten()

    def learn(self, worker_policy=None):
        self.total_it += 1
        # mini batch sample
        indices = np.random.choice(min(self.pointer, self.memory_capacity), size=self.batch_size)   # 注意，这里是默认有放回
        batch_trans = self.memory[indices, :]

        o1 = torch.FloatTensor(batch_trans[:, :self.obs1_dim]).to(device)
        o2 = torch.FloatTensor(batch_trans[:, self.obs1_dim: self.obs_dim]).to(device)
        a1 = torch.FloatTensor(batch_trans[:, self.obs_dim: self.obs_dim + self.a1_dim]).to(device)
        a2 = torch.FloatTensor(batch_trans[:, self.obs_dim + self.a1_dim: self.obs_dim + self.act_dim]).to(device)
        reward = torch.FloatTensor(batch_trans[:, -self.obs_dim - 1: -self.obs_dim]).to(device)
        next_o1 = torch.FloatTensor(batch_trans[:, -self.obs_dim: -self.obs2_dim]).to(device)
        next_o2 = torch.FloatTensor(batch_trans[:, -self.obs2_dim:]).to(device)

        with torch.no_grad():
            noise_a1 = (torch.randn_like(a1) * 0.2).clamp(-0.5, 0.5)
            next_a1 = (self.actor1_target(next_o1) + noise_a1).clamp(-1, 1)    # we fix max_action == 1
            noise_a2 = (torch.randn_like(a2) * 0.2).clamp(-0.5, 0.5)
            next_a2 = (self.actor2_target(next_o2) + noise_a2).clamp(-1, 1)
            # next_a2 = self.actor2_target(next_o2)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_o1, next_o2, torch.cat((next_a1, next_a2), 1))
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (self.gamma * target_Q).detach()

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(o1, o2, torch.cat((a1, a2), 1))  # 联合动作

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.scheduler_critic.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            actors = [self.actor1, self.actor2]  # 添加更多的actor
            optimizers = [self.actor1_optimizer, self.actor2_optimizer]  # 添加更多的optimizer
            o_list = [o1, o2]  # 添加更多的o

            # 随机打乱更新顺序
            order = np.random.permutation(len(actors))

            for i in order:
                joint_action = torch.cat([actors[j](o_list[j]) for j in range(len(actors))], dim=1)
                actor_loss = -self.critic.Q1(o1, o2, joint_action).mean()
                optimizers[i].zero_grad()
                actor_loss.backward()
                optimizers[i].step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor1.parameters(), self.actor1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor2.parameters(), self.actor2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.scheduler_actor1.step()
            self.scheduler_actor2.step()

    def save(self, filename):
        # 保存模型
        torch.save(self.actor1.state_dict(), filename + "_actor1")
        torch.save(self.actor1_optimizer.state_dict(), filename + "_actor1_optimizer")
        torch.save(self.actor2.state_dict(), filename + "_actor2")
        torch.save(self.actor2_optimizer.state_dict(), filename + "_actor2_optimizer")
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

    def load(self, filename):
        # 加载模型
        self.actor1.load_state_dict(torch.load(filename + "_actor1"))
        self.actor1_optimizer.load_state_dict(torch.load(filename + "_actor1_optimizer"))
        self.actor1_target = copy.deepcopy(self.actor1)
        self.actor2.load_state_dict(torch.load(filename + "_actor2"))
        self.actor2_optimizer.load_state_dict(torch.load(filename + "_actor2_optimizer"))
        self.actor2_target = copy.deepcopy(self.actor2)
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)


# for tp
class HATD3(HATD3Double):
    def __init__(self, cfg):
        super().__init__(cfg, 'time')
        self.o2_dim = cfg['phase']['obs_dim']
        self.s2_dim = cfg['phase']['state_dim']  # LSTM输出维度
        self.a2_dim = cfg['phase']['act_dim']
        self.t2 = cfg['phase']['T']
        self.obs2_dim = self.o2_dim * self.t2

        self.obs_dim = self.obs1_dim + self.obs2_dim
        self.act_dim = self.a1_dim + self.a2_dim

        self.memory = np.zeros((self.memory_capacity, self.obs_dim * 2 + self.act_dim + 1))

        # 创建对应的四个网络
        self.actor1 = Actor(self.o1_dim, self.s1_dim, self.a1_dim, self.t1, cfg['actor_hidden_dim']).to(device)
        self.actor1_target = Actor(self.o1_dim, self.s1_dim, self.a1_dim, self.t1, cfg['actor_hidden_dim']).to(device)
        self.actor1_target.load_state_dict(self.actor1.state_dict())  # 存储网络名字和对应参数
        self.actor1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=cfg['actor_learning_rate'])
        self.scheduler_actor1 = torch.optim.lr_scheduler.StepLR(self.actor1_optimizer,
                                                                step_size=cfg['actor_scheduler_step'], gamma=0.5)

        self.actor2 = Actor(self.o2_dim, self.s2_dim, self.a2_dim, self.t2, cfg['actor_hidden_dim']).to(device)
        self.actor2_target = Actor(self.o2_dim, self.s2_dim, self.a2_dim, self.t2, cfg['actor_hidden_dim']).to(device)
        self.actor2_target.load_state_dict(self.actor2.state_dict())  # 存储网络名字和对应参数
        self.actor2_optimizer = torch.optim.Adam(self.actor2.parameters(), lr=cfg['actor_learning_rate'])
        self.scheduler_actor2 = torch.optim.lr_scheduler.StepLR(self.actor2_optimizer,
                                                                step_size=cfg['actor_scheduler_step'], gamma=0.5)

        self.critic = CriticDouble(self.o1_dim, self.s1_dim, self.t1, self.o2_dim, self.s2_dim, self.t2,
                                   self.act_dim, cfg['critic_hidden_dim']).to(device)
        self.critic_target = CriticDouble(self.o1_dim, self.s1_dim, self.t1, self.o2_dim, self.s2_dim, self.t2,
                                          self.act_dim, cfg['critic_hidden_dim']).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg['critic_learning_rate'])
        self.scheduler_critic = torch.optim.lr_scheduler.StepLR(self.critic_optimizer,
                                                                step_size=cfg['critic_scheduler_step'], gamma=0.5)

        self.choose_time_action = self.choose_light_action
        self.choose_phase_action = self.choose_goal


"""
    t+p+g的HATD3，有3个Actor
"""


class CriticTriple(nn.Module):  # 定义 critic 网络结构
    def __init__(self, o1_dim, s1_dim, t1, o2_dim, s2_dim, t2, o3_dim, s3_dim, t3, joint_action_dim, hidden_dim):
        super(CriticTriple, self).__init__()
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3

        if t1 > 1:
            self.lstm1 = LSTM(o1_dim, s1_dim, t1)
        else:
            s1_dim = o1_dim
        if t2 > 1:
            self.lstm2 = LSTM(o2_dim, s2_dim, t2)
        else:
            s2_dim = o2_dim
        if t3 > 1:
            self.lstm3 = LSTM(o3_dim, s3_dim, t3)
        else:
            s3_dim = o3_dim

        # Q1 architecture   计算 Q1
        self.l1 = nn.Linear(s1_dim + s2_dim + s3_dim + joint_action_dim, hidden_dim[0])
        self.l2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l3 = nn.Linear(hidden_dim[1], 1)

        # Q2 architecture   计算 Q2
        self.l4 = nn.Linear(s1_dim + s2_dim + s3_dim + joint_action_dim, hidden_dim[0])
        self.l5 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l6 = nn.Linear(hidden_dim[1], 1)

    def forward(self, s1, s2, s3, u):
        if self.t1 > 1:
            s1 = self.lstm1(s1)
        if self.t2 > 1:
            s2 = self.lstm2(s2)
        if self.t3 > 1:
            s3 = self.lstm3(s3)

        xu = torch.cat([s1, s2, s3, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, s1, s2, s3, u):
        if self.t1 > 1:
            s1 = self.lstm1(s1)
        if self.t2 > 1:
            s2 = self.lstm2(s2)
        if self.t3 > 1:
            s3 = self.lstm3(s3)

        xu = torch.cat([s1, s2, s3, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


class HATD3Triple:
    def __init__(self, cfg):    # l对应time，v对应phase，懒得改变量名了，就这吧
        self.use_opc = cfg['use_opc']
        self.o1_dim = cfg['time']['obs_dim']       # 单步观测维度
        self.s1_dim = cfg['time']['state_dim']     # LSTM输出维度
        self.a1_dim = cfg['time']['act_dim']       # 动作维度
        self.t1 = cfg['time']['T']
        self.obs1_dim = self.o1_dim * self.t1

        self.o2_dim = cfg['phase']['obs_dim']
        self.s2_dim = cfg['phase']['state_dim']  # LSTM输出维度
        self.a2_dim = cfg['phase']['act_dim']
        self.t2 = cfg['phase']['T']
        self.obs2_dim = self.o2_dim * self.t2

        self.o3_dim = cfg['vehicle']['obs_dim']
        self.s3_dim = cfg['vehicle']['state_dim']  # LSTM输出维度
        self.a3_dim = cfg['vehicle']['act_dim']
        self.t3 = cfg['vehicle']['T']
        self.obs3_dim = self.o3_dim * self.t3

        self.obs_dim = self.obs1_dim + self.obs2_dim + self.obs3_dim
        self.act_dim = self.a1_dim + self.a2_dim + self.a3_dim

        self.gamma = cfg['gamma']
        self.tau = cfg['tau']  # 软更新系数
        self.batch_size = cfg['batch_size']

        self.memory_capacity = cfg['memory_capacity']  # 记忆库大小
        self.learn_begin = self.memory_capacity * cfg['learn_start_ratio']   # 存满一定比例的记忆库之后开始学习并用网络输出动作
        self.memory = np.zeros((self.memory_capacity, self.obs_dim * 2 + self.act_dim + 1))
        self.pointer = 0

        # 创建对应的四个网络
        self.actor1 = Actor(self.o1_dim, self.s1_dim, self.a1_dim, self.t1, cfg['actor_hidden_dim']).to(device)
        self.actor1_target = Actor(self.o1_dim, self.s1_dim, self.a1_dim, self.t1, cfg['actor_hidden_dim']).to(device)
        self.actor1_target.load_state_dict(self.actor1.state_dict())  # 存储网络名字和对应参数
        self.actor1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=cfg['actor_learning_rate'])
        self.scheduler_actor1 = torch.optim.lr_scheduler.StepLR(self.actor1_optimizer,
                                                                step_size=cfg['actor_scheduler_step'], gamma=0.5)

        self.actor2 = Actor(self.o2_dim, self.s2_dim, self.a2_dim, self.t2, cfg['actor_hidden_dim']).to(device)
        self.actor2_target = Actor(self.o2_dim, self.s2_dim, self.a2_dim, self.t2, cfg['actor_hidden_dim']).to(device)
        self.actor2_target.load_state_dict(self.actor2.state_dict())  # 存储网络名字和对应参数
        self.actor2_optimizer = torch.optim.Adam(self.actor2.parameters(), lr=cfg['actor_learning_rate'])
        self.scheduler_actor2 = torch.optim.lr_scheduler.StepLR(self.actor2_optimizer,
                                                                step_size=cfg['actor_scheduler_step'], gamma=0.5)

        self.actor3 = Actor(self.o3_dim, self.s3_dim, self.a3_dim, self.t3, cfg['actor_hidden_dim']).to(device)
        self.actor3_target = Actor(self.o3_dim, self.s3_dim, self.a3_dim, self.t3, cfg['actor_hidden_dim']).to(device)
        self.actor3_target.load_state_dict(self.actor3.state_dict())  # 存储网络名字和对应参数
        self.actor3_optimizer = torch.optim.Adam(self.actor3.parameters(), lr=cfg['actor_learning_rate'])
        self.scheduler_actor3 = torch.optim.lr_scheduler.StepLR(self.actor3_optimizer,
                                                                step_size=cfg['actor_scheduler_step'], gamma=0.5)

        self.critic = CriticTriple(self.o1_dim, self.s1_dim, self.t1,
                                   self.o2_dim, self.s2_dim, self.t2,
                                   self.o3_dim, self.s3_dim, self.t3,
                                   self.act_dim, cfg['critic_hidden_dim']).to(device)
        self.critic_target = CriticTriple(self.o1_dim, self.s1_dim, self.t1,
                                          self.o2_dim, self.s2_dim, self.t2,
                                          self.o3_dim, self.s3_dim, self.t3,
                                          self.act_dim, cfg['critic_hidden_dim']).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg['critic_learning_rate'])
        self.scheduler_critic = torch.optim.lr_scheduler.StepLR(self.critic_optimizer,
                                                                step_size=cfg['critic_scheduler_step'], gamma=0.5)

        self.policy_freq = 2
        self.total_it = 0

    def store_transition(self, o1, o2, o3, a1, a2, a3, r, o1_, o2_, o3_):
        transition = np.hstack((o1, o2, o3, a1, a2, a3, r, o1_, o2_, o3_))
        index = self.pointer % self.memory_capacity
        self.memory[index, :] = transition
        self.pointer += 1

    def choose_time_action(self, o1):
        obs = torch.FloatTensor(o1).view(1, -1, self.o1_dim).to(device)
        return self.actor1(obs).cpu().data.numpy().flatten()

    def choose_phase_action(self, o2):
        obs = torch.FloatTensor(o2).view(1, -1, self.o2_dim).to(device)
        return self.actor2(obs).cpu().data.numpy().flatten()

    def choose_goal(self, o3):
        obs = torch.FloatTensor(o3).view(1, -1, self.o3_dim).to(device)
        return self.actor3(obs).cpu().data.numpy().flatten()

    def learn(self, worker_policy):
        self.total_it += 1
        # mini batch sample
        indices = np.random.choice(min(self.pointer, self.memory_capacity), size=self.batch_size)   # 注意，这里是默认有放回
        batch_trans = self.memory[indices, :]

        o1 = torch.FloatTensor(batch_trans[:, :self.obs1_dim]).to(device)
        o2 = torch.FloatTensor(batch_trans[:, self.obs1_dim: self.obs1_dim + self.obs2_dim]).to(device)
        o3 = torch.FloatTensor(batch_trans[:, self.obs1_dim + self.obs2_dim: self.obs_dim]).to(device)
        a1 = torch.FloatTensor(batch_trans[:, self.obs_dim: self.obs_dim + self.a1_dim]).to(device)
        a2 = torch.FloatTensor(batch_trans[:, self.obs_dim + self.a1_dim: self.obs_dim + self.a1_dim + self.a2_dim]).to(device)
        a3 = torch.FloatTensor(batch_trans[:, self.obs_dim + self.a1_dim + self.a2_dim: self.obs_dim + self.act_dim]).to(device)
        reward = torch.FloatTensor(batch_trans[:, -self.obs_dim - 1: -self.obs_dim]).to(device)
        next_o1 = torch.FloatTensor(batch_trans[:, -self.obs_dim: -(self.obs2_dim + self.obs3_dim)]).to(device)
        next_o2 = torch.FloatTensor(batch_trans[:, -(self.obs2_dim + self.obs3_dim): -self.obs3_dim]).to(device)
        next_o3 = torch.FloatTensor(batch_trans[:, -self.obs3_dim:]).to(device)

        with torch.no_grad():
            noise_a1 = (torch.randn_like(a1) * 0.2).clamp(-0.5, 0.5)
            next_a1 = (self.actor1_target(next_o1) + noise_a1).clamp(-1, 1)    # we fix max_action == 1
            noise_a2 = (torch.randn_like(a2) * 0.2).clamp(-0.5, 0.5)
            next_a2 = (self.actor2_target(next_o2) + noise_a2).clamp(-1, 1)
            noise_a3 = (torch.randn_like(a3) * 0.2).clamp(-0.5, 0.5)
            next_a3 = (self.actor3_target(next_o3) + noise_a3).clamp(-1, 1)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_o1, next_o2, next_o3, torch.cat((next_a1, next_a2, next_a3), 1))
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (self.gamma * target_Q).detach()

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(o1, o2, o3, torch.cat((a1, a2, a3), 1))  # 联合动作

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.scheduler_critic.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            actors = [self.actor1, self.actor2, self.actor3]  # 添加更多的actor
            optimizers = [self.actor1_optimizer, self.actor2_optimizer, self.actor3_optimizer]  # 添加更多的optimizer
            o_list = [o1, o2, o3]  # 添加更多的o

            # 随机打乱更新顺序
            order = np.random.permutation(len(actors))

            for i in order:
                joint_action = torch.cat([actors[j](o_list[j]) for j in range(len(actors))], dim=1)
                actor_loss = -self.critic.Q1(o1, o2, o3, joint_action).mean()
                optimizers[i].zero_grad()
                actor_loss.backward()
                optimizers[i].step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor1.parameters(), self.actor1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor2.parameters(), self.actor2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor3.parameters(), self.actor3_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.scheduler_actor1.step()
            self.scheduler_actor2.step()
            self.scheduler_actor3.step()

    def save(self, filename):
        # 保存模型
        torch.save(self.actor1.state_dict(), filename + "_actor1")
        torch.save(self.actor1_optimizer.state_dict(), filename + "_actor1_optimizer")
        torch.save(self.actor2.state_dict(), filename + "_actor2")
        torch.save(self.actor2_optimizer.state_dict(), filename + "_actor2_optimizer")
        torch.save(self.actor3.state_dict(), filename + "_actor3")
        torch.save(self.actor3_optimizer.state_dict(), filename + "_actor3_optimizer")
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

    def load(self, filename):
        # 加载模型
        self.actor1.load_state_dict(torch.load(filename + "_actor1"))
        self.actor1_optimizer.load_state_dict(torch.load(filename + "_actor1_optimizer"))
        self.actor1_target = copy.deepcopy(self.actor1)
        self.actor2.load_state_dict(torch.load(filename + "_actor2"))
        self.actor2_optimizer.load_state_dict(torch.load(filename + "_actor2_optimizer"))
        self.actor2_target = copy.deepcopy(self.actor2)
        self.actor3.load_state_dict(torch.load(filename + "_actor3"))
        self.actor3_optimizer.load_state_dict(torch.load(filename + "_actor3_optimizer"))
        self.actor3_target = copy.deepcopy(self.actor3)
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)


class ManagerCritic(nn.Module):  # 定义 critic 网络结构
    def __init__(self, o_dim, s_dim, a_dim, t, hidden_dim):
        super(ManagerCritic, self).__init__()
        self.t = t

        if t > 1:
            self.lstm = LSTM(o_dim, s_dim, t)
        else:
            s_dim = o_dim

        # Q1 architecture   计算 Q1
        self.l1 = nn.Linear(s_dim + a_dim, hidden_dim[0])
        self.l2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l3 = nn.Linear(hidden_dim[1], 1)

        # Q2 architecture   计算 Q2
        self.l4 = nn.Linear(s_dim + a_dim, hidden_dim[0])
        self.l5 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l6 = nn.Linear(hidden_dim[1], 1)

    def forward(self, state, action):  # 注意此处，直接把两个网络写在一起，这样就可以只用一个梯度下降优化器
        if self.t > 1:
            state = self.lstm(state)

        sa = torch.cat([state, action], 1)  # 将s和a横着拼接在一起

        x1 = F.relu(self.l1(sa))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)    # 直接输出线性计算后的值作为Q值

        x2 = F.relu(self.l4(sa))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, state, action):  # 新增一个Q值输出的方法，只使用其中一个网络的结果作为输出，避免重复计算
        if self.t > 1:
            state = self.lstm(state)

        sa = torch.cat([state, action], 1)

        x1 = F.relu(self.l1(sa))
        x1 = F.relu(self.l2(x1))
        q = self.l3(x1)
        return q


# only for G
class ManagerTD3:
    class ReplayBuffer:
        def __init__(self, capacity, obs_dim, act_dim, seq_len, useOPC):
            self.capacity = capacity
            worker_obs_dim, worker_act_dim = 8, 1   # 车辆obs和act维度
            # self.memory = np.zeros((self.memory_capacity, self.obs_dim * 2 + self.a_dim + 1))
            self.buffer = {
                'obs': np.zeros((capacity, obs_dim), dtype=np.float32),
                'next_obs': np.zeros((capacity, obs_dim), dtype=np.float32),
                'action': np.zeros((capacity, act_dim), dtype=np.float32),  # goal_dim
                'reward': np.zeros((capacity, 1), dtype=np.float32),
                # 'worker_obs': [],
                # 'worker_act': [],
                'worker_obs': np.zeros((capacity, seq_len, 8, worker_obs_dim), dtype=np.float32),
                'worker_act': np.zeros((capacity, seq_len, 8, worker_act_dim), dtype=np.float32),
            }
            self.pointer = 0
            self.useOPC = useOPC

        def store_transition(self, *args):
            o, a, r, o_, wo, wa = args
            index = self.pointer % self.capacity
            self.buffer['obs'][index] = o
            self.buffer['action'][index] = a
            self.buffer['reward'][index] = r
            self.buffer['next_obs'][index] = o_
            # self.buffer['worker_obs'].append(wo)
            # self.buffer['worker_act'].append(wa)
            if self.useOPC:
                wo, wa = np.array(wo), np.array(wa)
                self.buffer['worker_obs'][index] = wo
                self.buffer['worker_act'][index] = wa

            self.pointer += 1

        def __getitem__(self, item):
            return self.buffer[item]

    def __init__(self, cfg):   # opt = 'time' or 'phase'
        self.use_opc = cfg['use_opc']
        self.o_dim = cfg['vehicle']['obs_dim']    # 单步观测维度
        self.a_dim = cfg['vehicle']['act_dim']    # 动作维度
        self.s_dim = cfg['vehicle']['state_dim']  # LSTM输出维度
        self.t = cfg['vehicle']['T']
        hidden_dim = cfg['vehicle']['hidden_dim']

        self.obs_dim = self.o_dim * self.t
        self.choose_goal = self.choose_action      # 方便外部调用

        self.gamma = cfg['gamma']
        self.tau = cfg['tau']  # 软更新系数
        self.batch_size = cfg['batch_size']

        self.memory_capacity = cfg['memory_capacity']  # 记忆库大小
        self.learn_begin = self.memory_capacity * cfg['learn_start_ratio']   # 存满一定比例的记忆库之后开始学习并用网络输出动作
        # self.memory = np.zeros((self.memory_capacity, self.obs_dim * 2 + self.a_dim + 1))
        # self.pointer = 0
        self.memory = self.ReplayBuffer(self.memory_capacity, self.obs_dim, self.a_dim, 25, self.use_opc)
        self.store_transition = self.memory.store_transition

        # 创建对应的四个网络
        self.actor = Actor(self.o_dim, self.s_dim, self.a_dim, self.t, hidden_dim).to(device)
        self.actor_target = Actor(self.o_dim, self.s_dim, self.a_dim, self.t, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())  # 存储网络名字和对应参数
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg['actor_learning_rate'])
        self.scheduler_actor = torch.optim.lr_scheduler.StepLR(self.actor_optimizer,
                                                               step_size=cfg['actor_scheduler_step'], gamma=0.5)

        self.critic = ManagerCritic(self.o_dim, self.s_dim, self.a_dim, self.t, hidden_dim).to(device)
        self.critic_target = ManagerCritic(self.o_dim, self.s_dim, self.a_dim, self.t, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg['critic_learning_rate'])
        self.scheduler_critic = torch.optim.lr_scheduler.StepLR(self.critic_optimizer,
                                                                step_size=cfg['critic_scheduler_step'], gamma=0.5)

        self.policy_freq = 2
        self.total_it = 0

    @property
    def pointer(self):
        return self.memory.pointer

    # def store_transition(self, o, a, r, o_):
    #     transition = np.hstack((o, a, r, o_))
    #     index = self.pointer % self.memory_capacity
    #     self.memory[index, :] = transition
    #     self.pointer += 1

    def choose_action(self, o_l):
        obs = torch.FloatTensor(o_l).view(1, -1, self.o_dim).to(device)
        return self.actor(obs).cpu().data.numpy().flatten()

    @staticmethod
    def off_policy_corrections(worker_policy, sgoals, wstates, wactions, candidate_goals=8):
        """
            HIRO论文中的off policy correction实现
        :param worker_policy: 下层policy，输入o_low, 输出a_low
        :param sgoals: 上层动作，即给下层的目标 8
        :param states: 下层在sgoal作用期间的观测，即下层policy每一步的输入
        :param actions: 下层在sgoal作用期间的动作，即下层policy每一步的输出
        :param candidate_goals: 除s_final-s_origin和sgoal之外的候选目标数
        :return: 修正后的sgoal
        """
        # states需要是(batch, seq_len, obs_dim)形状,实际是(batch, seq_len, 8, wo=8)
        # actions形状为(batch, seq_len, act_dim),注意是每一步的下层action (batch, seq_len, 8, wa=1)
        batch_size = len(wstates)
        obs_dim = wstates[0][0][0].shape  # worker o_dim=8
        action_dim = wactions[0][0][0].shape  # worker a_dim=1

        state = np.transpose(wstates, (2, 0, 1, 3))  # (lane=8, batch, seq_len, wo=8)
        action = np.array(wactions)  # (batch, seq_len, 8, wa=1)
        action = np.transpose(action, (2, 0, 1, 3))  # (lane=8, batch, seq_len, wa=1)
        real_subgoal = np.array(sgoals)
        real_subgoal = np.transpose(real_subgoal, (1, 0))
        res = np.zeros((8, batch_size, 1))
        for lane in range(8):
            states = state[lane]
            # 提取状态序列的第一个和最后一个状态
            first_s = [s[0] for s in states]  # 第一个状态(batch, wo)
            last_s = [s[-1] for s in states]  # 最后一个状态(batch, wo)

            gdim = 1  # 为了区分原本的opc，我这里gdim是说给下层每个智能体其实只用了维度为1的goal，是因为每个车道用一个目标导致了self.a_dim=8
            # # # 注意，因为我把平均车速放在cav_obs的最后一维，也就相当于最后一维是目标对应的状态空间，所以是-1:
            # 计算目标差值，形状为(batch_size, 1, subgoal_dim)
            diff_goal = (np.array(last_s) - np.array(first_s))[:, np.newaxis, -gdim:]

            # diff_goal = (np.array(last_s) - np.array(first_s))[..., -gdim:]  # 获取每个obs_dim维度的最后一个元素(batch, lane=8, 1)
            # diff_goal = np.transpose(diff_goal, (0, 2, 1))  # 将结果转变为(batch_size,1,lane=8)维度

            # 生成原始目标和随机目标，形状为(batch_size, 1, subgoal_dim)
            # original_goal = np.array(sgoals)[:, np.newaxis, :]  # 也就是原本manager的动作 (batch, 1, self.a_dim=lane=8)
            lane_real_subgoal = real_subgoal[lane]
            original_goal = lane_real_subgoal[:, np.newaxis, np.newaxis]  # ! 也就是原本manager的动作 (batch, 1, 1)
            random_goals = np.random.normal(loc=diff_goal, scale=.5,
                                            size=(batch_size, candidate_goals, original_goal.shape[-1]))
            random_goals = random_goals.clip(-1, 1)

            # 合并原始目标、目标差值和随机目标，形状为(batch_size, 10, subgoal_dim)
            candidates = np.concatenate([original_goal, diff_goal, random_goals], axis=1)  # (batch, 10, lane=8)
            ncands = candidates.shape[1]  # =1+1+8=10

            # 将动作序列转换为一维数组
            # actions = np.array(actions)     # (batch, seq_len, 8, wa=1)
            actions = action[lane]
            seq_len = len(states[0])
            new_batch_sz = seq_len * batch_size  # !
            # new_batch_sz = seq_len * batch_size * 8     # ! 注意这里为了后面能输入worker_policy，要把每个车道的目标分开

            observations = states.reshape((new_batch_sz,) + obs_dim)  # (b*seq*lane, wo)
            true_actions = actions.reshape((new_batch_sz,) + action_dim)  # (b*seq*lane, wa)
            # goal_shape = (new_batch_sz, self.a_dim)   # !
            goal_shape = (new_batch_sz, gdim)  # !这里也是改成了(batch*seq_len*8, 1)

            # 计算候选目标下的动作
            policy_actions = np.zeros((ncands, new_batch_sz) + action_dim)  # (10, b*seq*lane, wa=1)
            for c in range(ncands):
                subgoal = candidates[:, c]  # (batch, lane=8)
                # (batch_size, 1, self.a_dim) - (batch_size, seq_len, self.a_dim) = (batch_size, seq_len, self.a_dim)   #
                candidate = (subgoal + states[:, 0, -gdim:])[:, None] - states[:, :, -gdim:]  #
                # state_start = states[:, 0, :, -1:]
                # assert state_start.shape == (batch_size, 8, 1)
                # state_start = state_start.reshape((batch_size, 8))
                # state_final = states[:, :, :, -1:]  # (batch_size, seq_len, 8, 1)
                # assert state_final.shape == (batch_size, seq_len, 8, 1)
                # state_final = state_final.reshape((batch_size, seq_len, 8))
                # candidate = (subgoal + state_start)[:, None] - state_final  # (batch_size, seq_len, 8)
                candidate = candidate.reshape(*goal_shape)  # 合并前两维度
                policy_actions[c] = worker_policy(observations, candidate)  # (b*seq*lane, wa=1)
                # 新增逻辑：如果没有真正控制到车，则不用网络输出动作
                not_real_ctrl = np.where(observations[:, 0] == -1)
                if np.any(not_real_ctrl):
                    policy_actions[c][not_real_ctrl] = -1

            # 计算真实动作与候选目标动作的差异
            difference = (policy_actions - true_actions)  # (10, b*seq*lane, wa=1) - (b*seq*lane, wa=1) = (10, ..., 1)
            difference = np.where(difference != -np.inf, difference, 0)
            difference = difference.reshape((ncands, batch_size, seq_len) + action_dim).transpose(1, 0, 2, 3)  # !

            # difference = difference.reshape((ncands, batch_size, seq_len, 8)).transpose(1, 0, 2, 3)

            # 计算差异的对数概率
            logprob = -0.5 * np.sum(np.linalg.norm(difference, axis=-1) ** 2,
                                    axis=-1)  # 求范数后(b,ncand, seq), 求和后(b,ncand)
            # 选择对数概率最大的候选目标
            max_indices = np.argmax(logprob, axis=-1)  # (batch_size)

            res[lane] = candidates[np.arange(batch_size), max_indices]

        # 返回最优候选目标
        # return candidates[np.arange(batch_size), max_indices]   # (batch_size, self.a_dim=lane=8)
        res = np.transpose(res, (1, 0, 2))  # (batch_size, lane=8, 1)
        res = res.reshape(batch_size, 8)
        return res

    def learn(self, worker_policy):
        self.total_it += 1
        # mini batch sample

        indices = np.random.choice(min(self.pointer, self.memory_capacity), size=self.batch_size)   # 注意，这里是默认有放回

        # batch_trans = self.memory[indices, :]
        # obs = torch.FloatTensor(batch_trans[:, :self.obs_dim]).to(device)
        # action = torch.FloatTensor(batch_trans[:, self.obs_dim: self.obs_dim + self.a_dim]).to(device)
        # reward = torch.FloatTensor(batch_trans[:, -self.obs_dim - 1: -self.obs_dim]).to(device)
        # next_obs = torch.FloatTensor(batch_trans[:, -self.obs_dim:]).to(device)
        obs = torch.FloatTensor(self.memory['obs'][indices]).to(device)
        next_obs = torch.FloatTensor(self.memory['next_obs'][indices]).to(device)
        reward = torch.FloatTensor(self.memory['reward'][indices]).to(device)

        ori_action = self.memory['action'][indices]
        if not self.use_opc:
            action = torch.FloatTensor(ori_action).to(device)
        else:
            worker_obs = self.memory['worker_obs'][indices]
            worker_act = self.memory['worker_act'][indices]
            corrected_action = self.off_policy_corrections(worker_policy, ori_action, worker_obs, worker_act)
            # print('\n,ori:', ori_action, '\nnew:', corrected_action)
            action = torch.FloatTensor(corrected_action).to(device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * 0.2).clamp(-0.5, 0.5)       # noise=0.2, clip=0.5
            next_action = (self.actor_target(next_obs) + noise).clamp(-1, 1)    # 默认动作空间[-1,1]

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (self.gamma * target_Q).detach()

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.scheduler_critic.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(obs, self.actor(obs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.scheduler_actor.step()

    def save(self, filename):
        # 保存模型
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

    def load(self, filename):
        # 加载模型
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)


class WorkerActor(nn.Module):  # 定义 actor 网络结构
    def __init__(self, obs_dim, state_dim, action_dim, goal_dim, t, hidden_dim, max_action=1):
        super(WorkerActor, self).__init__()
        self.T = t

        if t > 1:
            self.lstm = LSTM(obs_dim, state_dim, t)
        else:
            state_dim = obs_dim

        self.l1 = nn.Linear(state_dim + goal_dim, hidden_dim[0])
        self.l2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l3 = nn.Linear(hidden_dim[1], action_dim)
        self.max_action = max_action

    def forward(self, s, g):
        if self.T > 1:
            s = self.lstm(s)
        else:
            s = s.squeeze(1)
        x = torch.cat([s, g], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class WorkerCritic(nn.Module):  # 定义 critic 网络结构
    def __init__(self, o_dim, s_dim, a_dim, g_dim, t, hidden_dim):
        super(WorkerCritic, self).__init__()
        self.t = t

        if t > 1:
            self.lstm = LSTM(o_dim, s_dim, t)
        else:
            s_dim = o_dim

        # Q1 architecture   计算 Q1
        self.l1 = nn.Linear(s_dim + g_dim + a_dim, hidden_dim[0])
        self.l2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l3 = nn.Linear(hidden_dim[1], 1)

        # Q2 architecture   计算 Q2
        self.l4 = nn.Linear(s_dim + g_dim + a_dim, hidden_dim[0])
        self.l5 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l6 = nn.Linear(hidden_dim[1], 1)

    def forward(self, state, goal, action):  # 注意此处，直接把两个网络写在一起，这样就可以只用一个梯度下降优化器
        if self.t > 1:
            state = self.lstm(state)

        sa = torch.cat([state, goal, action], 1)  # 将s和a横着拼接在一起

        x1 = F.relu(self.l1(sa))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)  # 直接输出线性计算后的值作为Q值

        x2 = F.relu(self.l4(sa))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, state, goal, action):  # 新增一个Q值输出的方法，只使用其中一个网络的结果作为输出，避免重复计算
        if self.t > 1:
            state = self.lstm(state)

        sa = torch.cat([state, goal, action], 1)

        x1 = F.relu(self.l1(sa))
        x1 = F.relu(self.l2(x1))
        q = self.l3(x1)
        return q


class WorkerTD3:
    def __init__(self, cfg):
        self.o_dim = cfg['cav']['obs_dim']  # 单步观测维度
        self.a_dim = cfg['cav']['act_dim']  # 动作维度
        self.s_dim = cfg['cav']['state_dim']  # LSTM输出维度
        self.t = cfg['cav']['T']
        self.g_dim = cfg['goal_dim']  # 上层来的目标维度
        self.obs_dim = self.o_dim * self.t

        self.gamma = cfg['gamma']
        self.tau = cfg['tau']  # 软更新系数
        self.batch_size = cfg['batch_size']

        self.memory_capacity = cfg['memory_capacity']  # 记忆库大小
        self.learn_begin = self.memory_capacity * cfg['learn_start_ratio']  # 存满一定比例的记忆库之后开始学习并用网络输出动作
        self.memory = np.zeros((self.memory_capacity, (self.obs_dim + self.g_dim) * 2 + self.a_dim + 1))
        self.pointer = 0

        # 创建对应的四个网络
        self.actor = WorkerActor(self.o_dim, self.s_dim, self.a_dim, self.g_dim, self.t, cfg['hidden_dim']).to(device)
        self.actor_target = WorkerActor(self.o_dim, self.s_dim, self.a_dim, self.g_dim, self.t, cfg['hidden_dim']).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())  # 存储网络名字和对应参数
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg['actor_learning_rate'])
        self.scheduler_actor = torch.optim.lr_scheduler.StepLR(self.actor_optimizer,
                                                               step_size=cfg['actor_scheduler_step'], gamma=0.5)

        # self.critic = CriticSingle(self.o_dim, self.s_dim, self.a_dim, self.t, cfg['hidden_dim']).to(device)
        # self.critic_target = CriticSingle(self.o_dim, self.s_dim, self.a_dim, self.t, cfg['hidden_dim']).to(device)
        self.critic = WorkerCritic(self.o_dim, self.s_dim, self.a_dim, self.g_dim, self.t, cfg['hidden_dim']).to(device)
        self.critic_target = WorkerCritic(self.o_dim, self.s_dim, self.a_dim, self.g_dim, self.t, cfg['hidden_dim']).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg['critic_learning_rate'])
        self.scheduler_critic = torch.optim.lr_scheduler.StepLR(self.critic_optimizer,
                                                                step_size=cfg['critic_scheduler_step'], gamma=0.5)

        self.policy_freq = 2
        self.total_it = 0

    def store_transition(self, o, a, g, g_, r, o_):
        transition = np.hstack((o, a, g, g_, r, o_))
        index = self.pointer % self.memory_capacity
        self.memory[index, :] = transition
        self.pointer += 1

    def choose_action(self, o, g):
        obs = torch.FloatTensor(o).view(1, -1, self.o_dim).to(device)
        goal = torch.FloatTensor(g).view(1, self.g_dim).to(device)
        return self.actor(obs, goal).cpu().data.numpy().flatten()

    def policy(self, o, g):
        """for manager"""
        obs = torch.FloatTensor(o).to(device)
        goal = torch.FloatTensor(g).to(device)
        return self.actor(obs, goal).cpu().data.numpy()

    def learn(self):
        self.total_it += 1
        # mini batch sample
        indices = np.random.choice(min(self.pointer, self.memory_capacity), size=self.batch_size)  # 注意，这里是默认有放回
        batch_trans = self.memory[indices, :]

        obs = torch.FloatTensor(batch_trans[:, :self.obs_dim]).to(device)
        action = torch.FloatTensor(batch_trans[:, self.obs_dim: self.obs_dim + self.a_dim]).to(device)
        goal = torch.FloatTensor(batch_trans[:, self.obs_dim + self.a_dim: self.obs_dim + self.a_dim + self.g_dim]).to(device)
        next_goal = torch.FloatTensor(batch_trans[:, self.obs_dim + self.a_dim + self.g_dim: self.obs_dim + self.a_dim + 2 * self.g_dim]).to(device)
        reward = torch.FloatTensor(batch_trans[:, -self.obs_dim - 1: -self.obs_dim]).to(device)
        next_obs = torch.FloatTensor(batch_trans[:, -self.obs_dim:]).to(device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * 0.2).clamp(-0.5, 0.5)  # noise=0.2, clip=0.5
            next_action = (self.actor_target(next_obs, next_goal) + noise).clamp(-1, 1)  # 默认动作空间[-1,1]

            # Compute the target Q value
            # target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_goal, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (self.gamma * target_Q).detach()

        # Get current Q estimates
        # current_Q1, current_Q2 = self.critic(obs, action)
        current_Q1, current_Q2 = self.critic(obs, goal, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.scheduler_critic.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # actor_loss = -self.critic.Q1(obs, self.actor(obs, goal)).mean()
            actor_loss = -self.critic.Q1(obs, goal, self.actor(obs, goal)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.scheduler_actor.step()

    def save(self, filename):
        # 保存模型
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

    def load(self, filename):
        # 加载模型
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

