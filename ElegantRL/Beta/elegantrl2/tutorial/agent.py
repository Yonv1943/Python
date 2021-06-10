import torch
import numpy as np
import numpy.random as rd
from copy import deepcopy
from elegantrl2.tutorial.net import QNet, QNetTwin
from elegantrl2.tutorial.net import Actor, ActorSAC, ActorPPO, ActorDiscretePPO
from elegantrl2.tutorial.net import Critic, CriticAdv, CriticTwin


class AgentBase:
    def __init__(self):
        self.state = None
        self.device = None
        self.action_dim = None
        self.if_on_policy = False
        self.criterion = torch.nn.SmoothL1Loss()
        self.cri = self.cri_optim = self.Cri = None  # self.Cri is the class of cri
        self.act = self.act_optim = self.Act = None  # self.Act is the class of cri
        self.cri_target = self.if_use_cri_target = None
        self.act_target = self.if_use_act_target = None

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4):  # explict call self.init() for multiprocessing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim

        self.cri = self.Cri(net_dim, state_dim, action_dim).to(self.device)
        self.act = self.Act(net_dim, state_dim, action_dim).to(self.device) if self.Act is not None else self.cri
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
        self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate) if self.Act is not None else self.cri
        del self.Cri, self.Act, self.if_use_cri_target, self.if_use_act_target

    def select_action(self, state) -> np.ndarray:
        pass  # sample form an action distribution

    def explore_env(self, env, target_step, reward_scale, gamma) -> list:
        trajectory_list = list()

        state = self.state
        for _ in range(target_step):
            action = self.select_action(state)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, *action)
            trajectory_list.append((state, other))

            state = env.reset() if done else next_s
        self.state = state
        return trajectory_list

    @staticmethod
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1 - tau))


class AgentDQN(AgentBase):
    def __init__(self):
        super().__init__()
        self.explore_rate = 0.25  # the probability of choosing action randomly in epsilon-greedy
        self.if_use_cri_target = True
        self.Cri = QNet

    def select_action(self, state) -> int:  # for discrete action space
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_int = rd.randint(self.action_dim)  # choosing action randomly
        else:
            states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
            action = self.act(states)[0]
            a_int = action.argmax(dim=0).detach().cpu().numpy()
        return a_int

    def explore_env(self, env, target_step, reward_scale, gamma) -> list:
        trajectory_list = list()

        state = self.state
        for _ in range(target_step):
            action = self.select_action(state)  # assert isinstance(action, int)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, action)
            trajectory_list.append((state, other))

            state = env.reset() if done else next_s
        self.state = state
        return trajectory_list

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        buffer.update_now_len()

        obj_critic = q_value = None
        for _ in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, q_value = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau)
        return obj_critic.item(), q_value.mean().item()

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q_value = self.cri(state).gather(1, action.long())
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, q_value


class AgentDoubleDQN(AgentDQN):
    def __init__(self):
        super().__init__()
        self.softMax = torch.nn.Softmax(dim=1)
        self.Cri = QNetTwin

    def select_action(self, state) -> int:  # for discrete action space
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        actions = self.act(states)
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_prob = self.softMax(actions)[0].detach().cpu().numpy()
            a_int = rd.choice(self.action_dim, p=a_prob)  # choose action according to Q value
        else:
            action = actions[0]
            a_int = action.argmax(dim=0).detach().cpu().numpy()
        return a_int

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s)).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q1, q2 = [qs.gather(1, action.long()) for qs in self.act.get_q1_q2(state)]
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        return obj_critic, q1


class AgentDDPG(AgentBase):
    def __init__(self):
        super().__init__()
        self.explore_noise = 0.1  # explore noise of action
        self.if_use_cri_target = self.if_use_act_target = True
        self.Act = Actor
        self.Cri = Critic

    def select_action(self, state) -> np.ndarray:
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        action = self.act(states)[0]
        action = (action + torch.randn_like(action) * self.explore_noise).clamp(-1, 1)
        return action.cpu().numpy()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> (float, float):
        buffer.update_now_len()

        obj_critic = obj_actor = None
        for _ in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau)

            action_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri(state, action_pg).mean()
            self.optim_update(self.act_optim, obj_actor)
            self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_actor.item(), obj_critic.item()

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s, self.act_target(next_s))
            q_label = reward + mask * next_q
        q_value = self.cri(state, action)
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, state


class AgentTD3(AgentDDPG):
    def __init__(self):
        super().__init__()
        self.policy_noise = 0.2  # standard deviation of policy noise
        self.update_freq = 2  # delay update frequency
        self.Cri = CriticTwin

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        buffer.update_now_len()

        obj_critic = obj_actor = None
        for update_c in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic)

            action_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri_target(state, action_pg).mean()  # use cri_target instead of cri for stable training
            self.optim_update(self.act_optim, obj_actor)
            if update_c % self.update_freq == 0:  # delay update
                self.soft_update(self.cri_target, self.cri, soft_update_tau)
                self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_critic.item() / 2, obj_actor.item()

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_a = self.act_target.get_action(next_s, self.policy_noise)  # policy noise
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
            q_label = reward + mask * next_q

        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)  # twin critics
        return obj_critic, state


class AgentSAC(AgentBase):
    def __init__(self):
        super().__init__()
        self.if_use_cri_target = True
        self.Act = ActorSAC
        self.Cri = CriticTwin

    def select_action(self, state) -> np.ndarray:
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        action = self.act.get_action(states)[0]
        return action.cpu().numpy()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        buffer.update_now_len()

        log_alpha = self.act.log_alpha
        obj_critic = obj_actor = None
        for update_c in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, state = self.get_obj_critic(buffer, batch_size, log_alpha.exp())
            self.optim_update(self.cri_optim, obj_critic)

            action_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            obj_actor = (-torch.min(*self.cri_target.get_q1_q2(state, action_pg)).mean()
                         + logprob.mean() * log_alpha.exp().detach()
                         + self.act.get_obj_alpha(logprob))
            self.optim_update(self.act_optim, obj_actor)
            self.soft_update(self.cri_target, self.cri, soft_update_tau)
        return obj_critic.item() / 2, obj_actor.item(), log_alpha.item()

    def get_obj_critic(self, buffer, batch_size, alpha) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_a, next_logprob = self.act.get_action_logprob(next_s)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))
            q_label = reward + mask * (next_q + next_logprob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)  # twin critics
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        return obj_critic, state


class AgentPPO(AgentBase):
    def __init__(self):
        super().__init__()
        self.if_on_policy = True
        self.ratio_clip = 0.2  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02  # could be 0.02
        self.Act = ActorPPO
        self.Cri = CriticAdv

    def select_action(self, state):
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        actions, noises = self.act.get_action(states)  # plan to be get_action_a_noise
        return actions[0].detach().cpu().numpy(), noises[0].detach().cpu().numpy()

    def explore_env(self, env, target_step, reward_scale, gamma):
        trajectory_list = list()

        state = self.state
        for _ in range(target_step):
            action, noise = self.select_action(state)
            next_s, reward, done, _ = env.step(np.tanh(action))
            other = (reward * reward_scale, 0.0 if done else gamma, *action, *noise)
            trajectory_list.append((state, other))

            state = env.reset() if done else next_s
        self.state = state
        return trajectory_list

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        buffer.update_now_len()
        buf_len, buf_state, buf_action, buf_r_sum, buf_logprob, buf_advantage = self.prepare_buffer(buffer)
        buffer.empty_buffer()

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = obj_actor = old_logprob = None
        for _ in range(int(buf_len / batch_size * repeat_times)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            advantage = buf_advantage[indices]
            old_logprob = buf_logprob[indices]

            new_logprob, obj_entropy = self.act.get_new_logprob_entropy(state, action)
            ratio = (new_logprob - old_logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            self.optim_update(self.act_optim, obj_actor)

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)
            self.optim_update(self.cri_optim, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau) if self.cri_target is not self.cri else None

        return obj_critic.item(), obj_actor.item(), old_logprob.mean().item()  # logging_tuple

    def prepare_buffer(self, buffer):
        buf_len = buffer.now_len
        with torch.no_grad():  # compute reverse reward
            reward, mask, action, a_noise, state = buffer.sample_all()

            # print(';', [t.shape for t in (reward, mask, action, a_noise, state)])
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            value = torch.cat([self.cri_target(state[i:i + bs]) for i in range(0, state.size(0), bs)], dim=0)
            logprob = self.act.get_old_logprob(action, a_noise)

            pre_state = torch.as_tensor((self.state,), dtype=torch.float32, device=self.device)
            pre_r_sum = self.cri(pre_state).detach()
            r_sum, advantage = self.get_reward_sum(buf_len, reward, mask, value, pre_r_sum)
        return buf_len, state, action, r_sum, logprob, advantage

    def get_reward_sum(self, buf_len, reward, mask, value, pre_r_sum) -> (torch.Tensor, torch.Tensor):
        r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

        for i in range(buf_len - 1, -1, -1):
            r_sum[i] = reward[i] + mask[i] * pre_r_sum
            pre_r_sum = r_sum[i]
        advantage = r_sum - (mask * value.squeeze(1))
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)
        return r_sum, advantage


class AgentDiscretePPO(AgentPPO):
    def __init__(self):
        super().__init__()
        self.Act = ActorDiscretePPO

    def explore_env(self, env, target_step, reward_scale, gamma):
        trajectory_list = list()

        state = self.state
        for _ in range(target_step):
            a_int, a_prob = self.select_action(state)
            next_s, reward, done, _ = env.step(a_int)
            other = (reward * reward_scale, 0.0 if done else gamma, a_int, *a_prob)
            trajectory_list.append((state, other))

            state = env.reset() if done else next_s
        self.state = state
        return trajectory_list


class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, if_discrete, if_on_policy):
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.max_len = max_len
        self.if_on_policy = if_on_policy
        self.action_dim = 1 if if_discrete else action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if if_on_policy:
            other_dim = 1 + 1 + self.action_dim + action_dim
            # other = (reward, mask, action, a_noise) for continuous action
            # other = (reward, mask, a_int, a_prob) for discrete action
            self.buf_other = np.empty((max_len, other_dim), dtype=np.float32)
            self.buf_state = np.empty((max_len, state_dim), dtype=np.float32)
        else:
            other_dim = 1 + 1 + self.action_dim
            self.buf_other = torch.empty((max_len, other_dim), dtype=torch.float32, device=self.device)
            self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)

    def append_buffer(self, state, other):
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = other

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    def extend_buffer(self, state, other):
        size = len(other)
        next_idx = self.next_idx + size

        if next_idx > self.max_len:
            self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
            self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True

            next_idx = next_idx - self.max_len
            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx

    def extend_buffer_from_list(self, trajectory_list):
        if self.if_on_policy:
            state = np.array([item[0] for item in trajectory_list], dtype=np.float32)
            other = np.array([item[1] for item in trajectory_list], dtype=np.float32)
        else:
            state = torch.as_tensor([item[0] for item in trajectory_list], dtype=torch.float32)  # , device=self.device)
            other = torch.as_tensor([item[1] for item in trajectory_list], dtype=torch.float32)  # , device=self.device)
        self.extend_buffer(state, other)

    def sample_batch(self, batch_size) -> tuple:  # for off-policy only
        indices = rd.randint(self.now_len - 1, size=batch_size)
        other = self.buf_other[indices]  # reward, mask, action
        return (other[:, 0:1],  # reward
                other[:, 1:2],  # mask = 0.0 if done else gamma
                other[:, 2:],  # action
                self.buf_state[indices],  # state
                self.buf_state[indices + 1])  # next state

    def sample_all(self) -> tuple:  # for on-policy only
        all_state = torch.as_tensor(self.buf_state[:self.now_len], device=self.device)
        all_other = torch.as_tensor(self.buf_other[:self.now_len], device=self.device)
        return (all_other[:, 0],  # reward
                all_other[:, 1],  # mask = 0.0 if done else gamma
                all_other[:, 2:2 + self.action_dim],  # action
                all_other[:, 2 + self.action_dim:],  # action_noise or action_prob
                all_state,)  # state without last_state

    def update_now_len(self):
        self.now_len = self.max_len if self.if_full else self.next_idx

    def empty_buffer(self):
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
