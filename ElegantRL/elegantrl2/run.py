import os
import time
import torch
import numpy as np
import numpy.random as rd

from elegantrl2.evaluator import Evaluator
from elegantrl2.replay import ReplayBuffer, ReplayBufferMP
from elegantrl2.env import deepcopy_or_rebuild_env

"""[ElegantRL](https://github.com/AI4Finance-LLC/ElegantRL)"""


class Arguments:
    def __init__(self, agent=None, env=None, if_on_policy=False):
        self.agent = agent  # Deep Reinforcement Learning algorithm
        self.env = env  # the environment for training

        self.cwd = None  # current work directory. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = 2 ** 20  # break training after 'total_step > break_step'
        self.if_allow_break = True  # allow break training when reach goal (early termination)

        self.visible_gpu = '0'  # for example: os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'
        self.worker_num = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.num_threads = 8  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)

        '''Arguments for training'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.learning_rate = 2 ** -14  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3

        if if_on_policy:  # (on-policy)
            self.net_dim = 2 ** 9  # the network width
            self.batch_size = self.net_dim * 2  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 3  # collect target_step, then update network
            self.target_step = 2 ** 12  # repeatedly update network to keep critic's loss small
            self.max_memo = self.target_step  # capacity of replay buffer
            self.if_per_or_gae = False  # GAE for on-policy sparse reward: Generalized Advantage Estimation.
        else:
            self.net_dim = 2 ** 8  # the network width
            self.batch_size = self.net_dim  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
            self.target_step = 2 ** 10  # collect target_step, then update network
            self.max_memo = 2 ** 17  # capacity of replay buffer
            self.if_per_or_gae = False  # PER for off-policy sparse reward: Prioritized Experience Replay.

        '''Arguments for evaluate'''
        self.eval_env = None  # the environment for evaluating. None means set automatically.
        self.eval_gap = 2 ** 6  # evaluate the agent per eval_gap seconds
        self.eval_times1 = 2 ** 4  # number of times that get episode return in first
        self.eval_times2 = 2 ** 6  # number of times that get episode return in second
        self.random_seed = 0  # initialize random seed in self.init_before_training()

    def init_before_training(self, if_main):
        if self.agent is None:
            raise RuntimeError('\n| Why agent=None? Assignment args.agent = AgentXXX please.')
        if not hasattr(self.agent, 'init'):
            raise RuntimeError('\n| Should be agent=AgentXXX() instead of agent=AgentXXX')
        if self.env is None:
            raise RuntimeError('\n| Why env=None? Assignment args.env = XxxEnv() please.')
        if isinstance(self.env, str) or not hasattr(self.env, 'env_name'):
            raise RuntimeError('\n| What is env.env_name? use env=PreprocessEnv(env).')

        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            self.cwd = f'./{agent_name}_{self.env.env_name}_{self.visible_gpu}'

        if if_main:
            import shutil  # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input(f"| PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
            elif self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print(f"| Remove cwd: {self.cwd}")
            os.makedirs(self.cwd, exist_ok=True)

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpu)
        del self.visible_gpu


'''single processing training'''


def train_and_evaluate(args, agent_id=0):
    args.init_before_training(if_main=True)

    if True:
        '''basic arguments'''
        cwd = args.cwd
        agent = args.agent

        env = args.env
        state_dim = env.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete

        '''training arguments'''
        net_dim = args.net_dim
        max_memo = args.max_memo
        break_step = args.break_step
        batch_size = args.batch_size
        target_step = args.target_step
        repeat_times = args.repeat_times
        learning_rate = args.learning_rate
        if_break_early = args.if_allow_break

        gamma = args.gamma
        reward_scale = args.reward_scale
        if_per_or_gae = args.if_per_or_gae
        soft_update_tau = args.soft_update_tau

        '''evaluating arguments'''
        show_gap = args.eval_gap
        eval_env = args.eval_env
        eval_times1 = args.eval_times1
        eval_times2 = args.eval_times2
        del args

    '''init: Agent'''
    agent.init(net_dim, state_dim, action_dim, learning_rate, if_per_or_gae)
    agent.save_or_load_agent(cwd, if_save=False)
    if_on_policy = agent.if_on_policy

    '''init Evaluator'''
    eval_env = deepcopy_or_rebuild_env(env) if eval_env is None else eval_env
    evaluator = Evaluator(cwd=cwd, agent_id=agent_id, device=agent.device, env=eval_env,
                          eval_times1=eval_times1, eval_times2=eval_times2, eval_gap=show_gap)  # build Evaluator
    evaluator.save_or_load_recoder(if_save=False)

    '''init ReplayBuffer'''
    if if_on_policy:
        buffer = list()

        def update_buffer(_trajectory_list):
            buffer[:] = agent.prepare_buffer(_trajectory_list)  # buffer = (state, action, r_sum, logprob, advantage)

            _steps = buffer[2].size(0)  # buffer[2] = r_sum
            _r_exp = buffer[2].mean().item()  # buffer[2] = r_sum
            return _steps, _r_exp

        assert isinstance(buffer, list)
    else:
        buffer = ReplayBuffer(state_dim=state_dim, action_dim=1 if if_discrete else action_dim,
                              max_len=max_memo, if_use_per=if_per_or_gae)
        buffer.save_or_load_history(cwd, if_save=False)

        def update_buffer(_trajectory_list):
            _state = torch.as_tensor([item[0] for item in _trajectory_list], dtype=torch.float32)
            _other = torch.as_tensor([item[1] for item in _trajectory_list], dtype=torch.float32)
            buffer.extend_buffer(_state, _other)

            _steps = _other.size()[0]
            _r_exp = _other[:, 0].mean().item()  # other = (reward, mask, ...)
            return _steps, _r_exp

        assert isinstance(buffer, ReplayBuffer)

    '''start training'''
    if if_on_policy:
        agent.state = env.reset()
    elif buffer.max_len != 0:  # if_off_policy
        agent.state = env.reset()
    else:  # if_off_policy
        with torch.no_grad():  # update replay buffer
            trajectory_list = explore_before_training(env, target_step, reward_scale, gamma)
            steps, r_exp = update_buffer(trajectory_list)
            agent.state = trajectory_list[-1][0]  # trajectory_list[-1][0] = (state, other)[0] = state

        agent.update_net(buffer, target_step, batch_size, repeat_times)

        agent.act_target.load_state_dict(agent.act.state_dict()) if agent.if_use_act_target else None
        agent.cri_target.load_state_dict(agent.cri.state_dict()) if agent.if_use_cri_target else None
        evaluator.total_step += steps

    if_train = True
    while if_train:
        with torch.no_grad():
            trajectory_list = agent.explore_env(env, target_step, reward_scale, gamma)
            steps, r_exp = update_buffer(trajectory_list)

        logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)

        with torch.no_grad():
            if_reach_goal = evaluator.evaluate_and_save(agent.act, steps, r_exp, logging_tuple)
            if_train = not ((if_break_early and if_reach_goal)
                            or evaluator.total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))

    print(f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')

    agent.save_or_load_agent(cwd, if_save=True)
    buffer.save_or_load_history(cwd, if_save=True) if not if_on_policy else None
    evaluator.save_or_load_recoder(if_save=True)


'''multiple processing/GPU training'''


class CommEvaluate:
    def __init__(self):
        import multiprocessing as mp
        self.pipe = mp.Pipe()

    def evaluate_and_save0(self, act_cpu, evaluator, if_break_early, break_step, cwd):
        act_cpu_dict, steps, r_exp, logging_tuple = self.pipe[0].recv()

        if act_cpu_dict is None:
            if_reach_goal = False
            evaluator.total_step += steps
        else:
            act_cpu.load_state_dict(act_cpu_dict)
            if_reach_goal = evaluator.evaluate_and_save(act_cpu, steps, r_exp, logging_tuple)

        if_train = not ((if_break_early and if_reach_goal)
                        or evaluator.total_step > break_step
                        or os.path.exists(f'{cwd}/stop'))
        self.pipe[0].send(if_train)
        return if_train

    def evaluate_and_save1(self, agent_act, steps, r_exp, logging_tuple, if_train):
        if self.pipe[1].poll():  # if_evaluator_idle
            if_train = self.pipe[1].recv()

            act_cpu_dict = {k: v.cpu() for k, v in agent_act.state_dict().items()}
        else:
            act_cpu_dict = None

        self.pipe[1].send((act_cpu_dict, steps, r_exp, logging_tuple))
        return if_train


class CommExplore:
    def __init__(self, worker_num, if_on_policy):
        import multiprocessing as mp
        self.pipe_list = [mp.Pipe() for _ in range(worker_num)]

        self.worker_num = worker_num

        if if_on_policy:
            self.explore_env_update_buffer1 = self.explore1_on_policy
            self.explore_env_update_buffer0 = self.explore0_on_policy
        else:
            self.explore_env_update_buffer1 = self.explore1_off_policy
            self.explore_env_update_buffer0 = self.explore0_off_policy

    def explore1_on_policy(self, agent, buffer_mp):
        act_dict = agent.act.state_dict()
        cri_dict = agent.cri.state_dict()

        for i in range(self.worker_num):
            self.pipe_list[i][1].send((act_dict, cri_dict))

        del buffer_mp[:]  # on-policy
        steps, r_exp = 0, 0
        for i in range(self.worker_num):
            buffer_tuple, _steps, _r_exp = self.pipe_list[i][1].recv()

            buffer_mp.append(buffer_tuple)
            steps += _steps
            r_exp += _r_exp
        r_exp /= self.worker_num
        return buffer_mp, steps, r_exp

    def explore0_on_policy(self, worker_id, agent, env, target_step, reward_scale, gamma):
        act_dict, cri_dict = self.pipe_list[worker_id][0].recv()
        agent.act.load_state_dict(act_dict)
        agent.cri.load_state_dict(cri_dict)

        s_r_m_a_n_list = agent.explore_env(env, target_step, reward_scale, gamma)
        buffer_tuple = agent.prepare_buffer(s_r_m_a_n_list)
        _steps = buffer_tuple[2].size(0)  # buffer[2] = r_sum
        _r_exp = buffer_tuple[2].mean().item()  # buffer[2] = r_sum
        self.pipe_list[worker_id][0].send((buffer_tuple, _steps, _r_exp))

    def explore1_off_policy(self, agent, buffer_mp):
        act_dict = agent.act.state_dict()

        for i in range(self.worker_num):
            self.pipe_list[i][1].send(act_dict)

        buffer_tuples = list()
        steps, r_exp = 0, 0
        for i in range(self.worker_num):
            state, other, _steps, _r_exp = self.pipe_list[i][1].recv()

            state = torch.as_tensor(state, dtype=torch.float32, device=agent.device)
            other = torch.as_tensor(other, dtype=torch.float32, device=agent.device)
            buffer_mp.buffers[i].extend_buffer(state, other)
            buffer_tuples.append((state, other))
            steps += _steps
            r_exp += _r_exp
        r_exp /= self.worker_num
        return buffer_tuples, steps, r_exp

    def explore0_off_policy(self, worker_id, agent, env, target_step, reward_scale, gamma):
        act_dict = self.pipe_list[worker_id][0].recv()
        agent.act.load_state_dict(act_dict)

        trajectory_list = agent.explore_env(env, target_step, reward_scale, gamma)
        # state = torch.as_tensor([item[0] for item in trajectory_list], dtype=torch.float32, device=agent.device)
        # other = torch.as_tensor([item[1] for item in trajectory_list], dtype=torch.float32, device=agent.device)
        state = np.array([item[0] for item in trajectory_list], dtype=np.float16)
        other = np.array([item[1] for item in trajectory_list], dtype=np.float16)

        _steps = len(trajectory_list)
        _r_exp = other[0].mean().item()
        self.pipe_list[worker_id][0].send((state, other, _steps, _r_exp))

    def pre_explore1(self, agent, buffer_mp, batch_size, repeat_times, soft_update_tau):
        for i in range(self.worker_num):
            state, other = self.pipe_list[i][1].recv()
            buffer_mp.buffers[i].extend_buffer(state, other)

        agent.update_net(buffer_mp, batch_size, repeat_times, soft_update_tau)

        agent.act_target.load_state_dict(agent.act.state_dict()) if agent.if_use_act_target else None
        agent.cri_target.load_state_dict(agent.cri.state_dict()) if agent.if_use_cri_target else None

    def pre_explore0(self, worker_id, agent, env, target_step, reward_scale, gamma):
        trajectory_list = explore_before_training(env, target_step, reward_scale, gamma)
        state = torch.as_tensor([item[0] for item in trajectory_list], dtype=torch.float32, device=agent.device)
        other = torch.as_tensor([item[1] for item in trajectory_list], dtype=torch.float32, device=agent.device)
        self.pipe_list[worker_id][0].send((state, other))
        return trajectory_list[-1][0]  # trajectory_list[-1][0] = (state, other)[0] = state


def mp_learner(args, comm_eva, comm_exp):
    args.init_before_training(if_main=True)

    if True:
        '''basic arguments'''
        cwd = args.cwd
        agent = args.agent
        worker_num = args.worker_num

        env = args.env
        state_dim = env.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete

        '''training arguments'''
        net_dim = args.net_dim
        max_memo = args.max_memo
        # break_step = args.break_step
        batch_size = args.batch_size
        # target_step = args.target_step
        repeat_times = args.repeat_times
        learning_rate = args.learning_rate
        # if_break_early = args.if_allow_break

        # gamma = args.gamma
        # reward_scale = args.reward_scale
        if_per_or_gae = args.if_per_or_gae
        soft_update_tau = args.soft_update_tau

        '''evaluating arguments'''
        # show_gap = args.eval_gap
        # eval_env = args.eval_env
        # eval_times1 = args.eval_times1
        # eval_times2 = args.eval_times2
        del args

    '''init: Agent'''
    agent.init(net_dim, state_dim, action_dim, learning_rate, if_per_or_gae)
    agent.save_or_load_agent(cwd, if_save=False)
    if_on_policy = agent.if_on_policy

    '''init Evaluator'''
    # eval_env = deepcopy_or_rebuild_env(env) if eval_env is None else eval_env
    # evaluator = Evaluator(cwd=cwd, agent_id=agent_id, device=agent.device, env=eval_env,
    #                       eval_times1=eval_times1, eval_times2=eval_times2, eval_gap=show_gap)  # build Evaluator
    # evaluator.save_or_load_recoder(if_save=False)

    '''init ReplayBuffer'''
    if if_on_policy:
        buffer = [list() for _ in range(worker_num)]

        # def update_buffer(_buffer, _trajectory_list):
        #     _buffer[:] = agent.prepare_buffer(_trajectory_list)
        #     # _buffer = (state, action, r_sum, logprob, advantage)
        #
        #     _steps = _buffer[2].size(0)  # buffer[2] = r_sum
        #     _r_exp = _buffer[2].mean().item()  # buffer[2] = r_sum
        #     return _steps, _r_exp
    else:
        # buffer = ReplayBuffer(state_dim=state_dim, action_dim=1 if if_discrete else action_dim,
        #                       max_len=max_memo, if_use_per=if_per_or_gae)
        buffer = ReplayBufferMP(state_dim=state_dim, action_dim=1 if if_discrete else action_dim,
                                max_len=max_memo, if_use_per=if_per_or_gae,
                                worker_num=worker_num, gpu_id=0)  # todo
        buffer.save_or_load_history(cwd, if_save=False)

        # def update_buffer(_buffer, _trajectory_list):
        #     _state = torch.as_tensor([item[0] for item in _trajectory_list], dtype=torch.float32)
        #     _other = torch.as_tensor([item[1] for item in _trajectory_list], dtype=torch.float32)
        #     _buffer.extend_buffer(_state, _other, buffer_id=0)
        #
        #     _steps = _other.size()[0]
        #     _r_exp = _other[:, 0].mean().item()  # other = (reward, mask, ...)
        #     return _steps, _r_exp

    '''start training'''
    # if if_on_policy:
    #     agent.state = env.reset()
    # elif buffer.max_len != 0:  # if_off_policy
    #     agent.state = env.reset()
    # else:  # if_off_policy
    #     with torch.no_grad():  # update replay buffer
    #         trajectory_list = explore_before_training(env, target_step, reward_scale, gamma)
    #         steps, r_exp = update_buffer(buffer[0], trajectory_list)
    #         agent.state = trajectory_list[-1][0]  # trajectory_list[-1][0] = (state, other)[0] = state
    #
    #     agent.update_net(buffer, target_step, batch_size, repeat_times)
    #
    #     agent.act_target.load_state_dict(agent.act.state_dict()) if agent.if_use_act_target else None
    #     agent.cri_target.load_state_dict(agent.cri.state_dict()) if agent.if_use_cri_target else None
    #     # evaluator.total_step += steps
    if not if_on_policy:
        comm_exp.pre_explore1(agent, buffer, batch_size, repeat_times, soft_update_tau)

    if_train = True
    while if_train:
        # with torch.no_grad():
        #     trajectory_list = agent.explore_env(env, target_step, reward_scale, gamma)
        #     steps, r_exp = update_buffer(trajectory_list)
        buffer_tuples, steps, r_exp = comm_exp.explore_env_update_buffer1(agent, buffer)

        logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)

        # with torch.no_grad():
        #     if_reach_goal = evaluator.evaluate_and_save(agent.act, steps, r_exp, logging_tuple)
        #     if_train = not ((if_break_early and if_reach_goal)
        #                     or evaluator.total_step > break_step
        #                     or os.path.exists(f'{cwd}/stop'))
        if_train = comm_eva.evaluate_and_save1(agent.act, steps, r_exp, logging_tuple, if_train)

    # print(f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')

    agent.save_or_load_agent(cwd, if_save=True)
    buffer.save_or_load_history(cwd, if_save=True) if not if_on_policy else None
    # evaluator.save_or_load_recoder(if_save=True)


def mp_evaluator(args, comm_eva, agent_id=0):
    args.init_before_training(if_main=False)

    if True:
        '''basic arguments'''
        cwd = args.cwd
        agent = args.agent

        env = args.env
        state_dim = env.state_dim
        action_dim = env.action_dim
        # if_discrete = env.if_discrete

        '''training arguments'''
        net_dim = args.net_dim
        # max_memo = args.max_memo
        break_step = args.break_step
        # batch_size = args.batch_size
        # target_step = args.target_step
        # repeat_times = args.repeat_times
        learning_rate = args.learning_rate
        if_break_early = args.if_allow_break

        # gamma = args.gamma
        # reward_scale = args.reward_scale
        if_per_or_gae = args.if_per_or_gae
        # soft_update_tau = args.soft_update_tau

        '''evaluating arguments'''
        show_gap = args.eval_gap
        eval_env = args.eval_env
        eval_times1 = args.eval_times1
        eval_times2 = args.eval_times2
        del args

    '''init: Agent'''
    agent.init(net_dim, state_dim, action_dim, learning_rate, if_per_or_gae, gpu_id=-1)
    agent.save_or_load_agent(cwd, if_save=False)

    act_cpu = agent.act.to(torch.device("cpu"))
    act_cpu.eval()
    [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]
    del agent

    '''init Evaluator'''
    eval_env = deepcopy_or_rebuild_env(env) if eval_env is None else eval_env
    evaluator = Evaluator(cwd=cwd, agent_id=agent_id, device=torch.device("cpu"), env=eval_env,
                          eval_times1=eval_times1, eval_times2=eval_times2, eval_gap=show_gap)  # build Evaluator
    evaluator.save_or_load_recoder(if_save=False)

    if_train = True
    with torch.no_grad():
        while if_train:
            if_train = comm_eva.evaluate_and_save0(act_cpu, evaluator, if_break_early, break_step, cwd)

    print(f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')
    evaluator.save_or_load_recoder(if_save=True)


def mp_worker(args, comm_exp, worker_id, gpu_id=0):
    args.random_seed += gpu_id * args.worker_num + gpu_id
    args.init_before_training(if_main=False)

    if True:
        '''basic arguments'''
        cwd = args.cwd
        agent = args.agent

        env = args.env
        state_dim = env.state_dim
        action_dim = env.action_dim
        # if_discrete = env.if_discrete

        '''training arguments'''
        net_dim = args.net_dim
        # max_memo = args.max_memo
        # break_step = args.break_step
        # batch_size = args.batch_size
        target_step = args.target_step
        # repeat_times = args.repeat_times
        learning_rate = args.learning_rate
        # if_break_early = args.if_allow_break

        gamma = args.gamma
        reward_scale = args.reward_scale
        if_per_or_gae = args.if_per_or_gae
        # soft_update_tau = args.soft_update_tau

        '''evaluating arguments'''
        # show_gap = args.eval_gap
        # eval_env = args.eval_env
        # eval_times1 = args.eval_times1
        # eval_times2 = args.eval_times2
        del args

    '''init: Agent'''
    agent.init(net_dim, state_dim, action_dim, learning_rate, if_per_or_gae, gpu_id)
    agent.save_or_load_agent(cwd, if_save=False)
    if_on_policy = agent.if_on_policy

    if if_on_policy:
        agent.state = env.reset()
    else:
        agent.state = comm_exp.pre_explore0(worker_id, agent, env, target_step, reward_scale, gamma)

    with torch.no_grad():
        while True:
            comm_exp.explore_env_update_buffer0(worker_id, agent, env, target_step, reward_scale, gamma)


def train_and_evaluate_mp(args):
    import multiprocessing as mp  # Python built-in multiprocessing library

    '''init: mp.Pipe'''
    comm_eva = CommEvaluate()
    comm_exp = CommExplore(worker_num=args.worker_num, if_on_policy=args.agent.if_on_policy)

    process = list()
    process.append(mp.Process(target=mp_learner, args=(args, comm_eva, comm_exp)))
    process.append(mp.Process(target=mp_evaluator, args=(args, comm_eva)))
    for worker_id in range(args.worker_num):
        process.append(mp.Process(target=mp_worker, args=(args, comm_exp, worker_id,)))

    [p.start() for p in process]
    process[0].join()
    process_safely_terminate(process)


class CommGPU:
    def __init__(self, gpu_num, if_on_policy):
        import multiprocessing as mp
        self.pipe_list = [mp.Pipe() for _ in range(gpu_num)]
        self.device_list = [torch.device(f'cuda:{i}') for i in range(gpu_num)]
        self.gpu_num = gpu_num

        self.round_num = int(np.log2(gpu_num))
        if gpu_num == 2:
            self.idx_l = [(1,), (0,), ]
        elif gpu_num == 4:
            self.idx_l = [(1, 2), (0, 3),
                          (3, 0), (2, 1), ]
        elif gpu_num == 8:
            self.idx_l = [(1, 2, 4), (0, 3, 5),
                          (3, 0, 6), (2, 1, 7),
                          (5, 6, 0), (4, 7, 1),
                          (7, 4, 2), (6, 5, 3), ]
        else:
            print(f"| LearnerComm, ERROR: learner_num {gpu_num} should in (2, 4, 8)")
            exit()

        if if_on_policy:
            self.comm_buffer = self.comm_buffer_on_policy
        else:
            self.comm_buffer = self.comm_buffer_off_policy

    def comm_data(self, data, gpu_id, round_id, if_cuda=False):
        idx = self.idx_l[gpu_id][round_id]

        data = [[t.to(self.device_list[idx]) for t in item]
                for item in data] if if_cuda else data

        self.pipe_list[idx][0].send(data)
        return self.pipe_list[gpu_id][1].recv()

    def comm_buffer_on_policy(self, buffer, buffer_tuples, gpu_id):
        buffer_tuples = self.comm_data(buffer_tuples, gpu_id, round_id=0, if_cuda=True)
        buffer.extend(buffer_tuples)

    def comm_buffer_off_policy(self, buffer, buffer_tuples, gpu_id):
        new_buffer = self.comm_data(buffer_tuples, gpu_id, round_id=0)

        for worker_i, (state, other) in enumerate(new_buffer):
            buffer.buffers[worker_i].extend_buffer(state, other)

    def comm_network_optim(self, agent, gpu_id):
        for round_id in range(self.round_num):
            cri = agent.cri if (agent.cri is not agent.act) else None
            cri_optim = agent.cri_optim if (agent.cri_optim is not agent.act_optim) else None

            act_target = agent.act_target if agent.if_use_act_target else None
            cri_target = agent.cri_target if agent.if_use_cri_target else None

            data = agent.act, agent.act_optim, cri, cri_optim, act_target, cri_target,
            data = self.comm_data(data, gpu_id, round_id)

            if data is None:
                continue

            avg_update_net(agent.act, data[0], agent.device)
            avg_update_optim(agent.act_optim, data[1], agent.device)

            avg_update_net(agent.cri, data[2], agent.device) if (data[2] is not None) else None
            avg_update_optim(agent.cri_optim, data[3], agent.device) if (data[3] is not None) else None

            avg_update_net(agent.act_target, data[4], agent.device) if agent.if_use_act_target else None
            avg_update_net(agent.cri_target, data[5], agent.device) if agent.if_use_cri_target else None

    def close_itself(self):
        for pipe in self.pipe_list:
            for p in pipe:
                try:
                    while p.poll():
                        p.recv()
                except EOFError:
                    pass


def mg_learner(args, comm_eva, comm_exp, comm_gpu, gpu_id):
    args.init_before_training(if_main=bool(gpu_id == 0))

    if True:
        '''basic arguments'''
        cwd = args.cwd
        agent = args.agent
        worker_num = args.worker_num

        env = args.env
        state_dim = env.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete

        '''training arguments'''
        net_dim = args.net_dim
        max_memo = args.max_memo
        # break_step = args.break_step
        batch_size = args.batch_size
        # target_step = args.target_step
        repeat_times = args.repeat_times
        learning_rate = args.learning_rate
        # if_break_early = args.if_allow_break

        # gamma = args.gamma
        # reward_scale = args.reward_scale
        if_per_or_gae = args.if_per_or_gae
        soft_update_tau = args.soft_update_tau

        '''evaluating arguments'''
        # show_gap = args.eval_gap
        # eval_env = args.eval_env
        # eval_times1 = args.eval_times1
        # eval_times2 = args.eval_times2
        del args

    '''init: Agent'''
    agent.init(net_dim, state_dim, action_dim, learning_rate, if_per_or_gae, gpu_id)
    agent.save_or_load_agent(cwd, if_save=False)
    if_on_policy = agent.if_on_policy

    '''init Evaluator'''
    # eval_env = deepcopy_or_rebuild_env(env) if eval_env is None else eval_env
    # evaluator = Evaluator(cwd=cwd, agent_id=agent_id, device=agent.device, env=eval_env,
    #                       eval_times1=eval_times1, eval_times2=eval_times2, eval_gap=show_gap)  # build Evaluator
    # evaluator.save_or_load_recoder(if_save=False)

    '''init ReplayBuffer'''
    if if_on_policy:
        buffer = [list() for _ in range(worker_num)]

        # def update_buffer(_buffer, _trajectory_list):
        #     _buffer[:] = agent.prepare_buffer(_trajectory_list)
        #     # _buffer = (state, action, r_sum, logprob, advantage)
        #
        #     _steps = _buffer[2].size(0)  # buffer[2] = r_sum
        #     _r_exp = _buffer[2].mean().item()  # buffer[2] = r_sum
        #     return _steps, _r_exp
    else:
        # buffer = ReplayBuffer(state_dim=state_dim, action_dim=1 if if_discrete else action_dim,
        #                       max_len=max_memo, if_use_per=if_per_or_gae)
        buffer = ReplayBufferMP(state_dim=state_dim, action_dim=1 if if_discrete else action_dim,
                                max_len=max_memo, if_use_per=if_per_or_gae,
                                worker_num=worker_num, gpu_id=gpu_id)
        buffer.save_or_load_history(cwd, if_save=False)

        # def update_buffer(_buffer, _trajectory_list):
        #     _state = torch.as_tensor([item[0] for item in _trajectory_list], dtype=torch.float32)
        #     _other = torch.as_tensor([item[1] for item in _trajectory_list], dtype=torch.float32)
        #     _buffer.extend_buffer(_state, _other, buffer_id=0)
        #
        #     _steps = _other.size()[0]
        #     _r_exp = _other[:, 0].mean().item()  # other = (reward, mask, ...)
        #     return _steps, _r_exp

    '''start training'''
    # if if_on_policy:
    #     agent.state = env.reset()
    # elif buffer.max_len != 0:  # if_off_policy
    #     agent.state = env.reset()
    # else:  # if_off_policy
    #     with torch.no_grad():  # update replay buffer
    #         trajectory_list = explore_before_training(env, target_step, reward_scale, gamma)
    #         steps, r_exp = update_buffer(buffer[0], trajectory_list)
    #         agent.state = trajectory_list[-1][0]  # trajectory_list[-1][0] = (state, other)[0] = state
    #
    #     agent.update_net(buffer, target_step, batch_size, repeat_times)
    #
    #     agent.act_target.load_state_dict(agent.act.state_dict()) if agent.if_use_act_target else None
    #     agent.cri_target.load_state_dict(agent.cri.state_dict()) if agent.if_use_cri_target else None
    #     # evaluator.total_step += steps
    if not if_on_policy:
        comm_exp.pre_explore1(agent, buffer, batch_size, repeat_times, soft_update_tau)

    if_train = True
    while if_train:
        with torch.no_grad():
            # trajectory_list = agent.explore_env(env, target_step, reward_scale, gamma)
            # steps, r_exp = update_buffer(trajectory_list)
            buffer_tuples, steps, r_exp = comm_exp.explore_env_update_buffer1(agent, buffer)
            if comm_gpu is not None:
                comm_gpu.comm_buffer(buffer, buffer_tuples, gpu_id)

        logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)

        with torch.no_grad():
            # if_reach_goal = evaluator.evaluate_and_save(agent.act, steps, r_exp, logging_tuple)
            # if_train = not ((if_break_early and if_reach_goal)
            #                 or evaluator.total_step > break_step
            #                 or os.path.exists(f'{cwd}/stop'))
            if comm_gpu is not None:
                comm_gpu.comm_network_optim(agent, gpu_id)

            if comm_eva is not None:
                if_train = comm_eva.evaluate_and_save1(agent.act, steps, r_exp, logging_tuple, if_train)

    # print(f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')

    agent.save_or_load_agent(cwd, if_save=True)
    buffer.save_or_load_history(cwd, if_save=True) if not if_on_policy else None
    # evaluator.save_or_load_recoder(if_save=True)

    comm_gpu.close_itself() if comm_gpu is not None else None


def train_and_evaluate_mg(args):
    import multiprocessing as mp  # Python built-in multiprocessing library

    '''init: mp.Pipe'''

    eval_visible_gpu = eval(args.visible_gpu)
    gpu_num = len(eval_visible_gpu) if isinstance(eval_visible_gpu, tuple) else 1
    comm_gpu = CommGPU(gpu_num=gpu_num, if_on_policy=args.agent.if_on_policy)

    process = list()
    for gpu_id in range(gpu_num):
        comm_eva = CommEvaluate() if gpu_id == 0 else None
        comm_exp = CommExplore(worker_num=args.worker_num, if_on_policy=args.agent.if_on_policy)

        process.append(mp.Process(target=mg_learner, args=(args, comm_eva, comm_exp, comm_gpu, gpu_id)))
        process.append(mp.Process(target=mp_evaluator, args=(args, comm_eva))) if comm_eva is not None else None
        for worker_id in range(args.worker_num):
            process.append(mp.Process(target=mp_worker, args=(args, comm_exp, worker_id, gpu_id)))

    [p.start() for p in process]
    process[0].join()
    process_safely_terminate(process)


"""utils"""


def process_safely_terminate(process):
    for p in process:
        try:
            p.terminate()
        except OSError as e:
            print(e)
            pass


def explore_before_training(env, target_step, reward_scale, gamma) -> (list, np.ndarray):  # for off-policy only
    trajectory_list = list()

    if_discrete = env.if_discrete
    action_dim = env.action_dim

    state = env.reset()
    step = 0
    while True:
        if if_discrete:
            action = rd.randint(action_dim)  # assert isinstance(action_int)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, action)
        else:
            action = rd.uniform(-1, 1, size=action_dim)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, *action)

        trajectory_list.append((state, other))
        state = env.reset() if done else next_s

        step += 1
        if done and step > target_step:
            break
    return trajectory_list


def empty_pipe_list(pipe_list):
    for pipe in pipe_list:
        try:
            while pipe.poll():
                pipe.recv()
        except EOFError:
            pass


def get_optim_parameters(optim):  # for avg_update_optim()
    params_list = list()
    for params_dict in optim.state_dict()['state'].values():
        params_list.extend([t for t in params_dict.values() if isinstance(t, torch.Tensor)])
    return params_list


def avg_update_optim(dst_optim, src_optim, device):
    for dst, src in zip(get_optim_parameters(dst_optim), get_optim_parameters(src_optim)):
        dst.data.copy_((dst.data + src.data.to(device)) * 0.5)
        # dst.data.copy_(src.data * tau + dst.data * (1 - tau))


def avg_update_net(dst_net, src_net, device):
    for dst, src in zip(dst_net.parameters(), src_net.parameters()):
        dst.data.copy_((dst.data + src.data.to(device)) * 0.5)


def load_update_optim(dst_optim, src_optim, device):
    for dst, src in zip(get_optim_parameters(dst_optim), get_optim_parameters(src_optim)):
        dst.data.copy_(src.data.to(device))


def load_update_net(dst_net, src_net, device):
    for dst, src in zip(dst_net.parameters(), src_net.parameters()):
        dst.data.copy_(src.data.to(device))
