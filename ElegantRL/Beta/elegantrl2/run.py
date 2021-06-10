import os
import time
import torch
import numpy as np
import numpy.random as rd
from copy import deepcopy

from elegantrl2.replay import ReplayBuffer, ReplayBufferMP
from elegantrl2.env import deepcopy_or_rebuild_env

"""[ElegantRL](https://github.com/AI4Finance-LLC/ElegantRL)"""


class Arguments:
    def __init__(self, agent=None, env=None, gpu_id=None, if_on_policy=False):
        self.agent = agent  # Deep Reinforcement Learning algorithm
        self.cwd = None  # current work directory. cwd is None means set it automatically
        self.env = env  # the environment for training
        self.env_eval = None  # the environment for evaluating
        self.gpu_id = gpu_id  # choose the GPU for running. gpu_id is None means set it automatically
        self.worker_num = 2  # the number of rollout workers (larger is not always faster)
        self.num_threads = 8  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)

        '''Arguments for training (off-policy)'''
        self.learning_rate = 2 ** -14  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256

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
            self.target_step = 2 ** 10  # collect target_step, then update network
            self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
            self.max_memo = 2 ** 17  # capacity of replay buffer
            self.if_per_or_gae = False  # PER for off-policy sparse reward: Prioritized Experience Replay.

        '''Arguments for evaluate'''
        self.eval_gap = 2 ** 6  # evaluate the agent per eval_gap seconds
        self.eval_times1 = 2 ** 4  # todo 2  # evaluation times
        self.eval_times2 = 2 ** 6  # todo 4  # evaluation times if 'eval_reward > max_reward'
        self.random_seed = 0  # initialize random seed in self.init_before_training()

        self.break_step = 2 ** 20  # break training after 'total_step > break_step'
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.if_allow_break = True  # allow break training when reach goal (early termination)

    def init_before_training(self, process_id=0):
        if self.agent is None:
            raise RuntimeError('\n| Why agent=None? Assignment args.agent = AgentXXX please.')
        if not hasattr(self.agent, 'init'):
            raise RuntimeError('\n| Should be agent=AgentXXX() instead of agent=AgentXXX')
        if self.env is None:
            raise RuntimeError('\n| Why env=None? Assignment args.env = XxxEnv() please.')
        if isinstance(self.env, str) or not hasattr(self.env, 'env_name'):
            raise RuntimeError('\n| What is env.env_name? use env=PreprocessEnv(env). It is a Wrapper.')

        '''set None value automatically'''
        if self.gpu_id is None:  # set gpu_id as '0' in default
            self.gpu_id = '0'

        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            self.cwd = f'./{self.env.env_name}_{agent_name}'

        if process_id == 0:
            print(f'| GPU id: {self.gpu_id}, cwd: {self.cwd}')

            import shutil  # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input("PRESS 'y' to REMOVE: {}? ".format(self.cwd)) == 'y')
            if self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print("| Remove history")
            os.makedirs(self.cwd, exist_ok=True)

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        if isinstance(self.gpu_id, tuple) or isinstance(self.gpu_id, list):
            gpu_id_str = str(self.gpu_id)[1:-1]  # for example "0, 1"
        else:
            gpu_id_str = str(self.gpu_id)  # for example "1"
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id_str


'''single process training'''


def train_and_evaluate(args):
    args.init_before_training()

    '''basic arguments'''
    cwd = args.cwd
    env = args.env
    agent = args.agent
    gpu_id = args.gpu_id

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
    eval_times1 = args.eval_times1
    eval_times2 = args.eval_times2
    # if_vec_env = getattr(env, 'env_num', 1) > 1
    env_eval = deepcopy_or_rebuild_env(env) if args.env_eval is None else args.env_eval
    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: environment'''
    state_dim = env.state_dim
    action_dim = env.action_dim
    # if_discrete = env.if_discrete
    env_eval = deepcopy(env) if env_eval is None else deepcopy(env_eval)

    '''init: Agent, ReplayBuffer, Evaluator'''
    agent.init(net_dim, state_dim, action_dim, learning_rate, if_per_or_gae)
    if_on_policy = getattr(agent, 'if_on_policy', False)

    buffer = ReplayBuffer(max_len=max_memo, state_dim=state_dim, action_dim=action_dim,
                          if_use_per=if_per_or_gae) if if_on_policy else tuple()

    evaluator = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval,
                          eval_times1=eval_times1, eval_times2=eval_times2, eval_gap=show_gap)  # build Evaluator

    '''prepare for training'''
    agent.state = env.reset()
    total_step = 0

    '''start training'''
    if_train = True
    while if_train:
        with torch.no_grad():
            if if_on_policy:
                buffer_tuple1 = agent.explore_env(env, target_step, reward_scale, gamma)
                buffer_tuple2 = agent.prepare_buffer(buffer_tuple1)
                steps = buffer_tuple2[0].size(0)

                buffer = buffer_tuple2
            else:
                trajectory_list = agent.explore_env(env, target_step, reward_scale, gamma)
                steps = len(trajectory_list)

                buffer.extend_buffer_from_list(trajectory_list)
        total_step += steps

        # assert if_on_policy and isinstance(buffer, tuple)
        # assert (not if_on_policy) and isinstance(buffer, ReplayBuffer)
        logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)

        with torch.no_grad():
            if_reach_goal = evaluator.evaluate_save(agent.act, steps, logging_tuple)
            if_train = not ((if_break_early and if_reach_goal)
                            or total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))

    print(f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')


'''multiprocessing training'''


def train_and_evaluate_mg(args):  # multiple GPU
    import multiprocessing as mp  # Python built-in multiprocessing library
    process = list()

    args.gpu_id = (0, 1)
    pipe_net_list = list()
    for learner_id in range(len(args.gpu_id)):
        pipe_net_list.append(mp.Pipe())

    for learner_id in range(len(args.gpu_id)):
        pipe_eva = mp.Pipe()
        process.append(mp.Process(target=mp_evaluator, args=(args, pipe_eva)))

        pipe_exp_list = list()
        for i in range(args.worker_num):
            pipe_exp = mp.Pipe()
            pipe_exp_list.append(pipe_exp)
            process.append(mp.Process(target=mp_worker, args=(args, pipe_exp, i)))
        process.append(mp.Process(target=mg_learner, args=(args, pipe_exp_list, pipe_eva,
                                                           pipe_net_list, learner_id)))

    [p.start() for p in process]
    [p.join() for p in (process[-1],)]  # wait
    [p.terminate() for p in process]


def mg_learner(args, pipe_exp_list, pipe_eva, pipe_net_list, learner_id):
    args.init_before_training(process_id=learner_id)

    if True:
        '''arguments: basic'''
        # cwd = args.cwd
        env = args.env
        agent = args.agent
        # gpu_id = args.gpu_id
        worker_num = args.worker_num

        '''arguments: train'''
        net_dim = args.net_dim
        max_memo = args.max_memo
        # break_step = args.break_step
        batch_size = args.batch_size
        target_step = args.target_step
        repeat_times = args.repeat_times
        learning_rate = args.learning_rate
        # if_break_early = args.if_allow_break

        gamma = args.gamma
        reward_scale = args.reward_scale
        if_per_or_gae = args.if_per_or_gae
        soft_update_tau = args.soft_update_tau

        '''arguments: evaluate'''
        # show_gap = args.eval_gap
        # eval_times1 = args.eval_times1
        # eval_times2 = args.eval_times2
        # env_eval = deepcopy_or_rebuild_env(env) if args.env_eval is None else args.env_eval

        '''arguments: environment'''
        # max_step = env.max_step
        state_dim = env.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete
        del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: Agent, ReplayBuffer, Evaluator'''
    agent.init(net_dim, state_dim, action_dim, learning_rate, if_per_or_gae)
    agent.device = torch.device(f'cuda:{learner_id}')  # todo here
    if_on_policy = agent.if_on_policy

    '''init: ReplayBuffer'''
    # agent.state = env.reset()
    if if_on_policy:
        steps = 0
        buffer = None
    else:  # explore_before_training for off-policy
        buffer = ReplayBufferMP(max_len=target_step if if_on_policy else max_memo, worker_num=worker_num,
                                if_on_policy=if_on_policy, if_per_or_gae=if_per_or_gae,
                                state_dim=state_dim, action_dim=action_dim, if_discrete=if_discrete, )

        with torch.no_grad():  # update replay buffer
            trajectory_list, state = explore_before_training(env, target_step, reward_scale, gamma)
        agent.state = state
        steps = len(trajectory_list)

        buffer.buffers[0].extend_buffer_from_list(trajectory_list)
        agent.update_net(buffer, target_step, batch_size, repeat_times)  # pre-training and hard update

        # hard update for the first time
        agent.act_target.load_state_dict(agent.act.state_dict()) if getattr(agent, 'act_target', None) else None
        agent.cri_target.load_state_dict(agent.cri.state_dict()) if getattr(agent, 'cri_target', None) else None
    total_step = steps

    '''init: Evaluator'''
    # evaluator = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval,
    #                       eval_times1=eval_times1, eval_times2=eval_times2, eval_gap=show_gap)  # build Evaluator
    act_cpu = deepcopy(agent.act).to(torch.device("cpu"))  # for pipe1_eva
    act_cpu.eval()
    [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]

    pipe_eva[1].send((act_cpu, steps))
    # act_cpu, steps = pipe_eva[0].recv()

    '''start training'''
    if_train = True
    while if_train:
        '''explore'''
        steps = 0
        # with torch.no_grad():
        #     trajectory_list = agent.explore_env(env, target_step, reward_scale, gamma)
        # buffer.extend_buffer_from_list(trajectory_list)
        if if_on_policy:
            act_state_dict = agent.act.state_dict()
            cri_target_state_dict = agent.cri_target.state_dict()
            for pipe_exp in pipe_exp_list:
                pipe_exp[1].send((act_state_dict, cri_target_state_dict))
                # act_state_dict, cri_target_state_dict = pipe_exp[0].recv()

            # buffer.extend_buffer_from_list(trajectory_list)
            buffer_tuples = list()
            for pipe_exp in pipe_exp_list:
                # pipe_exp[0].send(buffer_tuple)
                buffer_tuple = pipe_exp[1].recv()

                # steps += buffer_tuple[0] #
                steps += buffer_tuple[0].size(0)  # todo
                buffer_tuples.append(buffer_tuple)
            logging_tuple = agent.update_net(buffer_tuples, batch_size, repeat_times, soft_update_tau)

        else:
            agent.state = list()
            for pipe_exp in pipe_exp_list:
                pipe_exp[1].send(agent.act.state_dict())
                # act_state_dict = pipe_exp[0].recv()

            for pipe_exp, buffer_i in zip(pipe_exp_list, buffer.buffers):
                # pipe_exp[0].send((trajectory_list, agent.state))
                trajectory_list, agent_state = pipe_exp[1].recv()

                agent.state.append(agent_state)
                # steps = len(trajectory_list)
                steps += trajectory_list[0].shape[0]

                state_ary, other_ary = trajectory_list
                buffer_i.extend_buffer(state_ary, other_ary)
            logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)
        total_step += steps

        '''evaluate'''
        if not pipe_eva[0].poll():
            act_cpu.load_state_dict(agent.act.state_dict())
            act_state_dict = act_cpu.state_dict()
        else:
            act_state_dict = None
        pipe_eva[1].send((act_state_dict, steps, logging_tuple))
        # act_state_dict, steps, logging_tuple = pipe_eva[0].recv()

        # with torch.no_grad():  # speed up running
        #     if_reach_goal = evaluator.evaluate_save(agent.act, steps, logging_tuple)
        # if_train = not ((if_break_early and if_reach_goal)
        #                 or total_step > break_step
        #                 or os.path.exists(f'{cwd}/stop'))

        if pipe_eva[1].poll():
            # pipe_eva[0].send(if_train)
            if_train = pipe_eva[1].recv()

    pipe_list = list()
    pipe_list.extend(pipe_eva)
    for pipe_exp in pipe_exp_list:
        pipe_list.extend(pipe_exp)
    for pipe in pipe_list:
        try:
            while pipe.poll():
                pipe.recv()
        except EOFError:
            pass


def soft_update(target_net, current_net, tau):
    """soft update a target network via current network

    `nn.Module target_net` target network update via a current network, it is more stable
    `nn.Module current_net` current network update via an optimizer
    """
    for tar, cur in zip(target_net.parameters(), current_net.parameters()):
        tar.data.copy_(cur.data * tau + tar.data * (1 - tau))


def train_and_evaluate_mp(args):  # multiple processing
    import multiprocessing as mp  # Python built-in multiprocessing library
    process = list()

    learner_id = 0
    pipe_eva = mp.Pipe()
    process.append(mp.Process(target=mp_evaluator, args=(args, pipe_eva)))

    pipe_exp_list = list()
    for worker_id in range(args.worker_num):
        pipe_exp = mp.Pipe()
        pipe_exp_list.append(pipe_exp)
        process.append(mp.Process(target=mp_worker, args=(args, pipe_exp, worker_id, learner_id)))
    process.append(mp.Process(target=mp_learner, args=(args, pipe_exp_list, pipe_eva)))

    [p.start() for p in process]
    [p.join() for p in (process[-1],)]  # wait
    [p.terminate() for p in process]


def mp_worker(args, pipe_exp, worker_id, learner_id):
    args.random_seed += worker_id + learner_id * args.worker_num
    args.init_before_training(process_id=-1)

    if True:
        '''arguments: basic'''
        # cwd = args.cwd
        env = args.env
        agent = args.agent
        # gpu_id = args.gpu_id
        worker_num = args.worker_num

        '''arguments: train'''
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

        '''arguments: evaluate'''
        # show_gap = args.eval_gap
        # eval_times1 = args.eval_times1
        # eval_times2 = args.eval_times2
        # env_eval = deepcopy_or_rebuild_env(env) if args.env_eval is None else args.env_eval

        '''arguments: environment'''
        # max_step = env.max_step
        state_dim = env.state_dim
        action_dim = env.action_dim
        # if_discrete = env.if_discrete
        del args  # In order to show these hyper-parameters clearly, I put them above.

    env_num = getattr(env, 'env_num', 0)
    '''init: Agent, ReplayBuffer, Evaluator'''
    agent.init(net_dim, state_dim, action_dim, learning_rate, if_per_or_gae, learner_id)
    # agent.act.eval()
    # [setattr(param, 'requires_grad', False) for param in agent.act.parameters()]

    if_on_policy = agent.if_on_policy

    if env_num:
        agent.state = env.reset_vec()
        agent.env_tensors = [[torch.zeros(0, dtype=torch.float32, device=agent.device)
                              for _ in range(5)]
                             for _ in range(env.env_num)]
        # 5 == len(states, actions, r_sums, logprobs, advantages)

        with torch.no_grad():
            while True:
                # pipe_exp[1].send((agent.act.state_dict(), agent.cri_target.state_dict()))
                act_state_dict, cri_target_state_dict = pipe_exp[0].recv()

                agent.act.load_state_dict(act_state_dict)
                agent.cri_target.load_state_dict(cri_target_state_dict)

                buffer = agent.explore_envs(env, target_step, reward_scale, gamma)
                buffer_tuple = agent.prepare_buffers(buffer)

                pipe_exp[0].send(buffer_tuple)
                # buffer_tuple = pipe_exp[1].recv()
    if if_on_policy:
        agent.state = env.reset()
        with torch.no_grad():
            while True:
                # pipe_exp[1].send((act_state_dict, cri_target_state_dict))
                act_state_dict, cri_target_state_dict = pipe_exp[0].recv()

                agent.act.load_state_dict(act_state_dict)
                agent.cri_target.load_state_dict(cri_target_state_dict)

                # trajectory_list = agent.explore_env(env, target_step, reward_scale, gamma)
                buffer = agent.explore_env(env, target_step, reward_scale, gamma)
                # buffer.extend_buffer_from_list(trajectory_list)
                buffer_tuple = agent.prepare_buffer(buffer)
                pipe_exp[0].send(buffer_tuple)
                # buffer_tuple = pipe_exp[1].recv()
    else:
        with torch.no_grad():
            while True:
                # pipe_exp[1].send(agent.act.state_dict())
                act_state_dict = pipe_exp[0].recv()

                agent.act.load_state_dict(act_state_dict)
                trajectory_list = agent.explore_env(env, target_step, reward_scale, gamma)

                state = torch.as_tensor([item[0] for item in trajectory_list],
                                        dtype=torch.float32, device=agent.device)
                other = torch.as_tensor([item[1] for item in trajectory_list],
                                        dtype=torch.float32, device=agent.device)
                trajectory_list = (state, other)

                pipe_exp[0].send(trajectory_list)
                # trajectory_list = pipe_exp[1].recv()


def mp_learner(args, pipe_exp_list, pipe_eva, process_id=0):
    args.init_before_training(process_id=process_id)

    if True:
        '''arguments: basic'''
        # cwd = args.cwd
        env = args.env
        agent = args.agent
        # gpu_id = args.gpu_id
        worker_num = args.worker_num

        '''arguments: train'''
        net_dim = args.net_dim
        max_memo = args.max_memo
        # break_step = args.break_step
        batch_size = args.batch_size
        target_step = args.target_step
        repeat_times = args.repeat_times
        learning_rate = args.learning_rate
        # if_break_early = args.if_allow_break

        gamma = args.gamma
        reward_scale = args.reward_scale
        if_per_or_gae = args.if_per_or_gae
        soft_update_tau = args.soft_update_tau

        '''arguments: evaluate'''
        # show_gap = args.eval_gap
        # eval_times1 = args.eval_times1
        # eval_times2 = args.eval_times2
        # env_eval = deepcopy_or_rebuild_env(env) if args.env_eval is None else args.env_eval

        '''arguments: environment'''
        # max_step = env.max_step
        state_dim = env.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete
        del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: Agent'''
    agent.init(net_dim, state_dim, action_dim, learning_rate, if_per_or_gae)
    if_on_policy = agent.if_on_policy

    '''init: ReplayBuffer'''
    # agent.state = env.reset()
    if if_on_policy:
        steps = 0
        buffer = None
    else:  # explore_before_training for off-policy
        buffer = ReplayBufferMP(max_len=target_step if if_on_policy else max_memo, worker_num=worker_num,
                                if_on_policy=if_on_policy, if_per_or_gae=if_per_or_gae,
                                state_dim=state_dim, action_dim=action_dim, if_discrete=if_discrete, )

        with torch.no_grad():  # update replay buffer
            trajectory_list, state = explore_before_training(env, target_step, reward_scale, gamma)
        agent.state = state
        steps = len(trajectory_list)

        buffer.buffers[0].extend_buffer_from_list(trajectory_list)
        agent.update_net(buffer, target_step, batch_size, repeat_times)  # pre-training and hard update

        # hard update for the first time
        agent.act_target.load_state_dict(agent.act.state_dict()) if getattr(agent, 'act_target', None) else None
        agent.cri_target.load_state_dict(agent.cri.state_dict()) if getattr(agent, 'cri_target', None) else None
    total_step = steps

    '''init: Evaluator'''
    # evaluator = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval,
    #                       eval_times1=eval_times1, eval_times2=eval_times2, eval_gap=show_gap)  # build Evaluator
    act_cpu = deepcopy(agent.act).to(torch.device("cpu"))  # for pipe1_eva
    act_cpu.eval()
    [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]

    pipe_eva[1].send((act_cpu, steps))
    # act_cpu, steps = pipe_eva[0].recv()

    '''start training'''
    if_train = True
    while if_train:
        '''explore'''
        steps = 0
        # with torch.no_grad():
        #     trajectory_list = agent.explore_env(env, target_step, reward_scale, gamma)
        # buffer.extend_buffer_from_list(trajectory_list)
        if if_on_policy:
            act_state_dict = agent.act.state_dict()
            cri_target_state_dict = agent.cri_target.state_dict()
            for pipe_exp in pipe_exp_list:
                pipe_exp[1].send((act_state_dict, cri_target_state_dict))
                # act_state_dict, cri_target_state_dict = pipe_exp[0].recv()

            # buffer.extend_buffer_from_list(trajectory_list)
            buffer_tuples = list()
            for pipe_exp in pipe_exp_list:
                # pipe_exp[0].send(buffer_tuple)
                buffer_tuple = pipe_exp[1].recv()

                # steps += buffer_tuple[0] #
                steps += buffer_tuple[0].size(0)  # todo
                buffer_tuples.append(buffer_tuple)
            logging_tuple = agent.update_net(buffer_tuples, batch_size, repeat_times, soft_update_tau)

        else:
            agent.state = list()
            for pipe_exp in pipe_exp_list:
                pipe_exp[1].send(agent.act.state_dict())
                # act_state_dict = pipe_exp[0].recv()

            for pipe_exp, buffer_i in zip(pipe_exp_list, buffer.buffers):
                # pipe_exp[0].send((trajectory_list, agent.state))
                trajectory_list, agent_state = pipe_exp[1].recv()

                agent.state.append(agent_state)
                # steps = len(trajectory_list)
                steps += trajectory_list[0].shape[0]

                state_ary, other_ary = trajectory_list
                buffer_i.extend_buffer(state_ary, other_ary)
            logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)
        total_step += steps

        '''evaluate'''
        if not pipe_eva[0].poll():
            act_cpu.load_state_dict(agent.act.state_dict())
            act_state_dict = act_cpu.state_dict()
        else:
            act_state_dict = None
        pipe_eva[1].send((act_state_dict, steps, logging_tuple))
        # act_state_dict, steps, logging_tuple = pipe_eva[0].recv()

        # with torch.no_grad():  # speed up running
        #     if_reach_goal = evaluator.evaluate_save(agent.act, steps, logging_tuple)
        # if_train = not ((if_break_early and if_reach_goal)
        #                 or total_step > break_step
        #                 or os.path.exists(f'{cwd}/stop'))

        if pipe_eva[1].poll():
            # pipe_eva[0].send(if_train)
            if_train = pipe_eva[1].recv()

    pipe_list = list()
    pipe_list.extend(pipe_eva)
    for pipe_exp in pipe_exp_list:
        pipe_list.extend(pipe_exp)
    for pipe in pipe_list:
        try:
            while pipe.poll():
                pipe.recv()
        except EOFError:
            pass


def mp_evaluator(args, pipe_eva, learner_id=0):
    args.init_before_training(process_id=-1)

    if True:
        '''arguments: basic'''
        cwd = args.cwd
        env = args.env
        agent = args.agent
        gpu_id = args.gpu_id
        # worker_num = args.worker_num

        '''arguments: train'''
        # net_dim = args.net_dim
        # max_memo = args.max_memo
        break_step = args.break_step
        # batch_size = args.batch_size
        # target_step = args.target_step
        # repeat_times = args.repeat_times
        # learning_rate = args.learning_rate
        if_break_early = args.if_allow_break

        # gamma = args.gamma
        # reward_scale = args.reward_scale
        # if_per_or_gae = args.if_per_or_gae
        # soft_update_tau = args.soft_update_tau

        '''arguments: evaluate'''
        show_gap = args.eval_gap
        eval_times1 = args.eval_times1
        eval_times2 = args.eval_times2
        env_eval = deepcopy_or_rebuild_env(env) if args.env_eval is None else args.env_eval

        '''arguments: environment'''
        # max_step = env.max_step
        # state_dim = env.state_dim
        # action_dim = env.action_dim
        # if_discrete = env.if_discrete
        del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: Evaluator'''
    learner_num = 1 if isinstance(gpu_id, int) else len(gpu_id)
    gpu_id = gpu_id if isinstance(gpu_id, int) else gpu_id[learner_id]
    evaluator = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval,
                          eval_times1=eval_times1, eval_times2=eval_times2, eval_gap=show_gap)  # build Evaluator
    evaluator.eval_time += learner_id * (show_gap / learner_num)

    # pipe_eva[1].send((act_cpu, steps))
    act_cpu, steps = pipe_eva[0].recv()

    '''start training'''
    sum_step = steps
    if_train = True
    while if_train:
        # pipe_eva[1].send((act_state_dict, steps, logging_tuple))
        act_state_dict, steps, logging_tuple = pipe_eva[0].recv()

        sum_step += steps
        if act_state_dict is not None:
            act_cpu.load_state_dict(act_state_dict)

            if_reach_goal = evaluator.evaluate_save(act_cpu, sum_step, logging_tuple)
            sum_step = 0

            if_train = not ((if_break_early and if_reach_goal)
                            or evaluator.total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))

    print(f'| SavedDir: {cwd}\n'
          f'| UsedTime: {time.time() - evaluator.start_time:.0f}')
    pipe_eva[0].send(if_train)
    # if_train = pipe_eva[1].recv()


# def train_and_evaluate_mp(args):
#     import multiprocessing as mp  # Python built-in multiprocessing library
#
#     pipe_eva_list = list()
#     pipe_net_list = list()
#     for i in range(len(args.gpu_ids)):
#         pipe_eva_list.append(mp.Pipe())
#         pipe_net_list.append(mp.Pipe())
#
#     process_net_list = list()
#     process_env_list = list()
#     for i in range(len(args.gpu_ids)):
#         pipe_env_list = list()
#         for _ in range(args.rollout_num):
#             pipe0_env, pipe1_env = mp.Pipe()
#             pipe_env_list.append((pipe0_env, pipe1_env))
#             process_env_list.append(mp.Process(target=mp_env, args=(args, pipe0_env, i)))
#
#         pipe1_eva = pipe_eva_list[i][1]
#         process_net_list.append(mp.Process(target=mp_net, args=(args, pipe_net_list, pipe1_eva, pipe_env_list, i)))
#
#     pipe0_eva_list = [pipe_eva[0] for pipe_eva in pipe_eva_list]
#     process_eva = mp.Process(target=mp_eva, args=(args, pipe0_eva_list))
#
#     '''start'''
#     process_eva.start()
#     [p.start() for p in process_net_list]
#     [p.start() for p in process_env_list]
#
#     process_eva.join()
#     [p.join() for p in process_net_list]
#     [p.join() for p in process_env_list]
#
#     process_eva.terminate()
#     [p.terminate() for p in process_net_list]
#     [p.terminate() for p in process_env_list]
#
#
# class PipeNetComm:
#     def __init__(self, pipe_net_list, i, net_params, device):
#         self.learn_num = len(pipe_net_list)
#
#         self.pipe0_net = pipe_net_list[i][0]
#         self.pipe1_net_list = [pipe_net[1] for pipe_net in pipe_net_list]
#         self.i = i
#         self.i1 = (self.i + 1) % self.learn_num
#         self.net_params_gpu = net_params  # list(net.parameters())
#         self.pub_params_gpu = None  # public network
#         self.dev_gpu = device
#         self.dev_cpu = torch.device('cpu')
#
#         if self.learn_num == 1:
#             self.comm = self.comm_len1
#         elif self.learn_num == 2:
#             self.comm = self.comm_len2
#         else:
#             if i % 2 == 0:
#                 self.i0, self.i1 = 0, 1
#                 self.i0s = [gpu_id for gpu_id in range(2, self.learn_num, 2)]
#             else:
#                 self.i0, self.i1 = 1, 0
#                 self.i0s = [gpu_id for gpu_id in range(3, self.learn_num, 2)]
#             self.comm = self.comm_len3
#
#     def comm_init(self):
#         main_i = 0
#         if self.i == main_i:
#             self.pub_params_gpu = deepcopy(self.net_params_gpu)
#             pub_params_cpu = [params.data.to(self.dev_cpu) for params in self.pub_params_gpu]
#             for i1 in range(1, self.learn_num):
#                 self.pipe1_net_list[i1].send(pub_params_cpu)
#             time.sleep(1)
#         else:
#             pub_params_cpu = self.pipe0_net.recv()
#             self.pub_params_gpu = [params.data.to(self.dev_gpu) for params in pub_params_cpu]
#             self.net_params_assign(self.net_params_gpu, self.pub_params_gpu)
#
#     def comm_len1(self):
#         pass
#
#     def comm_len2(self):
#         net_params_cpu = [params.data.to(self.dev_cpu) for params in self.net_params_gpu]
#         self.pipe1_net_list[self.i1].send(net_params_cpu)
#         net_params_cpu = self.pipe0_net.recv()
#         net_params_gpu = [params.data.to(self.dev_gpu) for params in net_params_cpu]
#         self.net_params_soft_update(self.net_params_gpu, net_params_gpu, 0.5)
#
#     def comm_len3(self):
#         if self.i == self.i0:
#             for i0 in self.i0s:
#                 net_params_cpu = self.pipe1_net_list[i0].recv()
#                 net_params_gpu = [params.data.to(self.dev_gpu) for params in net_params_cpu]
#                 self.net_params_add(self.net_params_gpu, net_params_gpu)
#
#             net_params_cpu = [params.data.to(self.dev_cpu) for params in self.net_params_gpu]
#             self.pipe0_net.send(net_params_cpu)
#             net_params_cpu = self.pipe1_net_list[self.i1].recv()
#             net_params_gpu = [params.data.to(self.dev_gpu) for params in net_params_cpu]
#             self.net_params_add(self.net_params_gpu, net_params_gpu)
#             self.net_params_mul(self.net_params_gpu, 1 / self.learn_num)
#
#             net_params_cpu = [params.data.to(self.dev_cpu) for params in self.net_params_gpu]
#             for i0 in self.i0s:
#                 self.pipe1_net_list[i0].send(net_params_cpu)
#
#         else:
#             net_params_cpu = [params.data.to(self.dev_cpu) for params in self.net_params_gpu]
#             self.pipe0_net.send(net_params_cpu)
#
#             net_params_cpu = self.pipe0_net.recv()
#             self.net_params_assign(self.net_params_gpu, net_params_cpu)
#
#     def comm_next(self):
#         net_params_cpu = [params.data.to(self.dev_cpu) for params in self.net_params_gpu]
#         self.pipe0_net.send(net_params_cpu)
#
#         net_params_cpu = self.pipe1_net_list[self.i1].recv()
#         net_params_gpu = [params.data.to(self.dev_gpu) for params in net_params_cpu]
#         self.net_params_soft_update(self.net_params_gpu, net_params_gpu, 1 - 1 / self.learn_num)
#
#     def comm_buffer(self, array_list):
#         self.pipe0_net.send(array_list)
#         array_list = self.pipe1_net_list[self.i1].recv()
#         return array_list
#
#     @staticmethod
#     def net_params_assign(target_params, current_params):
#         for tar, cur in zip(target_params, current_params):
#             tar.data.copy_(cur.data)
#
#     @staticmethod
#     def net_params_soft_update(target_params, current_params, tau):
#         for tar, cur in zip(target_params, current_params):
#             tar.data.copy_(tar.data * (1 - tau) + cur.data * tau)
#
#     @staticmethod
#     def net_params_add(target_params, current_params):
#         for tar, cur in zip(target_params, current_params):
#             tar.data.copy_(tar.data + cur.data)
#
#     @staticmethod
#     def net_params_mul(target_params, mul):
#         for tar in target_params:
#             tar.data.copy_(tar.data * mul)
#
#
# def mp_env(args, pipe0_env, i):
#     args.init_before_training(if_main=False)
#
#     '''basic arguments'''
#     # cwd = args.cwd
#     env = args.env
#     agent = args.agent
#
#     '''training arguments'''
#     gpu_ids = args.gpu_ids  # plan to make gpu_id gpu_ids elegant
#     gpu_id = gpu_ids[i]
#     net_dim = args.net_dim
#     # max_memo = args.max_memo
#     # break_step = args.break_step
#     # batch_size = args.batch_size
#     target_step = args.target_step
#     # repeat_times = args.repeat_times
#     # if_break_early = args.if_allow_break
#     if_per = args.if_per
#     gamma = args.gamma
#     reward_scale = args.reward_scale
#     rollout_num = args.rollout_num
#
#     '''evaluating arguments'''
#     # eval_gap = args.eval_gap
#     # eval_times1 = args.eval_times1
#     # eval_times2 = args.eval_times2
#     # env_eval = rebuild_or_deepcopy_env(env) if args.env_eval is None else args.env_eval
#     del args  # In order to show these hyper-parameters clearly, I put them above.
#
#     '''init: env'''
#     # max_step = env.max_step
#     state_dim = env.state_dim
#     action_dim = env.action_dim
#     # if_discrete = env.if_discrete
#
#     '''init: Agent'''
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
#     agent.init(net_dim, state_dim, action_dim, if_per)
#     if_on_policy = getattr(agent, 'if_on_policy', False)
#
#     roll_target_step = target_step // rollout_num
#     del target_step
#
#     '''prepare for training'''
#     env.last_state = env.reset()
#     if not if_on_policy:  # explore_before_training for off-policy
#         with torch.no_grad():  # update replay buffer
#             trajectory_list, state = explore_before_training(env, roll_target_step, reward_scale, gamma)
#             # plan to send state
#             step = len(trajectory_list)
#
#         state_ary = torch.as_tensor([item[0] for item in trajectory_list], dtype=torch.float16)
#         other_ary = torch.as_tensor([item[1] for item in trajectory_list], dtype=torch.float16)
#         pipe0_env.send((step, state_ary, other_ary))
#         # step, state_ary, other_ary = pipe1_env.recv()
#
#     # pipe1_env.send(env_tuple)
#     env_tuple = pipe0_env.recv()
#     if_train, act_net_cpu = env_tuple
#     while if_train:
#         agent.get_action = act_net_cpu.to(agent.device)
#
#         trajectory_list = agent.explore_env(env, roll_target_step, reward_scale, gamma)
#         step = len(trajectory_list)
#
#         state_ary = torch.as_tensor([item[0] for item in trajectory_list], dtype=torch.float16)
#         other_ary = torch.as_tensor([item[1] for item in trajectory_list], dtype=torch.float16)
#         pipe0_env.send((step, state_ary, other_ary))
#         # step, state_ary, other_ary = pipe1_env.recv()
#
#         # pipe1_env.send(env_tuple)
#         env_tuple = pipe0_env.recv()
#         if_train, act_net_cpu = env_tuple
#
#
# def mp_net(args, pipe_net_list, pipe1_eva, pipe_env_list, agent_id):
#     args.init_before_training(if_main=False)
#
#     '''basic arguments'''
#     # cwd = args.cwd
#     env = args.env
#     agent = args.agent
#
#     '''training arguments'''
#     gpu_ids = args.gpu_ids
#     gpu_id = gpu_ids[agent_id]
#     net_dim = args.net_dim
#     max_memo = args.max_memo
#     # break_step = args.break_step
#     batch_size = args.batch_size
#     # target_step = args.target_step
#     repeat_times = args.repeat_times
#     # if_break_early = args.if_allow_break
#     if_per = args.if_per
#     # gamma = args.gamma
#     # reward_scale = args.reward_scale
#     rollout_num = args.rollout_num
#
#     '''evaluating arguments'''
#     # eval_gap = args.eval_gap
#     # eval_times1 = args.eval_times1
#     # eval_times2 = args.eval_times2
#     # env_eval = rebuild_or_deepcopy_env(env) if args.env_eval is None else args.env_eval
#     del args  # In order to show these hyper-parameters clearly, I put them above.
#
#     '''init: environment'''
#     max_step = env.max_step
#     state_dim = env.state_dim
#     action_dim = env.action_dim
#     if_discrete = env.if_discrete
#     del env
#
#     '''init: Agent, ReplayBuffer, Evaluator'''
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
#     device_cpu = torch.device('cpu')
#     agent.init(net_dim, state_dim, action_dim, if_per)
#     if_on_policy = getattr(agent, 'if_on_policy', False)
#
#     learner_num = len(pipe_net_list)
#     net_params = list(agent.cri.parameters()) + list(agent.get_action.parameters())
#
#     pipe_comm = PipeNetComm(pipe_net_list, agent_id, net_params, agent.device)
#     pipe_comm.comm_init()  # send net0 to net_other
#
#     max_len = max_memo // learner_num
#     if if_on_policy:
#         max_len += max_step
#
#     if_send_buf = False
#     buf_rollout_num = rollout_num * learner_num if if_send_buf else rollout_num
#     buffer = ReplayBufferMP(max_len=max_len, max_episode_step=max_step,
#                             state_dim=state_dim, action_dim=1 if if_discrete else action_dim,
#                             if_on_policy=if_on_policy, if_per=if_per, rollout_num=buf_rollout_num)
#
#     # evaluator = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval,
#     #                       eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2, )
#
#     '''prepare for training'''
#     # env.state = env.reset()
#     if if_on_policy:
#         step = 0
#     else:  # explore_before_training for off-policy
#         step = 0
#
#         for env_id, pipe_env in enumerate(pipe_env_list):
#             pipe1_env = pipe_env[1]
#
#             # pipe0_env.send((step, state_ary, other_ary))
#             _step, state_ary, other_ary = pipe1_env.recv()
#             step += _step
#             buffer.extend_buffer(state_ary, other_ary, env_id)
#
#         pipe_comm.comm()
#
#         agent.update_net(buffer, batch_size, repeat_times)
#         # pre-training and hard update
#         agent.act_target.load_state_dict(agent.get_action.state_dict()) \
#             if getattr(agent, 'act_target', None) else None
#         agent.cri_target.load_state_dict(agent.cri.state_dict()) if getattr(agent, 'cri_target', None) else None
#
#     pipe1_eva.send(step)
#     # step = pipe0_eva.recv()
#     # total_step = step
#
#     '''start training'''
#     counter_eva = 0
#     if_train = True
#     while if_train:
#         act_net_env = deepcopy(agent.get_action).to(device_cpu)
#         env_tuple = (if_train, act_net_env)
#         for pipe_env in pipe_env_list:
#             pipe1_env = pipe_env[1]
#             pipe1_env.send(env_tuple)
#             # env_tuple = pipe0_env.recv()
#
#         if if_on_policy:
#             buffer.empty_buffer()
#
#         step = 0
#         for env_id, pipe_env in enumerate(pipe_env_list):
#             pipe1_env = pipe_env[1]
#
#             # pipe0_env.send((step, state_ary, other_ary))
#             _step, state_ary, other_ary = pipe1_env.recv()
#             step += _step
#             buffer.extend_buffer(state_ary, other_ary, env_id)
#         # total_step += step
#
#         obj_a, obj_c = agent.update_net(buffer, batch_size, repeat_times)
#         act_net_cpu = deepcopy(agent.get_action).to(device_cpu)
#         pipe_comm.comm()
#
#         # if_reach_goal = evaluator.evaluate_save(agent.act, step, obj_a, obj_c)
#         # evaluator.draw_plot()
#
#         counter_eva = (counter_eva + 1) % learner_num
#         eva_tuple = (step, obj_a, obj_c, act_net_cpu, agent_id) if counter_eva == agent_id else (step,)
#         pipe1_eva.send(eva_tuple)
#         # eva_tuple = pipe0_eva.recv()
#
#         while pipe1_eva.poll() and if_train:
#             # pipe0_eva.send(if_train)
#             if_train = pipe1_eva.recv()
#
#     env_tuple = (if_train, None)
#     for pipe_env in pipe_env_list:
#         pipe1_env = pipe_env[1]
#
#         pipe1_env.send(env_tuple)
#         # env_tuple = pipe0_env.recv()
#     for pipe_env in pipe_env_list:
#         pipe0_env = pipe_env[0]
#         while pipe0_env.poll():
#             time.sleep(1)
#
#
# def mp_eva(args, pipe0_eva_list):
#     args.init_before_training(if_main=True)
#
#     '''basic arguments'''
#     cwd = args.cwd
#     env = args.env
#     agent = args.agent
#
#     '''training arguments'''
#     net_dim = args.net_dim
#     break_step = args.break_step
#     if_break_early = args.if_allow_break
#     if_per = args.if_per
#
#     '''evaluating arguments'''
#     eval_gap = args.eval_gap
#     eval_times1 = args.eval_times1
#     eval_times2 = args.eval_times2
#     env_eval = deepcopy_or_rebuild_env(env) if args.env_eval is None else args.env_eval
#     del args  # In order to show these hyper-parameters clearly, I put them above.
#
#     '''init: environment'''
#     # max_step = env.max_step
#     state_dim = env.state_dim
#     action_dim = env.action_dim
#     # if_discrete = env.if_discrete
#
#     '''init: Agent, Evaluator'''
#     agent.init(net_dim, state_dim, action_dim, if_per)
#
#     evaluator = Evaluator(cwd=cwd, agent_id=0, device=torch.device('cpu'), env=env_eval,
#                           eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2, )
#
#     '''prepare for training'''
#     total_step = 0
#     for pipe0_eva in pipe0_eva_list:
#         # pipe1_eva.send(step)
#         step = pipe0_eva.recv()
#         total_step += step
#
#     '''start training'''
#     if_train = True
#     while if_train:
#         if_reach_goal_list = list()
#         for pipe0_eva in pipe0_eva_list:
#             # pipe1_eva.send(eva_tuple)
#
#             eva_tuples = None
#
#             steps = 0
#             while not pipe0_eva.poll():  # wait until pipe2_eva not empty
#                 time.sleep(1)
#             while pipe0_eva.poll():  # receive the latest object from pipe
#                 eva_tuple = pipe0_eva.recv()
#                 step = eva_tuple[0]
#                 if len(eva_tuple) == 1:
#                     steps += step
#                 else:
#                     eva_tuples = eva_tuple
#                     steps += step
#             total_step += steps
#
#             if eva_tuples is not None:
#                 step, obj_a, obj_c, act_net, i = eva_tuples
#
#                 agent.get_action = act_net
#                 evaluator.agent_id = i
#                 # plan to change to logging_tuple
#                 if_reach_goal_item = evaluator.evaluate_save(agent.get_action, steps, obj_a, obj_c)
#                 if_reach_goal_list.append(if_reach_goal_item)
#
#         if_reach_goal = any(if_reach_goal_list)
#         if_train = not ((if_break_early and if_reach_goal)
#                         or total_step > break_step
#                         or os.path.exists(f'{cwd}/stop'))
#
#         for pipe0_eva in pipe0_eva_list:
#             pipe0_eva.send(if_train)
#             # if_train = pipe1_eva.recv()
#
#     print(f'| SavedDir: {cwd}\n| UsedTime: {time.time() - evaluator.start_time:.0f}')


'''utils'''


class Evaluator:
    def __init__(self, cwd, agent_id, eval_times1, eval_times2, eval_gap, env, device):
        self.recorder = list()  # total_step, r_avg, r_std, obj_c, ...
        self.r_max = -np.inf
        self.total_step = 0

        self.cwd = cwd  # constant
        self.device = device
        self.agent_id = agent_id
        self.eval_gap = eval_gap
        self.eval_times1 = eval_times1
        self.eval_times2 = eval_times2
        self.env = env
        self.target_return = env.target_return

        self.used_time = None
        self.start_time = time.time()
        self.eval_time = time.time() - self.eval_gap
        print(f"{'ID':>2} {'Step':>8} {'MaxR':>8} |"
              f"{'avgR':>8} {'stdR':>8} |{'avgS':>5} {'stdS':>4} |"
              f"{'objC':>8} {'etc.':>8}")

    def evaluate_save(self, act, steps, log_tuple) -> bool:
        self.total_step += steps  # update total training steps

        if time.time() - self.eval_time > self.eval_gap:
            self.eval_time = time.time()

            rewards_steps_list = [get_episode_return_and_step(self.env, act, self.device) for _ in
                                  range(self.eval_times1)]
            r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)

            if r_avg > self.r_max:  # evaluate actor twice to save CPU Usage and keep precision
                rewards_steps_list += [get_episode_return_and_step(self.env, act, self.device)
                                       for _ in range(self.eval_times2 - self.eval_times1)]
                r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)
            if r_avg > self.r_max:  # save checkpoint with highest episode return
                self.r_max = r_avg  # update max reward (episode return)

                act_save_path = f'{self.cwd}/actor.pth'
                torch.save(act.state_dict(), act_save_path)  # save policy network in *.pth
                print(f"{self.agent_id:<2} {self.total_step:8.2e} {self.r_max:8.2f} |")  # save policy and print

            self.recorder.append((self.total_step, r_avg, r_std, *log_tuple))  # update recorder

            if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
            if if_reach_goal and self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(f"{'ID':>2} {'Step':>8} {'TargetR':>8} |{'avgR':>8} {'stdR':>8} |"
                      f"  {'UsedTime':>8}  ########\n"
                      f"{self.agent_id:<2} {self.total_step:8.2e} {self.target_return:8.2f} |"
                      f"{r_avg:8.2f} {r_std:8.2f} |"
                      f"  {self.used_time:>8}  ########")

            print(f"{self.agent_id:<2} {self.total_step:8.2e} {self.r_max:8.2f} |"
                  f"{r_avg:8.2f} {r_std:8.2f} |{s_avg:5.0f} {s_std:4.0f} |"
                  f"{' '.join(f'{n:8.2f}' for n in log_tuple)}")

            self.draw_plot()
        else:
            if_reach_goal = False
        return if_reach_goal

    def draw_plot(self):
        if len(self.recorder) == 0:
            print("| save_npy_draw_plot() WARNNING: len(self.recorder)==0")
            return None

        '''convert to array and save as npy'''
        np.save('%s/recorder.npy' % self.cwd, self.recorder)

        '''draw plot and save as png'''
        train_time = int(time.time() - self.start_time)
        total_step = int(self.recorder[-1][0])
        save_title = f"step_time_maxR_{int(total_step)}_{int(train_time)}_{self.r_max:.3f}"

        save_learning_curve(self.recorder, self.cwd, save_title)

    @staticmethod
    def get_r_avg_std_s_avg_std(rewards_steps_list):
        rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float)
        r_avg, s_avg = rewards_steps_ary.mean(axis=0)  # average of episode return and episode step
        r_std, s_std = rewards_steps_ary.std(axis=0)  # standard dev. of episode return and episode step
        return r_avg, r_std, s_avg, s_std


def get_episode_return_and_step(env, act, device) -> (float, int):
    episode_return = 0.0  # sum of rewards in an episode
    episode_step = 1
    max_step = env.max_step
    if_discrete = env.if_discrete

    state = env.reset()
    for episode_step in range(max_step):
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break
    episode_return = getattr(env, 'episode_return', episode_return)
    return episode_return, episode_step


def save_learning_curve(recorder, cwd='.', save_title='learning curve', fig_name='plot_learning_curve.jpg'):
    recorder = np.array(recorder)  # recorder_ary.append((self.total_step, r_avg, r_std, obj_a, obj_c))
    steps = recorder[:, 0]  # x-axis is training steps
    r_avg = recorder[:, 1]
    r_std = recorder[:, 2]
    obj_c = recorder[:, 3]
    obj_a = recorder[:, 4]

    '''plot subplots'''
    import matplotlib as mpl
    mpl.use('Agg')
    """Generating matplotlib graphs without a running X server [duplicate]
    write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
    https://stackoverflow.com/a/4935945/9293137
    """
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)

    axs0 = axs[0]
    axs0.cla()
    color0 = 'lightcoral'
    axs0.set_xlabel('Total Steps')
    axs0.set_ylabel('Episode Return')
    axs0.plot(steps, r_avg, label='Episode Return', color=color0)
    axs0.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)

    ax11 = axs[1]
    ax11.cla()
    color11 = 'royalblue'
    axs0.set_xlabel('Total Steps')
    ax11.set_ylabel('objA', color=color11)
    ax11.plot(steps, obj_a, label='objA', color=color11)
    ax11.tick_params(axis='y', labelcolor=color11)
    for plot_i in range(5, recorder.shape[1]):
        other = recorder[:, plot_i]
        ax11.plot(steps, other, label=f'{plot_i}', color='grey')

    ax12 = axs[1].twinx()
    color12 = 'darkcyan'
    ax12.set_ylabel('objC', color=color12)
    ax12.fill_between(steps, obj_c, facecolor=color12, alpha=0.2, )
    ax12.tick_params(axis='y', labelcolor=color12)

    '''plot save'''
    plt.title(save_title, y=2.3)
    plt.savefig(f"{cwd}/{fig_name}")
    plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
    # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()


def explore_before_training(env, target_step, reward_scale, gamma) -> (list, np.ndarray):  # for off-policy only
    trajectory_list = list()

    if_discrete = env.if_discrete
    action_dim = env.action_dim

    state = env.reset()
    for _ in range(target_step):
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
    return trajectory_list, state
