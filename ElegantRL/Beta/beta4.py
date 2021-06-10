import time
import torch

from elegantrl2.env import PreprocessEnv, PreprocessVecEnv

from elegantrl2.run import *
import gym

gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'


def get_optim_parameters(optim):
    params_list = list()
    for params_dict in optim.state_dict()['state'].values():
        params_list.extend([t for t in params_dict.values()
                            if isinstance(t, torch.Tensor)])
    return params_list


def avg_optim(dst_optim, src_optim, device):
    for dst, src in zip(get_optim_parameters(dst_optim),
                        get_optim_parameters(src_optim)):
        dst.data.copy_((dst.data + src.data.to(device)) * 0.5)
        # dst.data.copy_(src.data * tau + dst.data * (1 - tau))


def avg_net(dst_net, src_net, device):
    for tar, cur in zip(dst_net.parameters(), src_net.parameters()):
        tar.data.copy_((tar.data + cur.data.to(device)) * 0.5)


class LearnerComm:
    def __init__(self, pipe_net_list, learner_id):
        pipe_num = len(pipe_net_list)

        if pipe_num == 2:
            if learner_id == 0:
                self.pipe0 = pipe_net_list[0]
                self.pipe1 = pipe_net_list[1]
            else:  # if learner_id == 1:
                self.pipe0 = pipe_net_list[1]
                self.pipe1 = pipe_net_list[0]
        else:  # if pipe_num == 4:
            if learner_id == 0:
                self.pipe0 = pipe_net_list[learner_id]
                self.pipe1 = pipe_net_list[1], pipe_net_list[2]
            elif learner_id == 1:
                self.pipe0 = pipe_net_list[learner_id]
                self.pipe1 = pipe_net_list[0], pipe_net_list[3]
            elif learner_id == 2:
                self.pipe0 = pipe_net_list[learner_id]
                self.pipe1 = pipe_net_list[3],pipe_net_list[1]
            else:  # if learner_id == 3:
                self.pipe0 = pipe_net_list[learner_id]
                self.pipe1 = pipe_net_list[2],pipe_net_list[1]

    def comm(self, data, round_id):
        self.pipe1[round_id][0].send(data)
        return self.pipe0[1].recv()


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

    '''init: Comm'''
    comm = LearnerComm(pipe_net_list, learner_id)

    '''init: Agent'''
    agent.init(net_dim, state_dim, action_dim, learning_rate, if_per_or_gae, learner_id)
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

        for round_id in range(2):
            data = (agent.act, agent.cri, agent.act_optim, agent.cri_optim)
            data = comm.comm(data, round_id)
            comm_act, comm_cri, comm_act_optim, comm_cri_optim = data

            avg_net(agent.act, comm_act, agent.device)
            avg_net(agent.cri, comm_cri, agent.device)
            avg_optim(agent.act_optim, comm_act_optim, agent.device)
            avg_optim(agent.cri_optim, comm_cri_optim, agent.device)

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


def train_and_evaluate_mg(args):  # multiple GPU
    import multiprocessing as mp  # Python built-in multiprocessing library
    process = list()

    pipe_net_list = [mp.Pipe() for _ in args.gpu_id]

    for learner_id in range(len(args.gpu_id)):
        pipe_eva = mp.Pipe()
        process.append(mp.Process(target=mp_evaluator, args=(args, pipe_eva, learner_id)))

        pipe_exp_list = list()
        for worker_id in range(args.worker_num):
            pipe_exp = mp.Pipe()
            pipe_exp_list.append(pipe_exp)
            process.append(mp.Process(target=mp_worker, args=(args, pipe_exp, worker_id, learner_id)))
        process.append(mp.Process(target=mg_learner, args=(
            args, pipe_exp_list, pipe_eva, pipe_net_list, learner_id)))

    [p.start() for p in process]
    [p.join() for p in (process[-1],)]  # wait
    [p.terminate() for p in process]


def demo_continuous_action_on_policy():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    from elegantrl2.agent import AgentPPO
    args.agent = AgentPPO()
    args.agent.cri_target = True
    args.learning_rate = 2 ** -14
    args.random_seed = 1943
    args.gpu_id = (0, 1, 2, 3)

    '''choose environment'''
    if_train_pendulum = 1
    if if_train_pendulum:
        "TotalStep: 4e5, TargetReward: -200, UsedTime: 400s"
        env = gym.make('Pendulum-v0')
        env.target_return = -200  # set target_reward manually for env 'Pendulum-v0'
        args.env = PreprocessEnv(env=env)
        args.reward_scale = 2 ** -3  # RewardRange: -1800 < -200 < -50 < 0
        args.net_dim = 2 ** 7
        args.batch_size = args.net_dim * 2
        args.target_step = args.env.max_step * 16

    '''train and evaluate'''
    # train_and_evaluate(args)
    args.eval_gap /= len(args.gpu_id)
    args.worker_num = 2
    train_and_evaluate_mg(args)


if __name__ == '__main__':
    demo_continuous_action_on_policy()
