from elegantrl2.env import *
from elegantrl2.run import *


def demo_continuous_action_on_policy_temp_mg():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    from elegantrl2.agent import AgentPPO
    args.agent = AgentPPO()
    args.agent.cri_target = True
    args.learning_rate = 2 ** -14
    args.random_seed = 1943
    args.gpu_id = (0, 1, 2, 3)

    '''choose environment'''
    if_train_pendulum = 0
    if if_train_pendulum:
        "TotalStep: 4e5, TargetReward: -200, UsedTime: 400s"
        env = gym.make('Pendulum-v0')
        env.target_return = -200  # set target_reward manually for env 'Pendulum-v0'
        args.env = PreprocessEnv(env=env)
        args.reward_scale = 2 ** -3  # RewardRange: -1800 < -200 < -50 < 0
        args.net_dim = 2 ** 7
        args.batch_size = args.net_dim * 2
        args.target_step = args.env.max_step * 16

    if_train_bipedal_walker = 1
    if if_train_bipedal_walker:
        "TotalStep: 8e5, TargetReward: 300, UsedTime: 1800s"
        env_name = 'BipedalWalker-v3'
        args.env = PreprocessEnv(env=env_name)
        # args.env = PreprocessVecEnv(env=env_name, env_num=2)
        # args.env_eval = PreprocessEnv(env=env_name)
        args.reward_scale = 2 ** 0  # RewardRange: -200 < -150 < 300 < 334
        args.gamma = 0.97
        args.target_step = args.env.max_step * 4
        args.repeat_times = 2 ** 4
        args.if_per_or_gae = True
        args.agent.lambda_entropy = 0.05
        args.break_step = int(8e6)

    '''train and evaluate'''
    # train_and_evaluate(args)
    args.worker_num = 2
    train_and_evaluate_mg(args)


if __name__ == '__main__':
    demo_continuous_action_on_policy_temp_mg()
