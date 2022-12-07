import gym
import torch
import multiprocessing
from torch import Tensor
from multiprocessing import Process, Pipe


def kwargs_filter(function, kwargs: dict) -> dict:
    import inspect
    sign = inspect.signature(function).parameters.values()
    sign = {val.name for val in sign}
    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}  # filtered kwargs


class SubEnv(Process):
    def __init__(self, pipe: Pipe, env_class, env_args: dict, device: torch.device):
        super().__init__()
        self.pipe = pipe
        self.env_class = env_class
        self.env_args = env_args
        self.device = device

    def run(self):
        torch.set_grad_enabled(False)

        # from elegantrl.train.config import build_vec_env
        # env = build_vec_env(env_class=self.env_class, env_args=self.env_args)
        from elegantrl.train.config import kwargs_filter
        env = self.env_class(**kwargs_filter(self.env_class.__init__, self.env_args.copy()))

        state = env.reset()
        state = torch.as_tensor(state, device=self.device)
        self.pipe.send(state)

        while True:
            action = self.pipe.recv()
            state, reward, done, info_dict = env.step(action)
            if done:
                state = env.reset()
            state = torch.as_tensor(state, device=self.device)
            self.pipe.send((state, reward, done, info_dict))


class VecEnv:
    def __init__(self, env_class: object, env_args: dict, gpu_id: int = -1):
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        '''the necessary env information when you design a custom env'''
        self.env_name = env_args['env_name']  # the name of this env.
        self.num_envs = env_args['num_envs']  # the number of sub env in vectorized env.
        self.max_step = env_args['max_step']  # the max step number in an episode for evaluation
        self.state_dim = env_args['state_dim']  # feature number of state
        self.action_dim = env_args['action_dim']  # feature number of action
        self.if_discrete = env_args['if_discrete']  # discrete action or continuous action

        '''speed up with multiprocessing: Process, Pipe'''
        pipes = [Pipe() for _ in range(self.num_envs)]
        self.pipes, pipes = list(map(list, zip(*pipes)))
        self.sub_envs = [SubEnv(pipe=pipe, env_class=env_class, env_args=env_args, device=self.device)
                         for pipe in pipes]
        [p.start() for p in self.sub_envs]

    def reset(self) -> Tensor:  # reset the agent in env
        states = [p.recv() for p in self.pipes]
        states = torch.stack(states).to(self.device)
        return states

    def step(self, action: Tensor) -> (Tensor, Tensor, Tensor, (dict,)):  # agent interacts in env
        [p.send(a) for p, a in zip(self.pipes, action)]

        state_reward_done_info_dict = [p.recv() for p in self.pipes]
        states, rewards, dones, info_dicts = list(map(list, zip(*state_reward_done_info_dict)))

        states = torch.stack(states).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        return states, rewards, dones, info_dicts

    def close(self):
        [p.terminate() for p in self.sub_envs]


def demo__vec_env_with_gym_env():
    print('Use Process to run Process')
    num_envs = 4
    env_name = 'BipedalWalker-v3'

    env_class = gym.make
    env_args = {'env_name': env_name,
                'num_envs': num_envs,
                'max_step': 1600,
                'state_dim': 24,
                'action_dim': 4,
                'if_discrete': False}

    env = VecEnv(env_class=env_class, env_args=env_args)
    device = env.device
    action_dim = env_args['action_dim']

    states = env.reset()
    assert isinstance(states, Tensor)
    for t in range(8):
        actions = torch.rand((num_envs, action_dim), dtype=torch.float32, device=device)
        states, rewards, dones, _ = env.step(actions)
        print(';;', rewards)
    env.close()


def demo__vec_env_with_custom_env():
    print('Use Process to run Process')
    num_envs = 4

    from elegantrl.envs.CustomGymEnv import PendulumEnv
    env_class = PendulumEnv
    env_args = {'env_name': 'CustomEnv-v0',
                'num_envs': num_envs,
                'max_step': 200,
                'state_dim': 3,
                'action_dim': 1,
                'if_discrete': False}

    env = VecEnv(env_class=env_class, env_args=env_args)
    device = env.device
    action_dim = env_args['action_dim']

    states = env.reset()
    assert isinstance(states, Tensor)
    for t in range(8):
        actions = torch.rand((num_envs, action_dim), dtype=torch.float32, device=device)
        states, rewards, dones, _ = env.step(actions)
        print(';;', rewards)
    env.close()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    # demo__vec_env_with_gym_env()
    demo__vec_env_with_custom_env()
