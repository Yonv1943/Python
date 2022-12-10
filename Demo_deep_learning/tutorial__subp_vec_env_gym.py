import gym
import torch
import numpy as np
import multiprocessing
from typing import List
from torch import Tensor
from multiprocessing import Process, Pipe, Queue


def kwargs_filter(function, kwargs: dict) -> dict:
    import inspect
    sign = inspect.signature(function).parameters.values()
    sign = {val.name for val in sign}
    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}  # filtered kwargs


def build_vec_env(env_class, env_args: dict, gpu_id: int = -1):
    env_args['gpu_id'] = gpu_id  # set gpu_id for vectorized env before build it
    env_args.setdefault('num_envs', 1)
    env_args.setdefault('max_step', 12345)

    if_gym_env = env_class.__module__ == 'gym.envs.registration'  # is standard OpenAI Gym env
    if_vec_env = env_args['num_envs'] > 1  # is vectorized env

    if if_gym_env:  # is standard OpenAI Gym env
        import gym
        assert '0.18.0' <= gym.__version__ <= '0.25.2'  # pip3 install gym==0.24.0
        gym.logger.set_level(40)  # Block warning
        env_args['id'] = env_args['env_name']  # OpenAI gym build env by `gym.make(id=env_name)`
        # env = env_class(id=env_args['env_name'])

    if if_vec_env:
        env = VecEnvAsync(env_class=env_class, env_args=env_args)
        # env = VecEnvSync(env_class=env_class, env_args=env_args)
        '''The following code inside VecEnv.__init__() -> SubEnv.run()
        if self.env_class.__module__ == 'gym.envs.registration':  # is standard OpenAI Gym env
            env = self.env_class(id=self.env_args['env_name'])
        else:
            env = self.env_class(**kwargs_filter(self.env_class.__init__, self.env_args.copy()))

        '''
    elif if_gym_env:
        env = env_class(id=env_args['env_name'])
    else:  # if not if_gym_env
        env = env_class(**kwargs_filter(env_class.__init__, env_args.copy()))

    for attr_str in ('env_name', 'num_envs', 'max_step', 'state_dim', 'action_dim', 'if_discrete'):
        setattr(env, attr_str, env_args[attr_str])
    return env


class SubEnvSync(Process):
    def __init__(self, pipe: Pipe, env_class, env_args: dict, env_id: int = 0):
        super().__init__()
        self.pipe = pipe
        self.env_class = env_class
        self.env_args = env_args
        self.env_id = env_id

    def run(self):
        torch.set_grad_enabled(False)

        '''build env'''
        if self.env_class.__module__ == 'gym.envs.registration':  # is standard OpenAI Gym env
            env = self.env_class(id=self.env_args['env_name'])
        else:
            env = self.env_class(**kwargs_filter(self.env_class.__init__, self.env_args.copy()))

        '''set env random seed'''
        random_seed = self.env_id
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        state = env.reset()
        state = torch.tensor(state)
        self.pipe.send(state)

        while True:
            action = self.pipe.recv()
            state, reward, done, info_dict = env.step(action)
            state = env.reset() if done else state
            self.pipe.send((state, reward, done, info_dict))


class VecEnvSync:
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
        assert self.num_envs <= 64
        self.pipes, pipes = list(zip(*[Pipe() for _ in range(self.num_envs)]))
        self.sub_envs = [SubEnvSync(pipe=pipe, env_class=env_class, env_args=env_args, env_id=env_id)
                         for env_id, pipe in enumerate(pipes)]
        [p.start() for p in self.sub_envs]

    def reset(self) -> Tensor:  # reset the agent in env
        torch.set_grad_enabled(False)

        states = [p.recv() for p in self.pipes]
        states = torch.stack(states).to(self.device)
        return states

    def step(self, action: Tensor) -> (Tensor, Tensor, Tensor, List[dict]):  # agent interacts in env
        for pipe, a in zip(self.pipes, action):
            pipe.send(a)
        states, rewards, dones, info_dicts = list(zip(*[p.recv() for p in self.pipes]))
        states = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        return states, rewards, dones, info_dicts

    def close(self):
        [p.terminate() for p in self.sub_envs]


class SubEnvAsync(Process):
    def __init__(self, pipe: Pipe, main_p: Pipe, env_class, env_args: dict, env_id: int = 0):
        super().__init__()
        self.pipe = pipe
        self.main_p = main_p
        self.env_class = env_class
        self.env_args = env_args
        self.env_id = env_id

    def run(self):
        torch.set_grad_enabled(False)

        '''build env'''
        if self.env_class.__module__ == 'gym.envs.registration':  # is standard OpenAI Gym env
            env = self.env_class(id=self.env_args['env_name'])
        else:
            env = self.env_class(**kwargs_filter(self.env_class.__init__, self.env_args.copy()))

        '''set env random seed'''
        random_seed = self.env_id
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        state = env.reset()
        state = torch.tensor(state)
        self.main_p.send((self.env_id, state))

        while True:
            action = self.pipe.recv()
            state, reward, done, info_dict = env.step(action)
            state = env.reset() if done else state
            self.main_p.send((self.env_id, state, reward, done, info_dict))


class VecEnvAsync:
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
        assert self.num_envs <= 64
        pipes, self.pipes = list(zip(*[Pipe(duplex=False) for _ in range(self.num_envs)]))
        self.main_p, main_p = Pipe(duplex=False)
        self.sub_envs = [SubEnvAsync(pipe=pipe, env_class=env_class, env_args=env_args, env_id=env_id, main_p=main_p)
                         for env_id, pipe in enumerate(pipes)]
        [p.start() for p in self.sub_envs]

    def reset(self) -> Tensor:  # reset the agent in env
        torch.set_grad_enabled(False)

        _, states = self.get_orderly_zip_list_return()
        states = torch.stack(states).to(self.device)
        return states

    def step(self, action: Tensor) -> (Tensor, Tensor, Tensor, List[dict]):  # agent interacts in env
        for pipe, a in zip(self.pipes, action):
            pipe.send(a)

        _, states, rewards, dones, info_dicts = self.get_orderly_zip_list_return()
        states = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        return states, rewards, dones, info_dicts

    def close(self):
        [p.terminate() for p in self.sub_envs]

    def get_orderly_zip_list_return(self):
        res = [self.main_p.recv() for _ in range(self.num_envs)]
        env_ids = [r[0] for r in res]
        res = [res[i] for i in env_ids]  # orderly
        return list(zip(*res))


def demo__vec_env_with_gym_env(num_envs=8):
    env_name = 'BipedalWalker-v3'
    env_class = gym.make
    env_args = {'env_name': env_name,
                'num_envs': num_envs,
                'max_step': 1600,
                'state_dim': 24,
                'action_dim': 4,
                'if_discrete': False, }

    env = build_vec_env(env_class=env_class, env_args=env_args)
    device = env.device
    action_dim = env_args['action_dim']

    states = env.reset()
    assert isinstance(states, Tensor)

    from time import time
    timer = time()
    for t in range(2 ** 12):
        actions = torch.rand((num_envs, action_dim), dtype=torch.float32, device=device)
        states, rewards, dones, _ = env.step(actions)
        # print(';;', rewards)
    env.close()
    print(f'NumEnvs: {num_envs:4} | UsedTime:  {time() - timer:9.4f}  {(time() - timer) / num_envs:9.4f}')


def demo__vec_env_with_custom_env(num_envs=8):
    from elegantrl.envs.CustomGymEnv import PendulumEnv
    env_class = PendulumEnv
    env_args = {'env_name': 'CustomEnv-v0',
                'num_envs': num_envs,
                'max_step': 200,
                'state_dim': 3,
                'action_dim': 1,
                'if_discrete': False}

    env = build_vec_env(env_class=env_class, env_args=env_args)
    device = env.device
    action_dim = env_args['action_dim']

    states = env.reset()
    assert isinstance(states, Tensor)

    from time import time
    timer = time()
    for t in range(2 ** 12):
        actions = torch.rand((num_envs, action_dim), dtype=torch.float32, device=device)
        states, rewards, dones, _ = env.step(actions)
        # print(';;', rewards)
    env.close()
    print(f'NumEnvs: {num_envs:4} | UsedTime:  {time() - timer:9.4f}  {(time() - timer) / num_envs:9.4f}')


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    ListNumEnvs = [2 ** n for n in range(1, 10)]

    # print('| demo__vec_env_with_custom_env')
    # for n in ListNumEnvs:
    #     demo__vec_env_with_custom_env(num_envs=n)

    print('| demo__vec_env_with_gym_env')
    for n in ListNumEnvs:
        demo__vec_env_with_gym_env(num_envs=n)

"""
VecEnvSync
| demo__vec_env_with_custom_env                                                                                    
NumEnvs:    2 | UsedTime:    13.9059     6.9529
NumEnvs:    4 | UsedTime:    19.2333     4.8083
NumEnvs:    8 | UsedTime:    32.5779     4.0722
NumEnvs:   16 | UsedTime:    57.2705     3.5794
NumEnvs:   32 | UsedTime:   105.3062     3.2908
NumEnvs:   64 | UsedTime:   211.1338     3.2990

VecEnvAsync
| demo__vec_env_with_custom_env                                                                                    
NumEnvs:    2 | UsedTime:    10.8249     5.4124                                                                    
NumEnvs:    4 | UsedTime:    15.1600     3.7900                                                                    
NumEnvs:    8 | UsedTime:    25.9492     3.2437                                                                    
NumEnvs:   16 | UsedTime:    46.3975     2.8998                                                                    
NumEnvs:   32 | UsedTime:    87.0220     2.7194                                                                    
NumEnvs:   64 | UsedTime:   161.7657     2.5276                                                                    
NumEnvs:  128 | UsedTime:   335.8240     2.6236 

| demo__vec_env_with_gym_env
NumEnvs:    2 | UsedTime:    14.0697     7.0349
NumEnvs:    4 | UsedTime:    19.8187     4.9547
NumEnvs:    8 | UsedTime:    28.2857     3.5357
NumEnvs:   16 | UsedTime:    49.3061     3.0816
NumEnvs:   32 | UsedTime:    87.6757     2.7399
NumEnvs:   64 | UsedTime:   170.7165     2.6674
NumEnvs:  128 | UsedTime:   348.3861     2.7218
"""
