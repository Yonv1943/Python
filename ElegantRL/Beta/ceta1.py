import os
import sys
import ray
import time
import torch
import numpy
from ray.rllib.agents import ppo
from ray.rllib.agents import with_common_config
from StockTrading import StockEnvDOW30

gpu_id = int(sys.argv[-1][-4])
random_seed = 1943 + gpu_id

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
numpy.random.seed(random_seed)
torch.manual_seed(random_seed)

CHECKPOINT_DIR = f"./rllib_temp_{gpu_id}"
CHECKPOINT_FILE = f"last_checkpoint_{gpu_id}.out"

config = with_common_config({
    'gamma': 0.99,
    'lr': 1e-5,
    'num_workers': 4,
    # 'framework': 'torch',
    'num_gpus': 1,
    'sgd_minibatch_size': 256,
    'num_sgd_iter': (2515 * 8) * 2 ** 0 // 256,
    'train_batch_size': 2515 * 8,
    'entropy_coeff': 0.02,
    'vf_loss_coeff': 0.01,
    'model': {'fcnet_hiddens': [256, 256, 256]}})


def train_it():
    ray.init()
    # Configure RLLib with The Roadwork Environment

    agent = ppo.PPOTrainer(env=StockEnvDOW30, config=config)
    # agent = ppo.PPOTrainer(env=StockEnvDOW30)

    env_eval = StockEnvDOW30(if_eval=True)

    print(f"Starting training, you can view process through "
          f"`tensorboard --logdir={CHECKPOINT_DIR}` and opening http://localhost:6006")
    # Attempt to restore from checkpoint if possible.
    if os.path.exists(f"{CHECKPOINT_DIR}/{CHECKPOINT_FILE}"):
        checkpoint_path = open(f"{CHECKPOINT_DIR}/{CHECKPOINT_FILE}").read()
        print("Restoring from checkpoint path", checkpoint_path)
        agent.restore(checkpoint_path)

    total_step = 0
    evaluate_timer = 0
    train_time0 = 0

    counter = 0
    while total_step < int(5e6):
        counter += 1
        results = agent.train()

        total_step = results["timesteps_total"]
        total_time = results["time_total_s"]
        average_reward = results["episode_reward_mean"]

        if counter % 16:
            evaluate_time0 = time.time()
            actor = agent.workers.local_worker().get_policy()
            episode_return = get_episode_return(env_eval, actor)
            evaluate_timer += time.time() - evaluate_time0

            print(f"Step: {total_step:8}  "
                  f"Time: {total_time:8.3f}  "
                  f"AvgR: {average_reward:8.3f}  "
                  f"EpiR: {episode_return:8.3f}  ")
            checkpoint_path = agent.save(CHECKPOINT_DIR)
            # print("--> Last checkpoint", checkpoint_path)
            with open(f"{CHECKPOINT_DIR}/{CHECKPOINT_FILE}", "w") as f:
                f.write(checkpoint_path)
    total_time = time.time() - train_time0
    print(f'Total_time {total_time:.3f} - evaluate_time{evaluate_timer:.3f} '
          f'= {total_time - evaluate_timer:.3f}')


def get_episode_return(env, actor):
    state = env.reset()
    for _ in range(env.max_step):
        action = actor.compute_actions(state[None, :])[0][0]
        state, reward, done, _ = env.step(action)
    return env.episode_return


def evaluate_it():
    ray.init()

    agent = ppo.PPOTrainer(env=StockEnvDOW30)
    env_eval = StockEnvDOW30(if_eval=True)

    for n in range(64, 75):
        agent.restore(f'{CHECKPOINT_DIR}/checkpoint_{n}/checkpoint-{n}')

        actor = agent.workers.local_worker().get_policy()

        episode_return = get_episode_return(env_eval, actor)
        print(n, episode_return)
    ray.shutdown()


if __name__ == '__main__':
    train_it()
    # evaluate_it()
