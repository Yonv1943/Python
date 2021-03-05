from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy.random as rd
from time import sleep

"""Github: Yonv1943 2021-03-03
Demo_show_dymanic_images_in_colab.py
"""


import gym  # pip3 install gym==0.17 pyglet==1.5.0  # env.render() bug in gym==0.18, pyglet==1.6
# env = gym.make('CartPole-v0')
env = gym.make('LunarLander-v2')

from pyvirtualdisplay import Display
display = Display(visible=0, size=(400, 300))
display.start()

observation = env.reset()
cum_reward = 0
frames = []
for t in range(1000):
    frames.append(env.render(mode='rgb_array'))
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        break

for frame in frames:
    # action = rd.uniform(-1, 1, size=action_dim)
    # state, reward, done, _ = env.step(action)

    # image = env.render(mode='rgb_array')

    plt.imshow(frame)
    plt.show()
    sleep(0.01)
    clear_output(wait=True)


