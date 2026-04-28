"""Render a random LunarLander episode frame by frame in a notebook."""

from time import sleep

import gym
import matplotlib.pyplot as plt
from IPython.display import clear_output
from pyvirtualdisplay import Display


MAX_STEPS = 1000
DISPLAY_SIZE = (400, 300)
FRAME_DELAY = 0.01


def play_random_lunar_lander():
    """Run one random episode and display the captured frames."""
    display = Display(visible=0, size=DISPLAY_SIZE)
    display.start()

    env = gym.make("LunarLander-v2")

    try:
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            observation, _ = reset_result
        else:
            observation = reset_result

        frames = []

        for _ in range(MAX_STEPS):
            frames.append(env.render(mode="rgb_array"))

            action = env.action_space.sample()
            step_result = env.step(action)

            if len(step_result) == 5:
                observation, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                observation, reward, done, info = step_result

            if done:
                break

        plt.figure(figsize=(6, 4))
        for frame in frames:
            plt.imshow(frame)
            plt.axis("off")
            plt.show()
            sleep(FRAME_DELAY)
            clear_output(wait=True)

    finally:
        env.close()
        display.stop()


play_random_lunar_lander()