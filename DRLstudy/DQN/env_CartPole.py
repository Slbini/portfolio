# -*- coding: utf-8 -*-
# ## Requirement

# +
# #!pip install gymnasium
# #!pip install opencv-python==4.8.0.74

# +
import gymnasium as gym

import numpy as np
import PIL.Image
from IPython.display import Image
from pyvirtualdisplay import Display
# -

# ## **MDP problem(environment)** : [Cart Pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
# ---
#
# - **State space**  : (Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity)
# - Car Position $\in [-4.8, 4.8]$, Car Velocity $\in [-\infty, \infty]$, Pole Angle $\in [-24^{\circ}, 24^{\circ}]$(radian), Pole Angular Velocity $\in [-\infty, \infty]$
# - **Action space** : 2 discrete actions
# - **Reward** : +1
# ---
# - How to solve Cart Pole problem ?
# > keep the pole upright for as long as possible.
# - Episode End
#   1. Termination : Pole Angle is greater than ±12°.
#   2. Termination : Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display).
#   3. Truncation : Episode length is greater than 500.

env = gym.make('CartPole-v1', render_mode='rgb_array') # load environment

print(env.observation_space.shape) # shape of each state
print(env.action_space.n)          # |Action space|

state, info = env.reset(seed=2024) # reset the environment and get the starting state
img = env.render() # get the screen image of the environment
PIL.Image.fromarray(np.array(img))

state # (Car Position, Car Velocity, Pole Angle, Pole Angular Velocity)

info

action = env.action_space.sample()
action

next_state, reward, terminated, truncated, info = env.step(action) # take action

img = env.render()
PIL.Image.fromarray(np.array(img))

print(next_state)
print(reward)
print(terminated)
print(truncated)
print(info)


# create a GIF that shows the random play in the environment (1 episode)
def random_play(env, name='', seed=None):
    
    render_images = []
    total_reward = 0
    state, _ = env.reset(seed=seed)
    image_array = env.render()
    render_images.append(PIL.Image.fromarray(image_array))

    terminated, truncated = False, False
    
    # episode start
    while not terminated and not truncated:
        action = env.action_space.sample()
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        image_array = env.render()
        render_images.append(PIL.Image.fromarray(image_array))
    # episode finished
    filename = 'play_' + name + '.gif'

    # create and save GIF
    render_images[0].save(filename, save_all=True, optimize=False, append_images=render_images[1:], duration=100, loop=0)

    print(f'Episode Length : {len(render_images)-1}')
    print(f'Total rewards : {total_reward}')
    print('GIF is made successfully!')

    return filename


# +
env = gym.make('CartPole-v1', render_mode='rgb_array')
play = random_play(env, 'random_DQN', seed=2024)

Image(open(play,'rb').read())
# -


