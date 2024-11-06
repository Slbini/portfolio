# -*- coding: utf-8 -*-
# ## Requirement

# +
# #!pip install gymnasium
# #!pip install opencv-python==4.8.0.74

# +
import gymnasium as gym
import numpy as np
import random

import matplotlib.pyplot as plt
import PIL.Image
from IPython.display import Image
# -

# ## **MDP problem(environment)** : [Taxi](https://gymnasium.farama.org/environments/toy_text/taxi/)
# ---
#
# - **State space**  : 500(=25×4×5) discrete states 
# - **Action space** : 6 discrete actions
#     - 0: Move south (down)
#     - 1: Move north (up)
#     - 2: Move east (right)
#     - 3: Move west (left)
#     - 4: Pickup passenger
#     - 5: Drop off passenger
#
#
# - **Reward** : -1 , -10, +20
# ---
# - How to solve Taxi problem ?
#   > 1. Move to the passenger.
#   2. Pickup the passenger.
#   3. Drop off the passenger to the goal.
#
# - Episode End
#   1. Termination : Taxi drop off the passenger to the goal.
#   2. Truncation : The length of episode is 200.

env = gym.make('Taxi-v3', render_mode='rgb_array') # load environment

print(env.observation_space.n) # |State space|
print(env.action_space.n)      # |Action space|

state, info = env.reset(seed=2024) # get random starting state
img = env.render() # rendering the screen of the environment
PIL.Image.fromarray(np.array(img)) # invert numpy array to image

state # integer value corresnponding to the state

info # prob : probability that the action we choose will actually be taken & action_mask : possible action at the current state

next_state, reward, terminated, truncated, info = env.step(2) # take action

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
    render_images[0].save(filename, save_all=True, optimize=False, append_images=render_images[1:], duration=500, loop=0)

    print(f'Episode Length : {len(render_images)-1}')
    print(f'Total rewards : {total_reward}')
    print('GIF is made successfully!')

    return filename


# +
env = gym.make('Taxi-v3', render_mode='rgb_array')
play = random_play(env, 'random_Taxi', seed=2024)

Image(open(play,'rb').read())
# -


