# -*- coding: utf-8 -*-
# ## Requirement

# +
# #!pip install gymnasium
# #!pip install gymnasium[mujoco]
# #!pip install opencv-python==4.8.0.74

# +
import gymnasium as gym

import numpy as np
import PIL.Image
from IPython.display import Image
from pyvirtualdisplay import Display
# -

# ## **MDP problem(environment)** : [Inverted Pendulum](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/)
# ---
#
# - **State space**  : (Cart Position, Pole Angle, Cart Velocity, Pole Angular Velocity)
# - Car Position, Car Velocity, Pole Angle, Pole Angular Velocity $\in [-\infty, \infty]$
# - **Action space** : (Force) $\in [-3, 3]$
# - **Reward** : +1
# ---
# - How to solve Inverted Pendulum problem ?
# > keep the inverted pendulum upright for as long as possible.
# - Episode End
#   1. Termination : Any of the state space values is no longer finite.
#   2. Termination : The absolute value of the vertical angle between the pole and the cart is greater than 0.2 radian
#   3. Truncation : Episode length is greater than 1000.

env = gym.make('InvertedPendulum-v5', render_mode='rgb_array') # load environment

print(env.observation_space.shape) # shape of each state
print(env.action_space.shape)      # shape of each action
print(env.action_space.high)       # highest value of possible action

# +
state, info = env.reset() # reset the environment and get the starting state

display = Display(visible=0, size=(140, 90))
display.start()

img = env.render() # get the screen image of the environment
PIL.Image.fromarray(np.array(img))
# -

state

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
    render_images[0].save(filename, save_all=True, optimize=False, append_images=render_images[1:], duration=500, loop=0)

    print(f'Episode Length : {len(render_images)-1}')
    print(f'Total rewards : {total_reward}')
    print('GIF is made successfully!')

    return filename


# +
display = Display(visible=0, size=(140, 90))
display.start()

env = gym.make('InvertedPendulum-v5', render_mode='rgb_array')
play = random_play(env, 'random_play', seed=2024)

display.stop()
Image(open(play,'rb').read())
# -


