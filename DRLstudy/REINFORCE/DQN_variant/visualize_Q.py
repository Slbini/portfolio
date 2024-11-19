# -*- coding: utf-8 -*-
# +
# #!pip install gymnasium
# #!pip install opencv-python==4.8.0.74

# +
import gymnasium as gym
import numpy as np
import random
import time, os

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

import matplotlib.pyplot as plt
import PIL.Image
from PIL import ImageDraw, ImageFont
from IPython.display import Image


# -

# ## Utils

class DQN_net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN_net, self).__init__()
        self.relu = F.relu
        self.linear_1 = nn.Linear(state_dim,128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, action_dim)
        
    def forward(self, x):
        x = self.relu(self.linear_1(x))
        x = self.relu(self.linear_2(x))
        x = self.linear_3(x)

        return x


class DQN:    
    def __init__(self,  state_dim, action_dim, gamma):
        self.behavior_Q = DQN_net(state_dim, action_dim)
        self.target_Q   = DQN_net(state_dim, action_dim)
        
        self.gamma = gamma    
        self.state_dim =  state_dim
        self.action_dim = action_dim
    
    # get action from the Epsilon-greedy policy
    def get_action(self, state, epsilon=0):
        action_logit = self.behavior_Q.to('cpu')(torch.FloatTensor(np.array([state])))  # input shape = (batch, state_dim)
        explore = np.random.rand()
        if explore < epsilon:
            # random action with probability epsilon (Exploration)
            return np.random.choice(self.action_dim)
        else:
            # greedy action with probability (1 - epsilon) (Exploitation)
            return torch.argmax(action_logit, dim=1)[0].detach().numpy()


def play_with_Q(env, agent, name='', seed=None):
    
    Q_list = []
    render_images = []
    total_reward = 0
    state, _ = env.reset(seed=seed)
    image_array = env.render()
    render_images.append(PIL.Image.fromarray(image_array))

    terminated, truncated = False, False
    agent.behavior_Q = agent.behavior_Q.to('cpu')
    
    with torch.no_grad():
        # episode start
        while not terminated and not truncated:
            action = agent.get_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            Q = agent.behavior_Q(torch.FloatTensor(np.array(state)))
            Q_list.append(Q)
            image_array = env.render()
            
            # we will write text on image
            image = PIL.Image.fromarray(image_array)
            draw_image = ImageDraw.Draw(image)
            
            # write current reward
            text = f'reward = {total_reward}'
            font = ImageFont.load_default(15)
            #image size = (600,400)
            x = 50  # x_position of text(from left)
            y = 70  # y_position of text(from top)
            # apply text on image
            draw_image.text((x, y), text, fill='black', font=font)
            
            # write Q(s,left), Q(s,right)
            left_color  = 'red' if Q[0] > Q[1] else 'black'
            right_color = 'black' if left_color=='red' else 'red'  
            
            text = f'Q(s,   left ) = {Q[0]}'
            font = ImageFont.load_default(20)
            x, y = 50, 100
            draw_image.text((x, y), text, fill=left_color, font=font)

            text = f'Q(s, right) = {Q[1]}'
            font = ImageFont.load_default(20)
            x, y = 50, 120 
            draw_image.text((x, y), text, fill=right_color, font=font)

            render_images.append(image)
        
    # episode finished
    filename = 'play_with_Q' + name + '.gif'

    # create and save GIF
    render_images[0].save(filename, save_all=True, optimize=False, append_images=render_images[1:], duration=300, loop=0)

    print(f'Episode Length : {len(render_images)-1}')
    print(f'Total rewards : {total_reward}')
    print('GIF is made successfully!')

    return filename, Q_list


# ## Load the Agent and Play

env = gym.make('CartPole-v1', render_mode='rgb_array')
agent = DQN(env.observation_space.shape[0], env.action_space.n, gamma=None)

saved_agent =    # file path of saved agent
agent.behavior_Q = torch.load(saved_agent)

filename, Q_list  = play_with_Q(env, agent, name='', seed=2024)
Image(open(filename,'rb').read())

Q_list


