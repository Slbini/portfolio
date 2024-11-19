# -*- coding: utf-8 -*-
# ### **Deep Reinforcement Learning(DRL)** Code
#
# ---
#
# - DRL aims to **solve MDP**(Markov Decision Process) problems. That is, DRL aims to **find an optimal policy**.
# - In this notebook, we aims to implement the following DRL algorithm : `REINFORCE`, `REINFORCE with baseline`
# - As an environment in which agents will interact, we will use [OpenAI Gymnasium library](https://gymnasium.farama.org/)

# ## Requirement

# +
# #!pip install gymnasium
# #!pip install gymnasium[mujoco]
# #!pip install opencv-python==4.8.0.74

# +
import gymnasium as gym
import numpy as np
import random
import time, os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from collections import deque

import matplotlib.pyplot as plt
import PIL.Image
from IPython.display import Image
from pyvirtualdisplay import Display


# -

# ## Utils

# learning rate/epsilon scheduler (linear)
# decay linearly from 'initial' to 'final'
def linear_schedule(episode, max_episode, initial, final):
    start, end = initial, final
    if episode < max_episode:
        return (start*(max_episode-episode) + end*episode) / max_episode
    else:
        return end


# plot the experiment results (rewards)
# given the list of rewards, plot the highest, lowest, and mean reward graph.
def plot(rewards, title:str, save_path=None):  
      
    plt.figure(figsize=[4,2], dpi=300)
    plt.title(title , fontsize=9)
    # plot reward
    high_rewards= np.max(rewards , axis= 0)
    low_rewards= np.min(rewards , axis= 0)
    mean_rewards= np.mean(rewards , axis= 0)
    
    plt.xlabel('Episodes', fontsize=7)
    plt.ylabel('Total Rewards', fontsize=7)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.grid(linewidth=.1)

    x= np.arange(1, len(rewards[0])+1)
    plt.plot(x, high_rewards, 'b-', linewidth=.1, alpha=0.2)
    plt.plot(x, low_rewards, 'b-', linewidth=.1, alpha=0.2)
    plt.plot(x, mean_rewards, 'b-', linewidth=.2)
    
    if save_path!=None:
        plt.savefig(save_path, format='png')
        
    plt.show()


# create a GIF that shows how agent interacts with environment (play 1 episode)
# given the trained agent and environment, the agent interact with the environment and save into a GIF file 
def play_and_save(env, agent, name='', seed=None):
    
    render_images = []
    total_reward = 0
    state, _ = env.reset(seed=seed)
    image_array = env.render()
    render_images.append(PIL.Image.fromarray(image_array))

    terminated, truncated = False, False
    agent.Policy = agent.Policy.to('cpu')
    
    # episode start
    while not terminated and not truncated:
        action, log_prob = agent.get_action(state)
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


# ## Objective funtion of `REINFORCE`
# > $\nabla_{\theta} J(\theta) = \, \mathbb{E}_{\pi_{\theta}} \big[\,r(\tau)
# \sum_{t=0}^{T-1} \, \nabla_{\theta} log \pi_{\theta}(a_t|s_t) \,\big]
# \approx \, \frac{1}{M}\sum_{i=1}^M \big[
# \sum_{t=0}^{T-1} G_t^{(i)}\, \nabla_{\theta} log \pi_{\theta}(a_t^{(i)}|s_t^{(i)}) \,\big]$
#  <br>
#  where $M$ is the number of episodes.
#

class Policy_net(nn.Module):
    def __init__(self, state_dim, action_dim, action_high):
        super().__init__()
        self.action_high = torch.FloatTensor(action_high)
        hidden_space1 = 16
        hidden_space2 = 32

        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_space1),
            nn.ReLU(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.ReLU() )
        
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_dim),
            nn.Tanh())
        
        self.policy_std_net = nn.Sequential(
            nn.Linear(hidden_space2, action_dim) )
        
    def forward(self, x):

        shared_features = self.shared_net(x.float())

        action_mean = self.policy_mean_net(shared_features)
        action_std = torch.log(  1 + torch.exp(self.policy_std_net(shared_features))  )

        return self.action_high*action_mean, action_std


class Baseline_net(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        hidden_space1 = 16
        hidden_space2 = 32

        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_space1),
            nn.ReLU(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.ReLU(),
        )
        self.output_layer= nn.Linear(hidden_space2, 1)
        
    def forward(self, x):
        
        x = self.shared_net(x.float())
        state_value = self.output_layer(x)

        return state_value


# Agent that will be interact with environment and trained
class REINFORCE:
    def __init__(self, state_dim, action_dim,action_high, gamma):
        self.Policy = Policy_net(state_dim, action_dim, action_high)
        
        self.gamma = gamma
        self.state_dim  =  state_dim
        self.action_dim = action_dim
    
    # get action from the Epsilon-greedy policy
    def get_action(self, state):
        action_mean, action_std = self.Policy( torch.tensor(np.array([state]))) 
        distrib = Normal(action_mean[0], action_std[0] + 1e-8) # create normal distribution
        action = distrib.sample() # sample action from the distribution
        log_prob = distrib.log_prob(action)   # save the log probability of the sampled action

        return action.detach().numpy(), log_prob
    
    # update policy network
    def update(self, optimizer, rewards, log_probs):
        G = 0
        loss = 0
        for i in range(len(rewards)-1, -1, -1):
            G = rewards[i] + self.gamma * G
            loss += (-1) * G * log_probs[i].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
     
    # train agent
    def train(self, env, max_episode, evaluate_period, evaluate_num, initial_lr, final_lr):
        
        start= time.time()
        reward_list = []
        optimizer = torch.optim.Adam(self.Policy.parameters(), lr=initial_lr)
        
        for episode in range(max_episode):
            # new episode start
            done = False
            rewards = []
            log_probs = []
            
            lr      = linear_schedule(episode, max_episode//2, initial_lr, final_lr)
            optimizer.learning_rate = lr
            
            state, info = env.reset()
            
            # interact with environment and train
            while not done:
                action, log_prob = self.get_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                
                done = terminated or truncated
                state = next_state
            # episode finished
            # update policy
            self.update(optimizer, rewards, log_probs)
            # evaluate the current policy
            if (episode+1)%evaluate_period == 0 :
                reward = self.test(env, evaluate_num)
                reward_list.append(reward)
        
        end =time.time()
        print(f'Training time : {(end-start)/60:.2f}(min)')

        return reward_list
    
    # evaluate current policy
    # return average reward value over the several episodes
    def test(self, env, evaluate_num=10):

        reward_list = []

        for episode in range(evaluate_num):
            # new episode start
            done = False
            episode_reward = 0

            state, info = env.reset()
            while not done:
                action, _ = self.get_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                state = next_state

            # episode finished
            reward_list.append(episode_reward)

        return np.mean(reward_list)


# Agent that will be interact with environment and trained
class REINFORCE_with_baseline:
    def __init__(self, state_dim, action_dim,action_high, gamma):
        self.Policy = Policy_net(state_dim, action_dim, action_high)
        self.Baseline = Baseline_net(state_dim)
        
        self.gamma = gamma
        self.state_dim  =  state_dim
        self.action_dim = action_dim

    # get action from the Epsilon-greedy policy
    def get_action(self, state):
        action_mean, action_std = self.Policy( torch.tensor(np.array([state]))) 
        distrib = Normal(action_mean[0], action_std[0] + 1e-8) # create normal distribution
        action = distrib.sample() # sample action from the distribution
        log_prob = distrib.log_prob(action)   # save the log probability of the sampled action

        return action.detach().numpy(), log_prob

    # update policy network and baseline
    def update(self, policy_optimizer, baseline_optimizer, rewards, baseline, log_probs):
        n = len(rewards)
        
        G = 0
        policy_loss = 0
        baseline_loss = 0
        for i in range(n-1, -1, -1):
            G = rewards[i] + self.gamma * G
            policy_loss   += (-1) * (G - baseline[i].detach()) * log_probs[i].mean()
            baseline_loss += (G - baseline[i])**2
        
        policy_loss  /= n
        baseline_loss /= n
        
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        
        baseline_optimizer.zero_grad()
        baseline_loss.backward()
        baseline_optimizer.step()
    
    # train agent
    def train(self, env, max_episode, evaluate_period, evaluate_num,
              policy_initial_lr, policy_final_lr, baseline_initial_lr, baseline_final_lr):
        start= time.time()
        reward_list = []
        policy_optimizer = torch.optim.Adam(self.Policy.parameters(), lr=policy_initial_lr)
        baseline_optimizer = torch.optim.Adam(self.Baseline.parameters(), lr=baseline_initial_lr)
        
        for episode in range(max_episode):
            # new episode start
            done = False
            rewards = []
            baseline_values = []
            log_probs = []
            
            policy_lr      = linear_schedule(episode, max_episode//2, policy_initial_lr, policy_final_lr)
            policy_optimizer.learning_rate = policy_lr
            baseline_lr      = linear_schedule(episode, max_episode//2, baseline_initial_lr, baseline_final_lr)
            baseline_optimizer.learning_rate = baseline_lr
            
            state, info = env.reset()
            
            # interact with environment and train
            while not done:
                action, log_prob = self.get_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                
                baseline_value = self.Baseline(torch.FloatTensor(state))
                baseline_values.append(baseline_value)
                rewards.append(reward)
                log_probs.append(log_prob)
                
                done = terminated or truncated
                state = next_state
            # episode finished
            # update
            self.update(policy_optimizer, baseline_optimizer, rewards, baseline_values, log_probs)
            # evaluate the current policy
            if (episode+1)%evaluate_period == 0 :
                reward = self.test(env, evaluate_num)
                reward_list.append(reward)
        
        end =time.time()
        print(f'Training time : {(end-start)/60:.2f}(min)')

        return reward_list
    
    # evaluate current policy
    # return average reward value over the several episodes
    def test(self, env, evaluate_num=10):

        reward_list = []

        for episode in range(evaluate_num):
            # new episode start
            done = False
            episode_reward = 0

            state, info = env.reset()
            while not done:
                action, _ = self.get_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                state = next_state

            # episode finished
            reward_list.append(episode_reward)

        return np.mean(reward_list)


# ## **Train** Agent

# +
# hyperparameter for REINFORCE
max_episode =       # the number of episodes that agent will be trained
evaluate_period = 5    # episode period that agent's policy will be evaluated
evaluate_num    = 10   # the number of episodes that agent will be evaluated

initial_lr =    # starting learning rate
final_lr   =    # final learning rate
gamma =     # gamma : reward discount rate

repeat = 3 # repeat same experiment for the reliable result
# -

# train : REINFORCE
REINFORE_reward_list =[]
for i in range(repeat):
    # control randomness for reproducibility
    seed = 100*(i+1)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    env = gym.make('InvertedPendulum-v5', render_mode='rgb_array')
    agent = REINFORCE(env.observation_space.shape[0], env.action_space.shape[0],env.action_space.high, gamma)
    reward = agent.train( env, max_episode, evaluate_period, evaluate_num, initial_lr, final_lr )
    REINFORE_reward_list.append(reward)

# +
# hyperparameter for REINFORCE with baseline
max_episode =       # the number of episodes that agent will be trained
evaluate_period = 5    # episode period that agent's policy will be evaluated
evaluate_num    = 10   # the number of episodes that agent will be evaluated

policy_initial_lr = 
policy_final_lr   = 
baseline_initial_lr = 
baseline_final_lr   = 
gamma = 

repeat = 3 # repeat same experiment for the reliable result
# -

# train : REINFORCE with baseline
REINFORE_with_baseline_reward_list =[]
for i in range(repeat):
    # control randomness for reproducibility
    seed = 100*(i+1)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    env = gym.make('InvertedPendulum-v5', render_mode='rgb_array')
    agent = REINFORCE_with_baseline(env.observation_space.shape[0], env.action_space.shape[0],env.action_space.high, gamma)
    reward = agent.train( env, max_episode, evaluate_period, evaluate_num, 
                         policy_initial_lr, policy_final_lr, baseline_initial_lr, baseline_final_lr )
    REINFORE_with_baseline_reward_list.append(reward)

# ## **Results**

# ### Plot the results

# create folder to save the result
save_folder = './result'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# +
save_file = 'REINFORCE.png'
save_path  = os.path.join(save_folder, save_file)

plot(REINFORE_reward_list, 'REINFORCE', save_path=save_path)

# +
save_file = 'REINFORCE_with_baseline.png'
save_path  = os.path.join(save_folder, save_file)

plot(REINFORE_with_baseline_reward_list, 'REINFORCE with baseline', save_path=save_path)
# -

# ### Test the trained agent and save it into a GIF.

'''display = Display(visible=0, size=(140, 90))
display.start()

env = gym.make('InvertedPendulum-v5', render_mode='rgb_array')
play = play_and_save(env, agent, 'REINFORCE', seed=8)

display.stop()
Image(open(play,'rb').read())'''


