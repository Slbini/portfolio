# -*- coding: utf-8 -*-
# ### **Deep Reinforcement Learning(DRL)** Code
#
# ---
#
# - DRL aims to **solve MDP**(Markov Decision Process) problems. That is, DRL aims to **find an optimal policy**.
# - In this notebook, we aims to implement the following DRL algorithm : `TD Actor-Critic`
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
    agent.Actor = agent.Actor.to('cpu')
    
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


# ## Objective funtion of `TD Actor-Critic`
# >**Actor** <br> 
# $\nabla_{\theta} J(\theta) = \, \mathbb{E}_{\pi_{\theta}} \big[\,$
# <font color=blue>$(r+\gamma \, V_{\phi}(s')-V_{\phi}(s))$</font>
# <font color=brick>$ \,\nabla_{\theta} log \pi_{\theta}(a|s) \,\big]$</font>
# $\approx \, \frac{1}{N}\sum_{i=1}^N \big[\,
# (r_i+\gamma \, V_{\phi}(s'_i)-V_{\phi}(s_i))
#  \,\nabla_{\theta} log \pi_{\theta}(a_i|s_i)
# \,\big]$
#  <br><br>
#  **Critic** <br>
# $L(\phi) = \, \mathbb{E}_{\pi_{\theta}} \big[\,
# \big(r+\gamma \, V_{\phi}(s')-V_{\phi}(s)\big)^2 \,\big]
# \approx \, \frac{1}{N}\sum_{i=1}^N 
# \big(r_i+\gamma \, V_{\phi}(s'_i)-V_{\phi}(s_i)\big)^2$
#  <br><br>
# where $N$ is the number of transitions.
#

# +
class Actor_net(nn.Module):
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

    
class Critic_net(nn.Module):
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
        
    def forward(self, x: torch.Tensor):
        
        x = self.shared_net(x.float())
        state_value = self.output_layer(x)

        return state_value


# -

# Agent that will be interact with environment and trained
class TD_ActorCritic:
    def __init__(self, state_dim, action_dim, action_high, gamma):
        self.Actor = Actor_net(state_dim, action_dim, action_high)
        self.Critic = Critic_net(state_dim)
        
        self.gamma = gamma
        self.state_dim  =  state_dim
        self.action_dim = action_dim

    # get action from the policy(Actor)
    def get_action(self, state, test=False):
        if test == True:
            action_mean, action_std = self.Actor( torch.tensor(np.array([state])))
            return action_mean[0].detach().numpy()
        
        action_mean, action_std = self.Actor( torch.tensor(np.array([state]))) 
        distrib = Normal(action_mean[0], action_std[0] + 1e-8) # create normal distribution
        action = distrib.sample() # sample action from the distribution
        log_prob = distrib.log_prob(action)   # save the log probability of the sampled action

        return action.detach().numpy(), log_prob
    
    # update Actor/Critic network
    def update(self, actor_optimizer, critic_optimizer, transition):

        n = len(transition[0])
        actor_loss = 0
        critic_loss = 0
              
        td_error = [ transition[2][i] + self.gamma*transition[1][i]*(1-transition[3][i]) - transition[0][i] for i in range(len(transition[0]))]
        
        for i in range( n ):
            actor_loss  += (-1) * (transition[4][i].mean()) * (td_error[i].detach())
            critic_loss += (td_error[i])**2
        
        actor_loss  /= n
        critic_loss /= n
        
        # update Actor                                                            
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()   
        
        # update Critic
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()                                      
    
    # train agent
    def train(self, env, max_episode, evaluate_period, evaluate_num, update_period,
              actor_initial_lr, actor_final_lr, critic_initial_lr, critic_final_lr):
        start = time.time()
        reward_list = []
        
        actor_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=actor_initial_lr)
        critic_optimizer = torch.optim.Adam(self.Critic.parameters(), lr=critic_initial_lr)
        
        for episode in range(max_episode):
            # new episode start
            done = False
            episode_length = 0
            transition = [[],[],[],[],[]] # [value, next_value, reward, terminated, log_prob]
            
            actor_lr      = linear_schedule(episode, max_episode//2, actor_initial_lr, actor_final_lr)
            actor_optimizer.learning_rate = actor_lr
            critic_lr     = linear_schedule(episode, max_episode//2, critic_initial_lr, critic_final_lr)
            critic_optimizer.learning_rate = critic_lr
            
            state, info = env.reset()
            
            # interact with environment and train
            while not done:
                action, log_prob = self.get_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                episode_length += 1
                
                value = self.Critic(torch.FloatTensor(state))
                next_value = self.Critic(torch.FloatTensor(next_state)).detach()
                
                transition[0].append(value)
                transition[1].append(next_value)
                transition[2].append(reward)
                transition[3].append(terminated)
                transition[4].append(log_prob)
                
                done = terminated or truncated
                
                if episode_length%update_period==0 or done :
                    self.update(actor_optimizer, critic_optimizer, transition)
                    transition = [[],[],[],[],[]] # reset transition buffer
                    
                state = next_state
                                
            # episode finished and evaluate the current actor
            if (episode+1)%evaluate_period == 0 :
                reward = self.test(env, evaluate_num)
                reward_list.append(reward)
        
        end = time.time()
        print(f'Training time : {(end-start)/60:.2f}(min)')

        return reward_list
    
    # evaluate current policy
    # return average reward value over the several episodes
    def test(self, env, evaluate_num=10):

        reward_list = []

        for episode in range(evaluate_num):
            # new episode start
            terminated = False
            truncated = False
            done = terminated or truncated
            episode_reward = 0

            state, info = env.reset()
            while not done:
                action = self.get_action(state, test=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                state = next_state

            # episode finished
            reward_list.append(episode_reward)

        return np.mean(reward_list)


# ## **Train** Agent

# +
# hyperparameter for TD Actor-Critic
max_episode =          # the number of episodes that agent will be trained
update_period =        # update period(time-step) of agent
evaluate_period = 5    # episode period that agent's policy will be evaluated
evaluate_num    = 10   # the number of episodes that agent will be evaluated

actor_initial_lr =     # starting actor learning rate
actor_final_lr   =     # final actor learning rate
critic_initial_lr =    # starting critic learning rate
critic_final_lr   =    # final critic learning rate
gamma =    # gamma : reward discount rate

repeat = 3 # repeat same experiment for the reliable result
# -

# train : TD Actor-Critic
reward_list =[]
for i in range(repeat):
    # control randomness for reproducibility
    seed = 100*(i+1)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    env = gym.make('InvertedPendulum-v5', render_mode='rgb_array')
    agent = TD_ActorCritic(env.observation_space.shape[0], env.action_space.shape[0],env.action_space.high, gamma)
    reward = agent.train( env, max_episode, evaluate_period, evaluate_num, update_period,
                         actor_initial_lr, actor_final_lr, critic_initial_lr, critic_final_lr)
    reward_list.append(reward)

# ## **Results**

# ### Plot the results

# create folder to save the result
save_folder = './result'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# +
save_file = 'TD_ActorCritic.png'
save_path  = os.path.join(save_folder, save_file)

plot(reward_list, 'TD_ActorCritic', save_path=save_path)
# -

# ### Test the trained agent and save it into a GIF.

'''display = Display(visible=0, size=(140, 90))
display.start()

env = gym.make('InvertedPendulum-v5', render_mode='rgb_array')
play = play_and_save(env, agent, 'TD_ActorCritic', seed=8)

display.stop()
Image(open(play,'rb').read())'''


