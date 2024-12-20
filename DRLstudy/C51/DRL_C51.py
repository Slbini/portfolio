# -*- coding: utf-8 -*-
# ### **Deep Reinforcement Learning(DRL)** Code
#
# ---
#
# - DRL aims to **solve MDP**(Markov Decision Process) problems. That is, DRL aims to **find an optimal policy**.
# - In this notebook, we aims to implement the following DRL algorithm : `Categorical DQN` (C51)
# - As an environment in which agents will interact, we will use [OpenAI Gymnasium library](https://gymnasium.farama.org/)

# ## Requirement

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
from IPython.display import Image


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
    #plt.close()


# create a GIF that shows how agent interacts with environment (play 1 episode)
# given the trained agent and environment, the agent interact with the environment and save into a GIF file 
def play_and_save(env, agent, name='', seed=None):
    
    render_images = []
    total_reward = 0
    state, _ = env.reset(seed=seed)
    image_array = env.render()
    render_images.append(PIL.Image.fromarray(image_array))

    terminated, truncated = False, False
    agent.behavior_Q = agent.behavior_Q.to('cpu')
    
    # episode start
    while not terminated and not truncated:
        action = agent.get_action(state)
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


# ## Objective funtion of `Categorical DQN` 
# > $L(\theta) = \frac{1}{|B|}\sum_{i \in B}D_{KL}(\,\Phi
#  \mathit{T}Z_{\hat{\theta}}(x_i,a_i)\  \|\ Z_{\theta}(x_i,a_i)  \,  ) $$
# \\ \quad \ \ \   =\frac{1}{|B|}\sum_{i \in B}\big[\,
#  \sum_z \,(\Phi
#  \mathit{T}Z_{\hat{\theta}})(z)\ \text{log} \frac{ (\Phi\mathit{T}Z_{\hat{\theta}})(z)}{Z_{\theta\,}(z)}\,\big] $$
#  \quad \ \ \ =\frac{1}{|B|}\sum_{i \in B}\big[\,
# \sum_z \,(\Phi \mathit{T}Z_{\hat{\theta}})(z)\ \text{log} (\Phi\mathit{T}Z_{\hat{\theta}})(z)\ $<font color=blue>$-\ (\Phi\mathit{T}Z_{\hat{\theta}})(z)\ \text{log} Z_{\theta\,}(z)$</font>$\,\big]$
#  <br><br>
#  where $B$ is the sampled mini-batch from the replay buffer.
#

class Categorical_DQN_net(nn.Module):
    def __init__(self, state_dim, action_dim, atom_num):
        super(Categorical_DQN_net, self).__init__()
        self.relu = F.relu
        self.linear_1 = nn.Linear(state_dim,128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, action_dim*atom_num)
        
        self.action_dim = action_dim
        self.atom_num = atom_num
        
    def forward(self, x):
        x = self.relu(self.linear_1(x))
        x = self.relu(self.linear_2(x))
        x = F.softmax(self.linear_3(x).view(-1, self.action_dim, self.atom_num), dim=-1)

        return x


# Agent that will interact with environment and  be trained
class Categorical_DQN:    
    def __init__(self,  state_dim, action_dim, atom_num, v_max, v_min, gamma):
        self.behavior_Z = Categorical_DQN_net(state_dim, action_dim, atom_num)
        self.target_Z   = Categorical_DQN_net(state_dim, action_dim, atom_num)
        
        self.gamma = gamma    
        self.state_dim =  state_dim
        self.action_dim = action_dim
        
        self.v_max = v_max
        self.v_min = v_min
        self.atom_num = atom_num
        self.support = torch.FloatTensor(np.linspace(v_min, v_max, atom_num) )
        self.delta = (v_max - v_min)/(atom_num-1)
    
    # get action from the Epsilon-greedy policy
    def get_action(self, state, epsilon=0):
        action_logit = self.behavior_Z(torch.FloatTensor(np.array([state])))  # input shape = (batch, state_dim)
        explore = np.random.rand()
        if explore < epsilon:
            # random action with probability epsilon (Exploration)
            return np.random.choice(self.action_dim)
        else:
            # greedy action with probability (1 - epsilon) (Exploitation)
            return ((action_logit*self.support).sum(dim=2)).argmax(dim=1)[0].detach().numpy()
    
    # update Q network 1-step
    def update(self, optimizer, buffer, batch_size, device):
        self.behavior_Z = self.behavior_Z.to(device)
        self.target_Z   = self.target_Z.to(device)
        self.support    = self.support.to(device)
        
        state_arr, action_arr, reward_arr, next_state_arr, done_arr = buffer.sampling(batch_size,device)
        
        #predict = torch.stack([self.behavior_Z(state_arr)[i][int(action_arr[i])] for i in range(batch_size)])
        predict = self.behavior_Z(state_arr).gather(1, action_arr.long().repeat(self.atom_num,1).T.reshape(batch_size,1,self.atom_num)).squeeze(-2)
        next_action_arr = ((self.target_Z(next_state_arr)*self.support).sum(dim=2)).argmax(dim=1).detach()
        #next_predict = torch.stack([self.target_Z(next_state_arr)[i][int(next_action_arr[i])] for i in range(batch_size)]).detach().cpu()
        next_predict = self.target_Z(next_state_arr).gather(1, next_action_arr.long().repeat(self.atom_num,1).T.reshape(batch_size,1,self.atom_num)).squeeze(-2).detach()
        
        # support of the target distribution
        target_support = (reward_arr.repeat(self.atom_num,1).T + self.gamma*((self.support).repeat(batch_size,1))*((1-done_arr).repeat(self.atom_num,1).T)).clip(self.v_min,self.v_max) 
        # position of the target distribution support (similar to index)
        target_support_position = ((target_support - self.v_min)/self.delta)
        # nearest upper/lower bound index
        ub = torch.ceil(target_support_position).long()
        lb = torch.floor(target_support_position).long()
        # probability list of the target distribution
        target = torch.zeros(batch_size, self.atom_num, device=device)
        
        # project the target distribution support onto the fixed default support
        '''
        for i in range(batch_size):
            for j in range(self.atom_num):
                target[i, int(lb[i,j])] += (next_predict[i,j] * (ub[i,j] - target_support_position[i,j]))
                target[i, int(ub[i,j])] += (next_predict[i,j] * (target_support_position[i,j] - lb[i,j]))
        '''
        delta = target_support_position - lb.float()
        target.scatter_add_(1, lb, next_predict * (1 - delta))
        target.scatter_add_(1, ub, next_predict * delta)
        
        # take only cross-entropy loss term in the KL Divergence loss term
        #loss = torch.sum((-1)*torch.FloatTensor(target).to(device)*torch.log(predict + 1e-8))/batch_size
        loss = (-(target * torch.log(predict + 1e-8)).sum(dim=1)).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        self.behavior_Z = self.behavior_Z.to('cpu')
        self.target_Z   = self.target_Z.to('cpu')
        self.support    = self.support.to('cpu')
     
    # update target network
    def update_target(self, ):
        self.target_Z.load_state_dict(self.behavior_Z.state_dict())
    
    # train agent
    def train(self, env, max_episode, evaluate_period, evaluate_num, buffer_size, initial_epsilon, final_epsilon, initial_lr, final_lr,
              update_period, target_update_period, batch_size=64, device='cpu'):
        
        start= time.time()  
        reward_list = [] # evaluation result(reward) will be inserted during the training

        replay_buffer = ReplayBuffer(capacity=buffer_size)
        optimizer = torch.optim.Adam(self.behavior_Z.parameters(), lr=initial_lr)
        self.update_target()

        for episode in range(max_episode):
            # new episode start
            done = False
            episode_length = 0

            epsilon = linear_schedule(episode, max_episode//4, initial_epsilon, final_epsilon)
            lr      = linear_schedule(episode, max_episode//2, initial_lr, final_lr)
            optimizer.learning_rate = lr
            
            state, info = env.reset()

            # interact with environment and train
            while not done:
                action = self.get_action(state, epsilon)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated 
                episode_length += 1

                replay_buffer.store([state, action, reward, next_state, terminated])
                
                # update behavior
                if replay_buffer.size() >= batch_size and episode_length%update_period == 0:
                    self.update(optimizer, replay_buffer, batch_size, device)
                # update target
                if replay_buffer.size() >= batch_size and episode_length%target_update_period == 0:
                    self.update_target()

                state = next_state
                
            # episode finished and evaluate the current policy
            if (episode+1)%evaluate_period == 0 :
                reward = self.test(env, evaluate_num)
                reward_list.append(reward)
        
        end= time.time()  
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
                action = self.get_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                state = next_state

            # episode finished
            reward_list.append(episode_reward)

        return np.mean(reward_list)


# replay buffer for experience replay
class ReplayBuffer():
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def store(self, transition):
        self.buffer.append(transition)

    def sampling(self, batch_size, device):
        experience_samples = random.sample(self.buffer, batch_size)
        state_arr, action_arr, reward_arr, next_state_arr, done_arr = map(np.asarray, zip(*experience_samples))

        state_arr      = torch.FloatTensor(state_arr).to(device)
        action_arr       = torch.FloatTensor(action_arr).to(device)
        reward_arr       = torch.FloatTensor(reward_arr).to(device)
        next_state_arr = torch.FloatTensor(next_state_arr).to(device)
        done_arr         = torch.FloatTensor(done_arr).to(device)

        return state_arr, action_arr, reward_arr, next_state_arr, done_arr

    def size(self):
        return len(self.buffer)


# ## **Train** Agent

# +
# hyperparameter
atom_num =       # number of the elements in the support (of the discrete distribution)
v_max =          # maximum value of reward
v_min =          # minimum value of reward

max_episode =          # the number of episodes that agent will be trained
evaluate_period = 5    # episode period that agent's policy will be evaluated
evaluate_num    = 10   # the number of episodes that agent will be evaluated

initial_lr =         # starting learning rate
final_lr   =         # final learning rate
initial_epsilon =    # starting epsilon
final_epsilon   =    # final epsilon
gamma =              # gamma : reward discount rate

# Replay Buffer
buffer_size =     # size of the replay buffer 
batch_size  =     # size of the mini-batch

# Target network
target_update_period =   # period of episode to update target network
update_period = 1        # period of episode that agent will be trained

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
repeat = 3 # repeat same experiment for the reliable result
# -

# train : Categorical_DQN
reward_list =[]
for i in range(repeat):
    # control randomness for reproducibility
    seed = 100*(i+1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    agent = Categorical_DQN(env.observation_space.shape[0], env.action_space.n, atom_num, v_max, v_min, gamma)
    reward = agent.train( env, max_episode, evaluate_period, evaluate_num, buffer_size, initial_epsilon, final_epsilon, initial_lr, final_lr, 
                          update_period, target_update_period, batch_size, device)
    reward_list.append(reward)

# ## **Results**

# ### Plot the results

# create folder to save the result
save_folder = './result'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# +
save_file = 'C51.png'
save_path  = os.path.join(save_folder, save_file)

plot(reward_list, 'C51', save_path=save_path)
# -

# ### Test the trained agent and save it into a GIF.

'''env = gym.make('CartPole-v1', render_mode='rgb_array')
play = play_and_save(env, agent, 'C51', seed=8)
Image(open(play,'rb').read())'''
