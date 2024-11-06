# -*- coding: utf-8 -*-
# ### **Deep Reinforcement Learning(DRL)** Code
#
# ---
#
# - DRL aims to **solve MDP**(Markov Decision Process) problems. That is, DRL aims to **find an optimal policy**.
# - In this notebook, we aims to implement the following DRL algorithm : `DDQG`
# - As an environment in which agents will interact, we will use [OpenAI Gymnasium library](https://gymnasium.farama.org/)
#
# Code Reference
# - https://github.com/ghliu/pytorch-ddpg/blob/master/ddpg.py
#
#

# ## Requirements

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
from collections import deque

import matplotlib.pyplot as plt
import PIL.Image
from IPython.display import Image
from pyvirtualdisplay import Display


# -

# ## Utils

# learning rate scheduler (linear)
# decay learning rate linearly from 'initial_lr' to 'final_lr'
def linear_schedule(episode, max_episode, initial_lr, final_lr):
    start, end = initial_lr, final_lr
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


# ## Objective funtion of `DDPG`
# > **Actor** ($\theta$) : $\nabla J(\theta) \approx \frac{1}{|B|}\sum_{i \in B}
#  \nabla_{a}Q_{\phi}(s_i,a)\,\big|_{a=\mu_{\theta}(s_i)}
#  \, \nabla_{\theta} \mu_{\theta}(s_i)$
#  <br><br>
#  **Critic** ($\phi$) : $L(\phi)= \, \frac{1}{|B|}\sum_{i \in B}[$
# <font color=blue>$r_{i+1} + \gamma\,  \hat{Q}_{\hat{\phi}}(s_{i+1},\hat{\mu}_{\hat{\theta}}(s_{i+1}))$</font>
#  $ - \, Q_{\phi}(s_i,a_i)]^2$
#  <br>
#  where $B$ is the sampled mini-batch from the replay buffer.
#

# +
# noise generator for 'exploration' of deterministic policy
class RandomProcess(object):
    def reset_states(self):
        pass

class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma

class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)


# +
# how to use OrnsteinUhlenbeckProcess
nb_actions = 3 # number of noises to be generated
ou_theta = 0.15 # 'noise theta'
ou_mu = 0.0 # 'noise mu'
ou_sigma = 0.2 # 'noise sigma'

rp = OrnsteinUhlenbeckProcess(size=nb_actions, theta=ou_theta, mu=ou_mu, sigma=ou_sigma)
rp.sample()


# +
# neural network structure of Actor and Critic
class Actor_net(nn.Module):
    def __init__(self, state_dim, action_dim, action_high):
        super().__init__()
        hidden_space1 = 400
        hidden_space2 = 300

        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh() )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_space2, action_dim),
            nn.Tanh() ) # use 'Tanh' in the output layer for the bounded policy
        self.action_high = torch.nn.Parameter(torch.FloatTensor(action_high), requires_grad=False)
        
    def forward(self, x):
        x = self.shared_net(x.float())
        action = self.output_layer(x)

        return self.action_high*action

class Critic_net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        hidden_space1 = 16
        hidden_space2 = 32

        self.shared_net = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh() )
        
        self.output_layer= nn.Linear(hidden_space2, 1)
        
    def forward(self, s, a):
        x = torch.concat((s, a), dim =1 )
        x = self.shared_net(x)
        x = self.output_layer(x)
        
        return x


# -

# Agent that will be interact with environment and trained
class DDPG:
    def __init__(self, state_dim, action_dim, action_high, gamma, device):
        self.behavior_Actor = Actor_net(state_dim, action_dim, action_high)
        self.behavior_Critic = Critic_net(state_dim, action_dim)
        self.target_Actor = Actor_net(state_dim, action_dim, action_high)
        self.target_Critic = Critic_net(state_dim, action_dim)
        
        self.gamma = gamma
        self.state_dim  =  state_dim
        self.action_dim = action_dim
        self.device = device
        
        self.random_process = OrnsteinUhlenbeckProcess(size=action_dim, theta=0.15, mu=0.0, sigma=0.2)
    
    # get action from the actor
    # add noise only when training 
    def get_action(self, state, test=False):
        action = self.behavior_Actor.to('cpu')( torch.FloatTensor(np.array([state])))[0]
        if test == False:
            action += torch.FloatTensor(self.random_process.sample()) # add noise for exploration

        return action
    
    # update actor and critic 1-step
    def update(self, actor_optimizer, critic_optimizer, buffer, batch_size):
        state_arr, action_arr, reward_arr, next_state_arr, done_arr = buffer.sampling(batch_size,self.device)

        # update Critic
        target = reward_arr + self.gamma*self.target_Critic.to(self.device)(next_state_arr, self.target_Actor.to(self.device)(next_state_arr))*(1-done_arr)
        predict = self.behavior_Critic.to(self.device)(state_arr, action_arr)
        td_error = target.detach() -  predict
        critic_loss = torch.mean(td_error**2)
        
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step() 
        
        # update Actor
        action = self.behavior_Actor.to(self.device)(state_arr)
        predict_actor = self.behavior_Critic.to(self.device)(state_arr, action.clone())
        actor_loss  = torch.mean((-1)*predict_actor)
                                                            
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()   
    
    # update target network
    # we can choose 'soft update' or 'hard update'
    def update_target(self, soft_tau=None):
        if soft_tau == None:
            self.target_Actor.load_state_dict(self.behavior_Actor.state_dict())
            self.target_Critic.load_state_dict(self.behavior_Critic.state_dict())
        
        elif soft_tau:
            for behavior, target in zip(self.behavior_Actor.parameters(), self.target_Actor.parameters()):
                target.data.copy_(soft_tau * behavior.data + (1.0 - soft_tau) * target.data)
            for behavior, target in zip(self.behavior_Critic.parameters(), self.target_Critic.parameters()):
                target.data.copy_(soft_tau * behavior.data + (1.0 - soft_tau) * target.data)
    
    # train agent 
    def train(self, env, max_episode, evaluate_period, evaluate_num,
              actor_initial_lr, actor_final_lr, critic_initial_lr, critic_final_lr,
              update_period, target_update_period, buffer_size, batch_size, soft_tau):
        start = time.time()
        reward_list = [] # evaluation result(reward) will be inserted during the training
        
        replay_buffer = ReplayBuffer(capacity=buffer_size)
        
        actor_optimizer = torch.optim.Adam(self.behavior_Actor.parameters(), lr=actor_initial_lr)
        critic_optimizer = torch.optim.Adam(self.behavior_Critic.parameters(), lr=critic_initial_lr)
        self.update_target()
        
        for episode in range(max_episode):
            # new episode start
            done = False
            episode_length = 0
            
            actor_lr      = linear_schedule(episode, max_episode, actor_initial_lr, actor_final_lr)
            actor_optimizer.learning_rate = actor_lr
            critic_lr      = linear_schedule(episode, max_episode, critic_initial_lr, critic_final_lr)
            critic_optimizer.learning_rate = critic_lr
            
            state, info = env.reset()
            
            # interact with environment and train
            while not done:
                action = self.get_action(state).detach().numpy()
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_length += 1
                
                replay_buffer.store([state, action, [reward], next_state, [terminated]])
                
                # update behavior
                if replay_buffer.size() >= batch_size and episode_length%update_period == 0:
                    self.update(actor_optimizer, critic_optimizer, replay_buffer, batch_size)
                # update target
                if replay_buffer.size() >= batch_size and soft_tau == None and episode_length%target_update_period == 0:
                    self.update_target()
                elif replay_buffer.size() >= batch_size and soft_tau :
                    self.update_target(soft_tau)

                state = next_state
                                
            # episode finished and evaluate the current policy
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
            done = False
            episode_reward = 0

            state, info = env.reset()
            while not done:
                action = self.get_action(state, test=True).detach().numpy()
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
max_episode =   # the number of episodes that agent will be trained
evaluate_period = 5   # episode period that agent's policy will be evaluated
evaluate_num    = 10   # the number of episodes that agent will be evaluated

actor_initial_lr =    # start learning rate of Actor
actor_final_lr   =    # final learning rate of Actor  
critic_initial_lr =    # start learning rate of Critic
critic_final_lr   =    # final learning rate of Critic  
gamma =   # gamma : reward discount rate

# Replay Buffer
buffer_size =    # size of the replay buffer 
batch_size  =    # size of the mini-batch

# Target network
target_update_period =  # hard update (if use soft update, set this value to any positive integer such as '20')
soft_tau =   # soft update (if use hard update, set this value to 'None')
update_period = 1 # step period that agent will be trained

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
repeat = 3 # repeat same experiment for the reliable result
# -

# train : DDPG
reward_list =[]
for i in range(repeat):
    # control randomness for reproducibility
    seed = 100*(i+1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = gym.make('InvertedPendulum-v5', render_mode='rgb_array')
    agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0],env.action_space.high, gamma, device)
    reward = agent.train( env, max_episode, evaluate_period, evaluate_num,
                         actor_initial_lr, actor_final_lr, critic_initial_lr, critic_final_lr,
                        update_period, target_update_period, buffer_size, batch_size, soft_tau )

    reward_list.append(reward)

# ## **Results**

# ### Plot the results

# create folder to save the result
save_folder = './result'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# +
save_file = 'DDPG.png'
save_path  = os.path.join(save_folder, save_file)

plot(reward_list, 'DDPG', save_path=save_path)
# -

# ### Test the trained agent and save it into a GIF.

'''display = Display(visible=0, size=(140, 90))
display.start()

env = gym.make('InvertedPendulum-v5', render_mode='rgb_array')
play = play_and_save(env, agent, 'DDPG', seed=8)

display.stop()
Image(open(play,'rb').read())'''
