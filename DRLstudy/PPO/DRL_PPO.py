# -*- coding: utf-8 -*-
# ### **Deep Reinforcement Learning(DRL)** Code
#
# ---
#
# - DRL aims to **solve MDP**(Markov Decision Process) problems. That is, DRL aims to **find an optimal policy**.
# - In this notebook, we aims to implement the following DRL algorithm : `PPO`
# - As an environment in which agents will interact, we will use [OpenAI Gymnasium library](https://gymnasium.farama.org/)
#
# reference
# - https://github.com/openai/baselines/blob/master/baselines/ppo1/pposgd_simple.py

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


# ## Objective funtion of `PPO`  
# >**Actor** <br>
# $L(\theta) \ = \ $<font color=brick>$L^{\text{CLIP}}(\theta)$</font>$
# \  + \  L^{\text{S}}(\theta)  $
# <br><br><br>
# $\ \ \ \ $ <font color=brick>$L^{\text{CLIP}}(\theta)$</font>$
# \ = \, \mathbb{E}_{s \sim \rho_{\theta_{\text{old}}},\,
# a \sim \pi_{\theta_{\text{old}}}}
# \big[\,$
# <font color=green>$\text{min} \big($</font>$\
# r(\theta)\,A_{\theta_{\text{old}}}(s,a)\ , \
# $<font color=blue>$\text{clip}(r(\theta),\,1-\epsilon,\,1+\epsilon)$</font>$
# \,A_{\theta_{\text{old}}}(s,a) 
# \ $<font color=green>$\big)$</font>$\,\big]$
# <br><br>
# $\ \ \ \ $where $r(\theta)= \frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$
# $\ \ $ and $\ \ A_{\theta_{\text{old}}}(s,a) \approx (G^{\,\text{(n)}} - V_{\phi_{\text{old}}}(s))$.
# <br><br><br>
# $\ \ \ \ $$L^{\text{S}}(\theta) = 
# \mathbb{E}_{s \sim \rho_{\theta_{\text{old}}},\,
# a \sim \pi_{\theta_{\text{old}}}}
# \big[\,S\big(\, \pi_{\theta}(\cdot|s)\, \big)\big]
# =\mathbb{E}_{s \sim \rho_{\theta_{\text{old}}},\,
# a \sim \pi_{\theta_{\text{old}}}}
# \big[\,
# \mathbb{E}_{a \sim \pi_{\theta_{\text{old}}}(s)}
# [\, -\text{log}\,\pi_{\theta}(a|s)\,]
# \big]
# =\mathbb{E}_{s \sim \rho_{\theta_{\text{old}}},\,
# a \sim \pi_{\theta_{\text{old}}}}
# \big[-\text{log}\,\pi_{\theta}(a|s)
# \,\big]$
#  <br><br><br>
#  **Critic** <br>
# $L(\phi) = \, \mathbb{E}_{\pi_{\theta_{\text{old}}}} \big[\,
# \big(G^{\,\text{(n)}}-V_{\phi}(s)\big)^2 \,\big]
# \approx \, \frac{1}{|B|}\sum_{i \in B} 
# \big(G^{\,\text{(n)}}_i-V_{\phi}(s_i)\big)^2$
#  <br><br>
# where $B$ is the mini-batch.
#

# +
class Actor_net(nn.Module):
    def __init__(self, state_dim, action_dim, action_high):
        super().__init__()
        self.action_high = torch.nn.Parameter(torch.FloatTensor(action_high), requires_grad=False)
        hidden_space1 = 16
        hidden_space2 = 32

        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_space1),
            nn.ReLU(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.ReLU() )
        
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_dim),
            nn.Tanh()) # use 'Tanh' in the output layer for the bounded policy
        
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

# Agent that will interact with environment and be trained
class PPO:
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
    def update(self, actor_optimizer, critic_optimizer, transition, Horizon, epoch, batch_size, 
               clip_eps, entropy_coef, device):
        
        self.Actor = self.Actor.to(device)
        self.Critic = self.Critic.to(device)
        
        for _ in range(epoch):
            # randomly shuffle train data(transition)
            np.random.shuffle(transition)
            for i in range(int(Horizon/batch_size)):
                batch = transition[i*batch_size:(i+1)*batch_size]
                
                value_arr,next_value_arr, state_arr,action_arr,reward_arr,done_arr, old_log_prob_arr = map(np.asarray, zip(*batch))

                value_arr = torch.FloatTensor(value_arr).to(device)
                next_value_arr = torch.FloatTensor(next_value_arr).to(device)
                state_arr = torch.FloatTensor(state_arr).to(device)
                action_arr = torch.FloatTensor(action_arr).to(device)
                reward_arr = torch.FloatTensor(reward_arr).to(device)
                done_arr = torch.FloatTensor(done_arr).to(device)
                old_log_prob_arr = torch.FloatTensor(old_log_prob_arr).to(device)

                # update Actor
                advantage_arr = reward_arr + self.gamma*next_value_arr*(1-done_arr) - value_arr

                action_mean_arr, action_std_arr = self.Actor(state_arr)
                distrib = Normal(action_mean_arr, action_std_arr + 1e-8)
                new_log_prob_arr = distrib.log_prob(action_arr)

                ratio = torch.exp(new_log_prob_arr - old_log_prob_arr).squeeze(axis=-1)
                clipped_ratio = torch.clamp(ratio, 1-clip_eps, 1+clip_eps)
                surrogate = torch.min(advantage_arr*ratio, advantage_arr*clipped_ratio)
                entropy = (-1) * new_log_prob_arr

                actor_loss = torch.mean( (-1)*(surrogate + entropy_coef*entropy) )
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update Critic
                target_arr = reward_arr + self.gamma*next_value_arr*(1-done_arr)
                predict = self.Critic(state_arr).squeeze(axis=-1)
                
                critic_loss = torch.mean( (target_arr-predict)**2 )
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                
        self.Actor = self.Actor.to('cpu')
        self.Critic = self.Critic.to('cpu')    
    
    # train agent
    def train(self, env, evaluate_num, actor_initial_lr, actor_final_lr, critic_initial_lr, critic_final_lr,
              Horizon, epoch, batch_size, time_step, clip_eps, entropy_coef, device):
        start = time.time()
        reward_list = []
        
        actor_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=actor_initial_lr)
        critic_optimizer = torch.optim.Adam(self.Critic.parameters(), lr=critic_initial_lr)
        
        transition = [] # store transition [value, next_value, state, action, reward, terminated, log_prob]
        t = 0 # current time step
        while t < time_step :
            # new episode start
            actor_lr      = linear_schedule(t, time_step//2, actor_initial_lr, actor_final_lr)
            actor_optimizer.learning_rate = actor_lr
            critic_lr     = linear_schedule(t, time_step//2, critic_initial_lr, critic_final_lr)
            critic_optimizer.learning_rate = critic_lr
            
            state, info = env.reset()
            done = False
            
            # interact with environment and train
            while not done:
                action, log_prob = self.get_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                t += 1  
                value = self.Critic(torch.FloatTensor(state))[0]
                next_value = self.Critic(torch.FloatTensor(next_state))[0]
                transition.append([value.detach().numpy(),next_value.detach().numpy(),state, action, reward,terminated,log_prob.detach().numpy()])
                
                done = terminated or truncated
                
                if t%Horizon==0 :
                    self.update(actor_optimizer, critic_optimizer, transition, Horizon, epoch, batch_size,
                                clip_eps, entropy_coef, device)
                    transition = [] # reset transition buffer
                    
                    # evaluate the current actor                    
                    reward = self.test(env, evaluate_num)
                    reward_list.append(reward)
                    
                state = next_state
        
        end = time.time()
        print(f'Training time : {(end-start)/60:.2f}(min)')

        return reward_list
    
    # evaluate current policy
    # return average reward value over the several episodes
    def test(self, env, evaluate_num=10):

        reward_list = []
        with torch.no_grad():
            for episode in range(evaluate_num):
                # new episode start
                done = False
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
# hyperparameter
time_step =            # the number of time-steps that agent will interact with environment
evaluate_num    = 10   # the number of episodes that agent will be evaluated

Horizon =           # temporal buffer size (store Horizon number of transitions and train) 
epoch   =           # train epoch for each Horizion
batch_size =        # train batch_size
clip_eps   =        # clipping epsilon
entropy_coef  = 0   # coefficient of the entropy loss term

actor_initial_lr =       # starting actor learning rate
actor_final_lr   =       # final actor learning rate
critic_initial_lr =      # starting critic learning rate
critic_final_lr   =      # final critic learning rate
gamma =      # gamma : reward discount rate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
repeat = 3  # repeat same experiment for the reliable result
# -

# train : PPO
reward_list =[]
for i in range(repeat):
    # control randomness for reproducibility
    seed = 100*(i+1)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    env = gym.make('InvertedPendulum-v5', render_mode='rgb_array')
    agent = PPO(env.observation_space.shape[0], env.action_space.shape[0],env.action_space.high, gamma)
    reward = agent.train( env, evaluate_num, actor_initial_lr, actor_final_lr, critic_initial_lr, critic_final_lr,
                          Horizon, epoch, batch_size, time_step, clip_eps, entropy_coef, device )    
    
    reward_list.append(reward)

# ## **Results**

# ### Plot the results

# create folder to save the result
save_folder = './result'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# +
save_file = 'PPO.png'
save_path  = os.path.join(save_folder, save_file)

plot(reward_list, 'PPO', save_path=save_path)
# -

# ### Test the trained agent and save it into a GIF.

'''display = Display(visible=0, size=(140, 90))
display.start()

env = gym.make('InvertedPendulum-v5', render_mode='rgb_array')
play = play_and_save(env, agent, 'PPO', seed=8)

display.stop()
Image(open(play,'rb').read())'''


