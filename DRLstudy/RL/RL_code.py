#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
#
# # **Reinforcement Learning(RL)** Code
#  
# ---
#  
#  - RL aims to **solve MDP**(Markov Decision Process) problems. That is, RL aims to **find an optimal policy**.
#  - In this notebook, we aims to implement the following RL algorithms : `Monte Carlo`, `Sarsa`, `Q-learning`, `Expected Sarsa`, `Double Q-learning`, 
#  - As an environment in which agents will interact, we will use [OpenAI Gymnasium library](https://gymnasium.farama.org/)
#   

# %% [markdown]
# ## Requirements

# %%
# #!pip install gymnasium
# #!pip install opencv-python==4.8.0.74


# %%
import gymnasium as gym
import numpy as np
import random

import matplotlib.pyplot as plt
import PIL.Image
from IPython.display import Image


# %% [markdown]
# ## Utils

# %%
# epsilon scheduler (linear)
# decay epsilon linearly from 'initial_eps' to 'final_eps'
def epsilon_schedule(episode, max_episode, initial_eps, final_eps):
    start, end = initial_eps, final_eps
    if episode < max_episode:
        return (start*(max_episode-episode) + end*episode) / max_episode
    else:
        return end


# %%
# plot the experiment results (rewards, episode length)
# given the list of rewards/eipsode length, plot the highest, lowest, and mean graph.
def plot(rewards, episode, title:str, save_path=None):  
      
    plt.figure(figsize=[7,3], dpi=300)
    plt.suptitle(title, fontsize=10)
    
    # plot reward
    plt.subplot(1,2,1)
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
    
    # plot episode length
    plt.subplot(1,2,2)
    high_episode= np.max(episode , axis= 0)
    low_episode = np.min(episode , axis= 0)
    mean_episode= np.mean(episode , axis= 0)
    
    plt.xlabel('Episodes', fontsize=7)
    plt.ylabel('Episode length', fontsize=7)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.grid(linewidth=.1)

    x= np.arange(1, len(episode[0])+1)
    plt.plot(x, high_episode, 'b-', linewidth=.1, alpha=0.2)
    plt.plot(x, low_episode, 'b-', linewidth=.1, alpha=0.2)
    plt.plot(x, mean_episode, 'b-', linewidth=.2)
    
    if save_path!=None:
        plt.savefig(save_path, format='png')
        
    plt.show()


# %%
# create a GIF that shows how agent interacts with environment (play 1 episode)
# given the trained agent and environment, the agent interact with the environment and save into a GIF file 
def play_and_save(env, Q_table, name='', seed=None):
    render_images = []
    total_reward = 0

    state, _ = env.reset(seed=seed)
    image_array = env.render()
    render_images.append(PIL.Image.fromarray(image_array))

    terminated, truncated = False, False

    # episode start
    while not terminated and not truncated:
        action = np.argmax(Q_table[state]) # get action
        state, reward, terminated, truncated, _ = env.step(action) # step action
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


# %% [markdown]
# ## Q-table update method of `Sarsa`
#  > $Q(S_t,A_t) \longleftarrow Q(S_t,A_t) + \alpha \, [
#  $<font color=blue>$R_{t+1} + \gamma\, Q(S_{t+1},A_{t+1})$</font>
#   $ - \, Q(S_t,A_t)]$

# %%
class Sarsa:    
    def __init__(self,  state_dim, action_dim, alpha, gamma):
        self.Q_table = np.zeros((state_dim, action_dim))
        
        self.alpha = alpha
        self.gamma = gamma
        
        self.state_dim =  state_dim
        self.action_dim = action_dim
        
    def update_Q_table(self,  state, action, reward, next_state, next_action):
        self.Q_table[state, action] += self.alpha*(reward + self.gamma*self.Q_table[next_state, next_action] - self.Q_table[state, action])
    
    def get_action(self,  state, epsilon): # Epsilon-greedy policy
        prob = random.random()
        if prob < epsilon:
            # random action with probability epsilon (Exploration)
            return np.random.randint(self.action_dim)
        else:
            # greedy action with probability (1 - epsilon) (Exploitation)
            return np.argmax(self.Q_table[state])
        
    def train(self, env, max_episode, initial_epsilon, final_epsilon):
        
        total_rewards = []
        episode_lengths = []
        
        for episode in range(max_episode):
            # new episode start
            terminated = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            
            # linearly decay epsilon through episodes
            epsilon = epsilon_schedule(episode, max_episode//2, initial_epsilon, final_epsilon)

            state, info = env.reset() # initialize the environment
            action = self.get_action(state, epsilon) # get action
            
            while not terminated and not truncated:
                next_state, reward, terminated, truncated, info = env.step(action) # take action
                next_action = self.get_action(next_state, epsilon)
                self.update_Q_table(state, action, reward, next_state, next_action)

                episode_reward += reward
                episode_length += 1
                state = next_state
                action = next_action
                
            # episode finished
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
        return total_rewards, episode_lengths


# %% [markdown]
# ## Q-table update method of `Q-learning`
# >  $Q(S_t,A_t) \longleftarrow Q(S_t,A_t) + \alpha \, [
#  $<font color=blue>$R_{t+1} + \gamma\, \underset{a}{max}\, Q(S_{t+1},a)$</font>
#   $ - \, Q(S_t,A_t)]$
#  
#  

# %%
class Q_learning:    
    def __init__(self,  state_dim, action_dim, alpha, gamma):
        self.Q_table = np.zeros((state_dim, action_dim))
        
        self.alpha = alpha
        self.gamma = gamma
        
        self.state_dim =  state_dim
        self.action_dim = action_dim
        
    def update_Q_table(self,  state, action, reward, next_state):
        self.Q_table[state, action] += self.alpha*(reward + self.gamma*np.max(self.Q_table[next_state]) - self.Q_table[state, action])
        
    def get_action(self,  state, epsilon): # Epsilon-greedy policy
        prob = random.random()
        if prob < epsilon:
            # random action with probability epsilon (Exploration)
            return np.random.randint(self.action_dim)
        else:
            # greedy action with probability (1 - epsilon) (Exploitation)
            return np.argmax(self.Q_table[state])
        
    def train(self, env, max_episode, initial_epsilon, final_epsilon):
        
        total_rewards = []
        episode_lengths = []
        
        for episode in range(max_episode):
            # new episode start
            terminated = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            # linearly decay epsilon through episodes
            epsilon = epsilon_schedule(episode, max_episode//2, initial_epsilon, final_epsilon)

            state, info = env.reset() # initialize the environment
            
            while not terminated and not truncated:
                action = self.get_action(state, epsilon) # get action
                next_state, reward, terminated, truncated, info = env.step(action) # take action
                self.update_Q_table(state, action, reward, next_state)
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
            # episode finished
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
        return total_rewards, episode_lengths


# %% [markdown]
# ## Q-table update method of `Monte Carlo`
# >  $Q(S_t,A_t) \longleftarrow Q(S_t,A_t) + \alpha \, [
# $<font color=blue>$G_t$</font>
#  $ - \, Q(S_t,A_t)]$

# %%
class MonteCarlo:    
    def __init__(self,  state_dim, action_dim, alpha, gamma):
        self.Q_table = np.zeros((state_dim, action_dim))
        
        self.alpha = alpha
        self.gamma = gamma
        
        self.state_dim =  state_dim
        self.action_dim = action_dim
        
    def update_Q_table(self,  state, action, discounted_return):
        self.Q_table[state, action] += self.alpha*(discounted_return - self.Q_table[state, action])
        
    def get_action(self,  state, epsilon): # Epsilon-greedy policy
        prob = random.random()
        if prob < epsilon:
            # random action with probability epsilon (Exploration)
            return np.random.randint(self.action_dim)
        else:
            # greedy action with probability (1 - epsilon) (Exploitation)
            return np.argmax(self.Q_table[state])
        
    def train(self, env, max_episode, initial_epsilon, final_epsilon):
        
        total_rewards = []
        episode_lengths = []
        
        for episode in range(max_episode):
            # new episode start 
            state_list=   []
            action_list=  []
            reward_list = []
            
            terminated = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            # linearly decay epsilon through episodes
            epsilon = epsilon_schedule(episode, max_episode//2, initial_epsilon, final_epsilon)

            state, info = env.reset() # initialize the environment
            
            while not terminated and not truncated:
                action = self.get_action(state, epsilon) # get action
                next_state, reward, terminated, truncated, info = env.step(action) # take action
                state_list.append(state)
                action_list.append(action)
                reward_list.append(reward)
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
            # episode finished
            G = 0
            for i in range(episode_length-1,-1,-1):
                G = self.gamma*G + reward_list[i]
                self.update_Q_table(state_list[i], action_list[i], G)

            total_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
        return total_rewards, episode_lengths


# %% [markdown]
# ## Q-table update method of `Expected Sarsa`
# >  $Q(S_t,A_t) \longleftarrow Q(S_t,A_t) + \alpha \, [
# $<font color=blue>$R_{t+1} + \gamma\, \mathbb{E}_{\pi}[Q(S_{t+1},A_{t+1})| S_{t+1}]$</font>
#  $ - \, Q(S_t,A_t)]$

# %%
class Expected_Sarsa:    
    def __init__(self,  state_dim, action_dim, alpha, gamma):
        self.Q_table = np.zeros((state_dim, action_dim))
        
        self.alpha = alpha
        self.gamma = gamma
        
        self.state_dim =  state_dim
        self.action_dim = action_dim
        
    def update_Q_table(self,  state, action, reward, next_state, epsilon):
        expected_value= 0
        for i in range(self.action_dim):
            expected_value+= (epsilon/self.action_dim)* self.Q_table[next_state, i]
        expected_value+= (1-epsilon)* self.Q_table[next_state, np.argmax(self.Q_table[next_state])]

        self.Q_table[state, action] += self.alpha * (reward + self.gamma*expected_value - self.Q_table[state, action])
    
    def get_action(self,  state, epsilon): # Epsilon-greedy policy
        prob = random.random()
        if prob < epsilon:
            # random action with probability epsilon (Exploration)
            return np.random.randint(self.action_dim)
        else:
            # greedy action with probability (1 - epsilon) (Exploitation)
            return np.argmax(self.Q_table[state])
        
    def train(self, env, max_episode, initial_epsilon, final_epsilon):
        
        total_rewards = []
        episode_lengths = []
        
        for episode in range(max_episode):
            # new episode start
            terminated = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            # linearly decay epsilon through episodes
            epsilon = epsilon_schedule(episode, max_episode//2, initial_epsilon, final_epsilon)

            state, info = env.reset() # initialize the environment
            
            while not terminated and not truncated:
                action = self.get_action(state, epsilon) # get action
                next_state, reward, terminated, truncated, info = env.step(action) # take action
                self.update_Q_table(state, action, reward, next_state, epsilon)

                episode_reward += reward
                episode_length += 1
                state = next_state
                
            # episode finished
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
        return total_rewards, episode_lengths


# %% [markdown]
# ## Q-table update method of `Double Q-learning`
# >  $Q_1(S_t,A_t) \longleftarrow Q_1(S_t,A_t) + \alpha \, [
# $<font color=blue>$R_{t+1} + \gamma\, Q_2(S_{t+1},\underset{a}{argmax}\,Q_1(S_{t+1},a))$</font>
#  $ - \, Q_1(S_t,A_t)]$
#  <br>
#  $Q_2(S_t,A_t) \longleftarrow Q_2(S_t,A_t) + \alpha \, [
# $<font color=blue>$R_{t+1} + \gamma\, Q_1(S_{t+1},\underset{a}{argmax}\,Q_2(S_{t+1},a))$</font>
#  $ - \, Q_2(S_t,A_t)]$

# %%
class Double_Q_learning:    
    def __init__(self,  state_dim, action_dim, alpha, gamma):
        self.Q_table_1 = np.zeros((state_dim, action_dim))
        self.Q_table_2 = np.zeros((state_dim, action_dim))
        
        self.alpha = alpha
        self.gamma = gamma
        
        self.state_dim =  state_dim
        self.action_dim = action_dim
        
    def update_Q_table(self,  state, action, reward, next_state):
        tmp = random.random()
        if tmp < 0.5:
            self.Q_table_1[state, action] += self.alpha * (reward + self.gamma*self.Q_table_2[next_state,np.argmax(self.Q_table_1[next_state])]  - self.Q_table_1[state, action])
        else:
            self.Q_table_2[state, action] += self.alpha * (reward + self.gamma*self.Q_table_1[next_state,np.argmax(self.Q_table_2[next_state])]  - self.Q_table_2[state, action])        
        
    def get_action(self,  state, epsilon): # Epsilon-greedy policy
        prob = random.random()
        if prob < epsilon:
            # random action with probability epsilon (Exploration)
            return np.random.randint(self.action_dim)
        else:
            # greedy action with probability (1 - epsilon) (Exploitation)
            return np.argmax(self.Q_table_1[state]+self.Q_table_2[state])    
        
    def train(self, env, max_episode, initial_epsilon, final_epsilon):
        
        total_rewards = []
        episode_lengths = []
        
        for episode in range(max_episode):
            # new episode start
            terminated = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            # linearly decay epsilon through episodes
            epsilon = epsilon_schedule(episode, max_episode//2, initial_epsilon, final_epsilon)

            state, info = env.reset() # initialize the environment
            
            while not terminated and not truncated:
                action = self.get_action(state, epsilon) # get action
                next_state, reward, terminated, truncated, info = env.step(action) # take action
                self.update_Q_table(state, action, reward, next_state)
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
            # episode finished
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
        return total_rewards, episode_lengths


# %% [markdown]
# ## **Train** Agent

# %%
# hyperparameter
max_episode = 
initial_epsilon = 1.0
final_epsilon =          
alpha =      # alpha : kind of learning rate (value: 0~1 )
gamma =      # gamma : reward discount rate  (value: 0~1 )

repeat = 3 # repeat same experiment for the reliable result


# %%
# train : Sarsa
env = gym.make('Taxi-v3', render_mode='rgb_array')
sarsa_agent = Sarsa(env.observation_space.n, env.action_space.n, alpha, gamma)

sarsa_reward_list, sarsa_episode_list =[], []
for i in range(repeat):
    reward, episode = sarsa_agent.train(env, max_episode, initial_epsilon, final_epsilon)
    sarsa_reward_list.append(reward)
    sarsa_episode_list.append(episode)


# %%
# train : Q-learning
env = gym.make('Taxi-v3', render_mode='rgb_array')
Q_agent = Q_learning(env.observation_space.n, env.action_space.n, alpha, gamma)

Q_reward_list, Q_episode_list =[], []
for i in range(repeat):
    reward, episode = Q_agent.train(env, max_episode, initial_epsilon, final_epsilon)
    Q_reward_list.append(reward)
    Q_episode_list.append(episode)


# %%
# train : Monte Carlo
env = gym.make('Taxi-v3', render_mode='rgb_array')
MC_agent = MonteCarlo(env.observation_space.n, env.action_space.n, alpha, gamma)

MC_reward_list, MC_episode_list =[], []
for i in range(repeat):
    reward, episode = MC_agent.train(env, max_episode, initial_epsilon, final_epsilon)
    MC_reward_list.append(reward)
    MC_episode_list.append(episode)


# %%
# train : Expected Sarsa
env = gym.make('Taxi-v3', render_mode='rgb_array')
Expected_Sarsa_agent = Expected_Sarsa(env.observation_space.n, env.action_space.n, alpha, gamma)

Expected_Sarsa_reward_list, Expected_Sarsa_episode_list =[], []
for i in range(repeat):
    reward, episode = Expected_Sarsa_agent.train(env, max_episode, initial_epsilon, final_epsilon)
    Expected_Sarsa_reward_list.append(reward)
    Expected_Sarsa_episode_list.append(episode)


# %%
# train : Double Q-learning
env = gym.make('Taxi-v3', render_mode='rgb_array')
Double_Q_agent = Double_Q_learning(env.observation_space.n, env.action_space.n, alpha, gamma)

Double_Q_reward_list, Double_Q_episode_list =[], []
for i in range(repeat):
    reward, episode = Double_Q_agent.train(env, max_episode, initial_epsilon, final_epsilon)
    Double_Q_reward_list.append(reward)
    Double_Q_episode_list.append(episode)

# %% [markdown]
# ## **Results**
#
#  ### Plot the results

# %%
plot(sarsa_reward_list, sarsa_episode_list, 'SARSA')


# %%
plot(Q_reward_list, Q_episode_list, 'Q-learning')


# %%
plot(MC_reward_list, MC_episode_list, 'Monte Carlo')


# %%
plot(Expected_Sarsa_reward_list, Expected_Sarsa_episode_list, 'Expected Sarsa')


# %%
plot(Double_Q_reward_list, Double_Q_episode_list, 'Double Q-learning')

# %% [markdown]
# ### Test the trained agent and save it into a GIF.

# %%
env = gym.make('Taxi-v3', render_mode='rgb_array')


# %%
sarsa_play = play_and_save(env, sarsa_agent.Q_table, 'Sarsa', seed=8)
print(sarsa_agent.Q_table)
Image(open(sarsa_play,'rb').read())


# %%
Q_play = play_and_save(env, Q_agent.Q_table, 'Q', seed=8)
print(Q_agent.Q_table)
Image(open(Q_play,'rb').read())


# %%
MC_play = play_and_save(env, MC_agent.Q_table, 'MC', seed=8)
print(MC_agent.Q_table)
Image(open(MC_play,'rb').read())


# %%
Expected_Sarsa_play = play_and_save(env, Expected_Sarsa_agent.Q_table, 'Expected_Sarsa', seed=8)
print(Expected_Sarsa_agent.Q_table)
Image(open(Expected_Sarsa_play,'rb').read())


# %%
Double_Q_play = play_and_save(env, Double_Q_agent.Q_table_1+Double_Q_agent.Q_table_2, 'Double_Q', seed=8)
print(Double_Q_agent.Q_table_1+Double_Q_agent.Q_table_2)
Image(open(Double_Q_play,'rb').read())


# %%
