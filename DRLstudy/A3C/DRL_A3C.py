# -*- coding: utf-8 -*-
# ### **Deep Reinforcement Learning(DRL)** Code
#
# ---
#
# - DRL aims to **solve MDP**(Markov Decision Process) problems. That is, DRL aims to **find an optimal policy**.
# - In this notebook, we aims to implement the following DRL algorithm : `A3C`
# - As an environment in which agents will interact, we will use [OpenAI Gymnasium library](https://gymnasium.farama.org/)

# #### Reference : 
#
# - https://velog.io/@ss-hj/PyTorch에서-autograd-동작-이해하기 (backward() 관련)
# - https://velog.io/@steadycode/PyTorch-Optimizer-1 (optimizer share 관련)

# ## Requirement

# +
# #!pip install gymnasium
# #!pip install gymnasium[mujoco]
# #!pip install opencv-python==4.8.0.74

# +
import gymnasium as gym

import os
import torch
import random
import numpy as np

import torch.multiprocessing as mp

from pyvirtualdisplay import Display
from IPython.display import Image
# -

from source import A3C, plot, play_and_save, SharedAdam

# ## **Train** Agent

# +
# hyperparameter for A3C
process_num =     # the number of local agent

max_episode =          # the number of episodes that agent will be trained
update_period =        # update period(time-step) of agent
evaluate_period = 5    # episode period that agent's policy will be evaluated
evaluate_num    = 10   # the number of episodes that agent will be evaluated

actor_initial_lr =         # starting actor learning rate
actor_final_lr   =         # final actor learning rate
critic_initial_lr =        # starting critic learning rate
critic_final_lr   =        # final critic learning rate
gamma =      # gamma : reward discount rate

repeat = 3   # repeat same experiment for the reliable result
# -

if __name__ == "__main__":
    mp.set_start_method('spawn') # Must be spawn
    print("MultiProcessing start method:", mp.get_start_method())

    reward_list =[]
    for i in range(repeat):
        # control randomness for reproducibility
        seed = 100*(i+1)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        env = gym.make('InvertedPendulum-v5', render_mode='rgb_array')
        # initialize GLOBAL agent
        global_agent = A3C(env.observation_space.shape[0], env.action_space.shape[0],env.action_space.high, gamma)
        env.close()

        # share the GLOBAL agent
        global_agent.Actor.share_memory()
        global_agent.Critic.share_memory()

        # share the GLOBAL agent's optimizer
        global_optim_actor  = SharedAdam(global_agent.Actor.parameters(), lr=actor_initial_lr)
        global_optim_critic = SharedAdam(global_agent.Critic.parameters(), lr=critic_initial_lr)

        # setting for multi-processing
        processes = []
        global_reward = mp.Array('d',[-10 for i in range(int(max_episode*process_num/evaluate_period))])
        global_episode = mp.Value('i',1)
        
        # train GLOBAL agent from multiple local agents (multi-processing)
        for rank in range(process_num):
            env = gym.make('InvertedPendulum-v5', render_mode='rgb_array')
            agent = A3C(env.observation_space.shape[0], env.action_space.shape[0],env.action_space.high, gamma)
            p = mp.Process(target=agent.train, args=(env, max_episode, evaluate_period, evaluate_num, update_period,
                             actor_initial_lr, actor_final_lr, critic_initial_lr, critic_final_lr,
                             global_agent, global_optim_actor, global_optim_critic, rank, global_reward, global_episode ))
            p.start()
            processes.append(p)
            env.close()
        for p in processes:
            p.join()

        reward= [global_reward[i] for i in range(int(max_episode*process_num/evaluate_period))]
        reward_list.append(reward)
        # checking the missed value (values are often missed)
        print(f'Missed : {reward.count(-10)}, where : {np.where(np.array(reward) == -10)[0]}')

# ## **Results**

# ### Plot the results

# create folder to save the result
save_folder = './result'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# +
save_file = 'A3C.png'
save_path  = os.path.join(save_folder, save_file)

plot(reward_list, 'A3C', save_path=save_path)
# -

# ### Test the trained agent and save it into a GIF.

'''display = Display(visible=0, size=(140, 90))
display.start()

env = gym.make('InvertedPendulum-v5', render_mode='rgb_array')
play = play_and_save(env, agent, 'A3C', seed=8)

#display.stop()
Image(open(play,'rb').read())'''
