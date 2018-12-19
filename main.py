import gym
import random
import torch
import numpy as np

from tqdm import tqdm
from unityagents import UnityEnvironment
from collections import deque

from agent import Agent

env = UnityEnvironment(file_name='app/Reacher.x86_64')
print('Loaded env')

# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
print(brain)

# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# Number of agents
num_agents = len(env_info.agents)
print('Number of agents:{}'.format(num_agents))

# Number of actions
action_size = brain.vector_action_space_size
print('Number of actions:{}'.format(action_size))

# Size of each action
action_size = brain.vector_action_space_size
print("Size of each action:{}".format(action_size))

# Examine the state space (20 of them)
states = env_info.vector_observations
state_size = states.shape[1]
print("There are {} agents. Each observes a state with length: {}".format(states.shape[0], state_size))
print("The state for the first agent looks like:", states[0])

scores = np.zeros(num_agents)
seed = 1
agent = Agent(state_size=state_size, action_size=action_size, seed=seed, num_agents=num_agents)


def dqn(n_episodes=20, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """
    Deep Q-Learning

    Params
    ======
        n_episodes (int): max number of training episodes
        max_t (int): max number of timesteps per episode
        eps_start (float): start value of epsilon, for epsilon-greedy action selection
        eps_end (float): min value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in tqdm(range(1, n_episodes+1)):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = []                        # list containing scores from each episode
        for t in range(max_t):
            actions = agent.act(state=states, eps=eps)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += env_info.rewards
            states = next_states
            if np.any(dones):
                break
    return scores

scores = dqn()
env.close()