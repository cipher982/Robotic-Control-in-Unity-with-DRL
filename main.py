import gym
import random
import torch
import numpy as np
import sys

from tqdm import tqdm
from unityagents import UnityEnvironment
from collections import deque
import matplotlib.pyplot as plt

from ddpg_agent import Agent, ReplayBuffer, OUNoise

env = UnityEnvironment(file_name='app/Reacher.app')
print('Loaded env')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))

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

PRINT_EVERY = 1
NUM_AGENTS = 20
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 1028
SEED = 72

scores = np.zeros(num_agents)
agent = Agent(state_size=state_size, action_size=action_size,  num_agents=NUM_AGENTS, seed=SEED)
#noises = [OUNoise(action_size, SEED + i) for i in range(NUM_AGENTS)]
#[noise.reset() for noise in noises]

def ddpg(n_episodes=200, 
    num_agents=NUM_AGENTS, 
    max_t=1000, 
    eps_start=1.0, 
    eps_end=0.01, 
    eps_decay=0.995):
    """
    Deep Deterministic Policy Gradient

    Params
    ======
        n_episodes (int): max number of training episodes
        num_agents (int): count of agents to run
        max_t (int): max number of steps per episode
        eps_start (float): start value of epsilon, for epsilon-greedy action selection
        eps_end (float): min value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores_deque = deque(maxlen=PRINT_EVERY)
    scores_all = []
    steps = 0
    #max_steps = 1000

    for episode_ix in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        scores = np.zeros(NUM_AGENTS)
        while True:
            actions = agent.act(states)                        # select an action (for each agent)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            agent.step(states, actions, rewards, next_states, dones)
            scores += rewards                                  # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break
        scores_deque.append(np.mean(scores))
        scores_all.append(np.mean(scores))

        #print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode_ix, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_v1.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_v1.pth')
        if episode_ix % PRINT_EVERY == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode_ix, np.mean(scores_deque)))

    return scores_all

scores = ddpg()
env.close()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
