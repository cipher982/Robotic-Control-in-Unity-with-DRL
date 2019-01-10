import gym
import random
import torch
import numpy as np

from tqdm import tqdm
from unityagents import UnityEnvironment
from collections import deque

from ddpg_agent import Agent, ReplayBuffer, OUNoise

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

def ddpg(n_episodes=200, 
    num_agents=num_agents, 
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
    scores_deque = deque(maxlen=print_every)
    scores = []
    steps = 0

    for episode_ix in range(n_episodes):
        state = env.reset(train_mode=True)[brain_name].vector_observations
        agents.reset()
        score = 0
        for step in range(max_steps):
            for i, noise in enumerate(noises):
                action = agents.act(state) + noise.sample()
                action = np.clip(action, -1, 1)
                curr_env = env.step(action)['ReacherBrain']]
                next_state = curr_env.vector_observations
                reward = curr_env.rewards[0]
                done = curr_env.local_done[0]
                state = next_state
                agents.add(state, action, reward, next_state, done)
                steps += 1
                if steps > 20:
                    agent.learn_from_buffer(train_counter=10)
                    steps = 0
                score += reward
                if done:
                    break
        scores_deque.append(score)
        scores.append(score)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agents.actor_local.state_dict(), 'checkpoint_actor_v1.pth')
        torch.save(agents.critic_local.state_dict(), 'checkpoint_critic_v1.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        return scores

scores = ddpg()
env.close()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
