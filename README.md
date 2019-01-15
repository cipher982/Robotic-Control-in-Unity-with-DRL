[image_environment_sample]: https://raw.githubusercontent.com/cipher982/Robotic-Control-in-Unity-with-DRL/master/images/arm_environment_sample.gif

[image_training_performance]: https://raw.githubusercontent.com/cipher982/Robotic-Control-in-Unity-with-DRL/master/images/training_performance.png

# Controlling Multiple Robotic Arms with DDPG Reinforcement Learning
![image_environment_sample]

Like my last project https://github.com/cipher982/DRL-DQN-Model this one is built upon ideas of deep reinforcement learning, but with two additional features: dual actor/critic neural networks and multiple agents running simaltatneuously.

The specific algorithm we will use is referred to as **Deep Deterministic Policy Gradient learning**, or **DDPG**. This is a *model-free, off-policy, actor-critic algorithm* that excels at learning in high-dimensional (continuous) action spaces.

#### Why Actor-Critic?
The idea using two separate networks (actor/critic) enables the model to generalize and estimate reward values (using the critic) from the chosen action of the actor, therefore reducing the need for as much exploration of state/value combinations.

#### Why multiple agents?
Well for one, this environment contains 20 agents that must be controlled, but in general it is a good idea to learn from varied and indipendant sources. Given that they exist in the same environment and have the same state/action possiblities we can assume the reward function operates the same across them all. Using the idea of stored memory to sample from (the ReplayBuffer class from my last project) we can just toss all the experiences from the agents into one pile, and sample from it to learn optimal policies.

#### Incorporating DDPG to this environment
This environment, built in Unity, contains 20 parallel agents (robot arms) that each have 33 state values for: position, rotation, velocity, and angular velocity. Each robot arm has two joints for a total of 4 actions for each arm. 

The goal is to follow the green balloon-looking objects floating around the space, keeping the ends of the arms within that volume. There are no negative rewards in this particular case, just a positive value of +0.1 for each action within the balloon, and no reward if it misses. The environment is considered sovled if the agent can reach an average of +30 reward over 100 consecutive episodes.

#### A quick overview of the learning process
1. As usual with reinforcement learning we **begin with random noise** values to begin the exploration process. 
2. With each step of taking an action we **add to the experiences memory**.
3. If enough experiences get loaded in to the memory we can start the learning steps:
    a. **Update the Critic**: compute Q-targets, compute Critic loss, minimize the loss
    b. **Update the Actor**: compute Actor loss, minimize the loss
    c. **Update the target networks** (as in the DQN algorithm, we like to hold separate target/local models and periodically update the target to enable more steady aiming towards the goal and stable learning overall)
4. After an episode is done, check the most recent 100 episodes. If the average (mean) reward is 30 or higher, consider it solved!

#### Technical details of the setup & hyperparameters
Below are the values I used for the most recent model. These are a combination of default values from the DDPG paper, Udacity excersize values, and some I changed after expirementation. The most different I would say is BATCH_SIZE, which I kept increasing as I knew I had the GPU memory to sustain it, and did not negatively impact the training process. 
- BUFFER_SIZE = int(1e5)
- BATCH_SIZE = 1028
- GAMMA = 0.99
- LR_ACTOR = 1e-3
- LR_CRITIC = 1e-3
- SEED = 72
- TAU = 1e-3
- WEIGHT_DECAY = 0
- Model Weights:
    - Actor(Layer1=128, Layer2=64) 
    - Critic(Layer1=128, Layer2=64)

#### Performance
After 200 episodes the reward was beginning to approach the mid-to-high 30s, with the past 100 episodes reaching an average of 30 right before the 200 mark. In the image below you can see how the performance begins to level off around the 125 episode mark.

![image_training_performance]

#### Future Work
In my expirementation (due to slow training times) I limited the max episodes to 200. While performance seems plateu around 125, I did not play around with many of the hyperparameters too much and there is likely room for optimization. There are also many new algorithms that have been introduced recently such as: DDQN, A3C, A2C, Dueling DDQN, and even a method [combining them together called Rainbow](https://arxiv.org/abs/1710.02298).


## Running the Model
1. Download the environment from one of the links below. You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
2. Place the file in the DRLND GitHub repository, and unzip (or decompress) the file.
3. Open CMD/Terminal and run 'pip install -r requirements.txt'
4. Run 'python main.py' in your terminal to begin the training process.



