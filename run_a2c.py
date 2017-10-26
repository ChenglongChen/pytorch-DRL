
from A2C import A2C

import sys
import gym
import numpy as np
import matplotlib.pyplot as plt


MAX_EPISODES = 10000
EPISODES_BEFORE_TRAIN = 0
EVAL_EPISODES = 10
EVAL_INTERVAL = 100

# roll out n steps
N_STEPS = 10
# only remember the latest N_STEPS
MEMORY_CAPACITY = N_STEPS
# only use the latest N_STEPS for training a2c
BATCH_SIZE = N_STEPS

REWARD_DISCOUNTED_GAMMA = 0.99
# seems entropy regularization don't help
ENTROPY_REG = 0.00
#
DONE_PENALTY = -10.

MAX_GRAD_NORM = None

EPSILON_START = 0.99
EPSILON_END = 0.05
EPSILON_DECAY = 500

RANDOM_SEED = 2017


def run(env_id="CartPole-v0"):

    env = gym.make(env_id)
    env.seed(RANDOM_SEED)
    env_eval = gym.make(env_id)
    env_eval.seed(RANDOM_SEED)
    state_dim = env.observation_space.shape[0]
    if len(env.action_space.shape) > 1:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    a2c = A2C(env=env, memory_capacity=MEMORY_CAPACITY,
              state_dim=state_dim, action_dim=action_dim,
              batch_size=BATCH_SIZE, entropy_reg=ENTROPY_REG,
              done_penalty=DONE_PENALTY, n_steps=N_STEPS,
              reward_gamma=REWARD_DISCOUNTED_GAMMA,
              epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
              epsilon_decay=EPSILON_DECAY, max_grad_norm=MAX_GRAD_NORM,
              episodes_before_train=EPISODES_BEFORE_TRAIN)

    episodes =[]
    eval_rewards =[]
    while a2c.n_episodes < MAX_EPISODES:
        a2c.interact()
        if a2c.n_episodes >= EPISODES_BEFORE_TRAIN:
            a2c.train()
        if a2c.episode_done and ((a2c.n_episodes+1)%EVAL_INTERVAL == 0):
            rewards = a2c.evaluation(env_eval, EVAL_EPISODES)
            print("Episode %d, Average Reward %.2f" % (a2c.n_episodes+1, rewards))
            episodes.append(a2c.n_episodes+1)
            eval_rewards.append(rewards)

    episodes = np.array(episodes)
    eval_rewards = np.array(eval_rewards)
    np.savetxt("./output/%s_a2c_episodes.txt"%env_id, episodes)
    np.savetxt("./output/%s_a2c_eval_rewards.txt"%env_id, eval_rewards)

    plt.figure()
    plt.plot(episodes, eval_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend(["A2C"])
    plt.savefig("./output/%s_a2c.png"%env_id)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        run(sys.argv[1])
    else:
        run()
