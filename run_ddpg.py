
from DDPG import DDPG

import gym
import sys
import numpy as np
import matplotlib.pyplot as plt


MAX_EPISODES = 10000
EPISODES_BEFORE_TRAIN = 100
EVAL_EPISODES = 10
EVAL_INTERVAL = 20
# max steps in each episode
MAX_STEPS = 1000

MEMORY_CAPACITY = 10000
BATCH_SIZE = 1000
CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None

REWARD_DISCOUNTED_GAMMA = 0.99

EPSILON_START = 0.99
EPSILON_END = 0.05
EPSILON_DECAY = 500

RANDOM_SEED = 2017


def run(env_id="Pendulum-v0"):

    env = gym.make(env_id)
    env.seed(RANDOM_SEED)
    env_eval = gym.make(env_id)
    env_eval.seed(RANDOM_SEED)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ddpg = DDPG(env=env, memory_capacity=MEMORY_CAPACITY,
                state_dim=state_dim, action_dim=action_dim,
                batch_size=BATCH_SIZE, max_steps=MAX_STEPS,
                reward_gamma=REWARD_DISCOUNTED_GAMMA, critic_loss=CRITIC_LOSS,
                epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
                epsilon_decay=EPSILON_DECAY, max_grad_norm=MAX_GRAD_NORM,
                episodes_before_train=EPISODES_BEFORE_TRAIN)

    episodes =[]
    eval_rewards =[]
    while ddpg.n_episodes < MAX_EPISODES:
        ddpg.interact()
        if ddpg.n_episodes >= EPISODES_BEFORE_TRAIN:
            ddpg.train()
        if ddpg.episode_done and ((ddpg.n_episodes+1)%EVAL_INTERVAL == 0):
            rewards = ddpg.evaluation(env_eval, EVAL_EPISODES)
            print("Episode: %d, Average Reward: %.5f" % (ddpg.n_episodes+1, rewards))
            episodes.append(ddpg.n_episodes+1)
            eval_rewards.append(rewards)

    episodes = np.array(episodes)
    eval_rewards = np.array(eval_rewards)
    np.savetxt("./output/%s_ddpg_episodes.txt"%env_id, episodes)
    np.savetxt("./output/%s_ddpg_eval_rewards.txt"%env_id, eval_rewards)

    plt.figure()
    plt.plot(episodes, eval_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend(["DDPG"])
    plt.savefig("./output/%s_ddpg.png"%env_id)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        run(sys.argv[1])
    else:
        run()
