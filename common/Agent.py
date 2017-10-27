
import torch as th

from common.Memory import ReplayMemory
from common.utils import identity


class Agent(object):
    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=10000,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=32,
                 actor_output_act=identity, critic_loss="mse",
                 actor_lr=0.01, critic_lr=0.01,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 use_cuda=True):

        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env_state = self.env.reset()
        self.n_episodes = 0
        self.n_steps = 0
        self.max_steps = max_steps

        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale
        self.done_penalty = done_penalty

        self.memory = ReplayMemory(memory_capacity)
        self.actor_hidden_size = actor_hidden_size
        self.critic_hidden_size = critic_hidden_size
        self.actor_output_act = actor_output_act
        self.critic_loss = critic_loss
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optimizer_type = optimizer_type
        self.entropy_reg = entropy_reg
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train

        # params for epsilon greedy
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.use_cuda = use_cuda and th.cuda.is_available()

    # agent interact with the environment to collect experience
    def interact(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            self.env_state = self.env.reset()
            self.n_steps = 0
        state = self.env_state
        action = self.exploration_action(self.env_state)
        next_state, reward, done, _ = self.env.step(action)
        if done:
            if self.done_penalty is not None:
                reward = self.done_penalty
            next_state = [0] * len(state)
            self.env_state = self.env.reset()
            self.n_episodes += 1
            self.episode_done = True
        else:
            self.env_state = next_state
            self.episode_done = False
        self.n_steps += 1
        self.memory.push(state, action, reward, next_state, done)

    # train on a sample batch
    def train(self):
        pass

    # predict action based on state, added random noise for exploration in training
    def exploration_action(self, state):
        pass

    # choice an action
    def action(self, state):
        pass

    # evaluation
    def evaluation(self, env, eval_episodes=10):
        rewards = []
        infos = []
        for i in range(eval_episodes):
            rewards_i = []
            infos_i = []
            state = env.reset()
            action = self.action(state)
            state, reward, done, info = env.step(action)
            rewards_i.append(reward)
            infos_i.append(info)
            while not done:
                action = self.action(state)
                state, reward, done, info = env.step(action)
                rewards_i.append(reward)
                infos_i.append(info)
            rewards.append(rewards_i)
            infos.append(infos_i)
        return rewards, infos
