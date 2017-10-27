
import torch as th
from torch import nn
from torch.optim import Adam, RMSprop
import numpy as np

from common.Agent import Agent
from common.Memory import ReplayMemory
from common.Model import ActorNetwork, CriticNetwork
from common.utils import entropy, index_to_one_hot, to_tensor_var


class A2C(Agent):
    """
    An agent learned with Advantage Actor-Critic
    - Actor takes state as input
    - Critic takes both state and action as input
    - agent interact with environment to collect experience
    - agent training with experience to update policy
    """
    def __init__(self, env, memory_capacity, state_dim, action_dim,
                 actor_hidden_size=32, actor_lr=0.001,
                 critic_hidden_size=32, critic_lr=0.001,
                 max_grad_norm=None, entropy_reg=0.01,
                 optimizer_type="rmsprop", alpha=0.99, epsilon=1e-08,
                 use_cuda=False, batch_size=10, n_steps=5,
                 reward_gamma=0.99, done_penalty=None,
                 epsilon_start=0.9, epsilon_end=0.05,
                 epsilon_decay=200, episodes_before_train=100,
                 critic_loss="huber"):

        self.memory = ReplayMemory(memory_capacity)

        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_steps = n_steps
        self.env_state = self.env.reset()
        self.n_episodes = 0
        self.done_penalty = done_penalty
        self.reward_gamma = reward_gamma
        self.episodes_before_train = episodes_before_train

        self.max_grad_norm = max_grad_norm
        self.entropy_reg = entropy_reg
        self.batch_size = batch_size
        self.critic_loss = critic_loss

        # params for epsilon greedy
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.actor = ActorNetwork(self.state_dim, actor_hidden_size, self.action_dim, nn.functional.softmax)
        self.critic = CriticNetwork(self.state_dim, self.action_dim, critic_hidden_size, 1)
        if optimizer_type == "adam":
            self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        elif optimizer_type == "rmsprop":
            self.actor_optimizer = RMSprop(
                self.actor.parameters(), lr=actor_lr, alpha=alpha, eps=epsilon)
            self.critic_optimizer = RMSprop(
                self.critic.parameters(), lr=critic_lr, alpha=alpha, eps=epsilon)
        self.use_cuda = use_cuda and th.cuda.is_available()
        if self.use_cuda:
            self.actor.cuda()

    # discount roll out rewards
    def _discount_reward(self, rewards, final_r):
        discounted_r = np.zeros_like(rewards)
        running_add = final_r
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.reward_gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    # agent interact with the environment to collect experience
    def interact(self):
        states = []
        actions = []
        rewards = []
        # take n steps
        for i in range(self.n_steps):
            states.append(self.env_state)
            action = self.exploration_action(self.env_state)
            next_state, reward, done, _ = self.env.step(action)
            actions.append(index_to_one_hot(action, self.action_dim))
            if done and self.done_penalty is not None:
                reward = self.done_penalty
            rewards.append(reward)
            final_state = next_state
            self.env_state = next_state
            if done:
                self.env_state = self.env.reset()
                break
        # discount reward
        if done:
            final_r = 0.0
            self.n_episodes += 1
            self.episode_done = True
        else:
            self.episode_done = False
            final_action = self.action(final_state)
            final_r = self.value(final_state, index_to_one_hot(final_action, self.action_dim))
        rewards = self._discount_reward(rewards, final_r)

        self.memory.push(states, actions, rewards)

    # train on a roll out batch
    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            pass

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.state_dim)
        actions_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.action_dim)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, 1)

        # update actor network
        self.actor_optimizer.zero_grad()
        # actions_var is with noise, while softmax_actions is the accurate predictions
        softmax_actions = self.actor(states_var)
        neg_logloss = - th.sum(softmax_actions * actions_var, 1)
        values = self.critic(states_var, actions_var).detach()
        advantages = rewards_var - values
        pg_loss = th.mean(neg_logloss * advantages)
        entropy_loss = th.mean(entropy(softmax_actions))
        actor_loss = pg_loss - entropy_loss * self.entropy_reg
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # update critic network
        self.critic_optimizer.zero_grad()
        target_values = rewards_var
        values = self.critic(states_var, actions_var)
        if self.critic_loss == "huber":
            critic_loss = nn.functional.smooth_l1_loss(values, target_values)
        else:
            critic_loss = nn.MSELoss()(values, target_values)
        critic_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

    # predict softmax action based on state
    def _softmax_action(self, state):
        state_var = to_tensor_var([state], self.use_cuda)
        softmax_action_var = self.actor(state_var)
        if self.use_cuda:
            softmax_action = softmax_action_var.data.cpu().numpy()[0]
        else:
            softmax_action = softmax_action_var.data.numpy()[0]
        return softmax_action

    # predict action based on state, added random noise for exploration in training
    def exploration_action(self, state):
        softmax_action = self._softmax_action(state)
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                                  np.exp(-1. * self.n_steps / self.epsilon_decay)
        if np.random.rand() < epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = np.argmax(softmax_action)
        return action

    # predict action based on state for execution
    def action(self, state):
        softmax_action = self._softmax_action(state)
        action = np.argmax(softmax_action)
        return action

    # evaluate value
    def value(self, state, action):
        state_var = to_tensor_var([state], self.use_cuda)
        action_var = to_tensor_var([action], self.use_cuda)
        value_var = self.critic(state_var, action_var)
        if self.use_cuda:
            value = value_var.data.cpu().numpy()[0]
        else:
            value = value_var.data.numpy()[0]
        return value
