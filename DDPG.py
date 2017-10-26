
import torch as th
import torch.nn as nn
from torch.optim import Adam, RMSprop
from copy import deepcopy
import numpy as np

from common.Memory import ReplayMemory
from common.Model import ActorNetwork, CriticNetwork
from common.utils import to_tensor_var


class DDPG(object):
    """
    An agent learned with Deep Deterministic Policy Gradient using Actor-Critic framework
    - Actor takes state as input
    - Critic takes both state and action as input
    - Critic uses gradient temporal-difference learning
    """
    def __init__(self, env, memory_capacity, state_dim, action_dim,
                 actor_hidden_size=32, actor_lr=0.001,
                 actor_output_act=nn.functional.tanh,
                 critic_hidden_size=32, critic_lr=0.001,
                 max_grad_norm=None, max_steps=1000,
                 optimizer_type="rmsprop", alpha=0.99, epsilon=1e-08,
                 use_cuda=True, batch_size=10,
                 reward_gamma=0.99,
                 done_penalty=None, episodes_before_train=100,
                 target_tau=0.01, reward_scale=1.0,
                 critic_loss="huber", epsilon_start=0.99, epsilon_end=0.05,
                 epsilon_decay=200):

        self.memory = ReplayMemory(memory_capacity)

        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env_state = self.env.reset()
        self.n_episodes = 0
        self.n_steps = 0
        self.max_steps = max_steps
        self.done_penalty = done_penalty
        self.reward_gamma = reward_gamma
        self.target_tau = target_tau
        self.reward_scale = reward_scale

        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train
        self.critic_loss = critic_loss

        # params for epsilon greedy
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.actor = ActorNetwork(self.state_dim, actor_hidden_size, self.action_dim, actor_output_act)
        self.critic = CriticNetwork(self.state_dim, self.action_dim, critic_hidden_size, 1)
        # to ensure target network and learning network has the same weights
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

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
            self.critic.cuda()
            self.actor_target.cuda()
            self.critic_target.cuda()

    # agent interact with the environment to collect experience
    def interact(self):
        if self.n_steps >= self.max_steps:
            self.env_state = self.env.reset()
            self.n_steps = 0
        state = self.env_state
        # take one step action and get one step reward
        action = self.exploration_action(self.env_state)
        next_state, reward, done, _ = self.env.step(action)
        if done:
            if self.done_penalty is not None:
                reward = self.done_penalty
            next_state = [0]*len(state)
            self.env_state = self.env.reset()
            self.n_episodes += 1
            self.episode_done = True
        else:
            self.env_state = next_state
            self.episode_done = False
        self.n_steps += 1
        self.memory.push(state, action, reward, next_state, done)

    # soft update the actor target network or critic target network
    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)

    # train on a sample batch
    def train(self):
        # do not train until exploration is enough
        if self.n_episodes <= self.episodes_before_train:
            pass

        batch = self.memory.sample(self.batch_size)
        state_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.state_dim)
        action_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.action_dim)
        reward_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, 1)
        next_state_var = to_tensor_var(batch.next_states, self.use_cuda).view(-1, self.state_dim)
        done_var = to_tensor_var(batch.dones, self.use_cuda).view(-1, 1)

        # estimate the target q with actor_target network and critic_target network
        next_action_var = self.actor_target(next_state_var)
        next_q = self.critic_target(next_state_var, next_action_var).detach()
        target_q = self.reward_scale * reward_var + self.reward_gamma * next_q * (1. - done_var)

        # update critic network
        self.critic_optimizer.zero_grad()
        # current Q values
        current_q = self.critic(state_var, action_var)
        # rewards is target Q values
        if self.critic_loss == "huber":
            critic_loss = nn.functional.smooth_l1_loss(current_q, target_q)
        else:
            critic_loss = nn.MSELoss()(current_q, target_q)
        critic_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # update actor network
        self.actor_optimizer.zero_grad()
        # the accurate action prediction
        action = self.actor(state_var)
        # actor_loss is used to maximize the Q value for the predicted action
        actor_loss = - self.critic(state_var, action)
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # update actor target network and critic target network
        if self.n_steps % 100 == 0 and self.n_steps > 0:
            self._soft_update_target(self.critic_target, self.critic)
            self._soft_update_target(self.actor_target, self.actor)

    # predict action based on state, added random noise for exploration in training
    def exploration_action(self, state):
        action = self.action(state)
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                                  np.exp(-1. * self.n_steps / self.epsilon_decay)
        # add noise
        noise = np.random.randn(self.action_dim) * epsilon
        action += noise
        return action

    # predict action based on state for execution (using current actor)
    def action(self, state):
        action_var = self.actor(to_tensor_var([state], self.use_cuda))
        if self.use_cuda:
            action = action_var.data.cpu().numpy()[0]
        else:
            action = action_var.data.numpy()[0]
        return action

    # evaluation
    def evaluation(self, env, eval_episodes=10):
        rewards = 0
        for i in range(eval_episodes):
            state = env.reset()
            action = self.action(state)
            state, reward, done, _ = env.step(action)
            rewards += reward
            while not done:
                action = self.action(state)
                state, reward, done, _ = env.step(action)
                rewards += reward
        rewards /= float(eval_episodes)
        return rewards