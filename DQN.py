
import torch as th
from torch import nn
from torch.optim import Adam, RMSprop
import numpy as np

from common.Agent import Agent
from common.Memory import ReplayMemory
from common.Model import ActorNetwork
from common.utils import identity, to_tensor_var


class DQN(Agent):
    """
    An agent learned with DQN using replay memory and temporal difference
    - use a value network to estimate the state-action value
    """
    def __init__(self, env, memory_capacity, state_dim, action_dim,
                 hidden_size=32, lr=0.001, max_grad_norm=None,
                 optimizer_type="rmsprop", alpha=0.99, epsilon=1e-08,
                 use_cuda=False, batch_size=10, max_steps=1000,
                 reward_gamma=0.99,
                 done_penalty=None, episodes_before_train=100,
                 reward_scale=1., loss="huber",
                 epsilon_start=0.9, epsilon_end=0.05,
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
        self.reward_scale = reward_scale

        # params for epsilon greedy
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.loss = loss
        self.episodes_before_train = episodes_before_train

        self.value_network = ActorNetwork(self.state_dim, hidden_size, self.action_dim, identity)
        if optimizer_type == "adam":
            self.value_network_optimizer = Adam(self.value_network.parameters(), lr=lr)
        elif optimizer_type == "rmsprop":
            self.value_network_optimizer = RMSprop(
                self.value_network.parameters(), lr=lr, alpha=alpha, eps=epsilon)
        self.use_cuda = use_cuda and th.cuda.is_available()
        if self.use_cuda:
            self.value_network.cuda()

    # agent interact with the environment to collect experience
    def interact(self):
        if self.n_steps >= self.max_steps:
            self.env_state = self.env.reset()
            self.n_steps = 0
        state = self.env_state
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

    # train on a sample batch
    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            pass

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.state_dim)
        actions_var = to_tensor_var(batch.actions, self.use_cuda, "long").view(-1, 1)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, 1)
        next_states_var = to_tensor_var(batch.next_states, self.use_cuda).view(-1, self.state_dim)
        dones_var = to_tensor_var(batch.dones, self.use_cuda).view(-1, 1)

        # compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        current_q = self.value_network(states_var).gather(1, actions_var)

        # compute V(s_{t+1}) for all next states and all actions,
        # and we then take max_a { V(s_{t+1}) }
        next_state_action_values = self.value_network(next_states_var).detach()
        next_q = th.max(next_state_action_values, 1)[0].view(-1, 1)
        # compute target q by: r + gamma * max_a { V(s_{t+1}) }
        target_q = self.reward_scale * rewards_var + self.reward_gamma * next_q * (1. - dones_var)

        # update value network
        self.value_network_optimizer.zero_grad()
        if self.loss == "huber":
            loss = th.nn.functional.smooth_l1_loss(current_q, target_q)
        else:
            loss = th.nn.MSELoss()(current_q, target_q)
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.value_network.parameters(), self.max_grad_norm)
        self.value_network_optimizer.step()

    # predict action based on state, added random noise for exploration in training
    def exploration_action(self, state):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                                  np.exp(-1. * self.n_steps / self.epsilon_decay)
        if np.random.rand() < epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = self.action(state)
        return action

    # predict action based on state for execution
    def action(self, state):
        state_var = to_tensor_var([state], self.use_cuda)
        state_action_value_var = self.value_network(state_var)
        if self.use_cuda:
            state_action_value = state_action_value_var.data.cpu().numpy()[0]
        else:
            state_action_value = state_action_value_var.data.numpy()[0]
        action = np.argmax(state_action_value)
        return action
