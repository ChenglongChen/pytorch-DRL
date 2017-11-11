
import torch as th
from torch import nn
from torch.optim import Adam, RMSprop

import numpy as np

from common.Agent import Agent
from common.Model import ActorNetwork
from common.utils import identity, to_tensor_var


class DQN(Agent):
    """
    An agent learned with DQN using replay memory and temporal difference
    - use a value network to estimate the state-action value
    """
    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=10000,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=32,
                 actor_output_act=identity, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 use_cuda=True):
        super(DQN, self).__init__(env, state_dim, action_dim,
                 memory_capacity, max_steps,
                 reward_gamma, reward_scale, done_penalty,
                 actor_hidden_size, critic_hidden_size,
                 actor_output_act, critic_loss,
                 actor_lr, critic_lr,
                 optimizer_type, entropy_reg,
                 max_grad_norm, batch_size, episodes_before_train,
                 epsilon_start, epsilon_end, epsilon_decay,
                 use_cuda)

        self.actor = ActorNetwork(self.state_dim, self.actor_hidden_size,
                                          self.action_dim, self.actor_output_act)
        if self.optimizer_type == "adam":
            self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizer = RMSprop(self.actor.parameters(), lr=self.actor_lr)
        if self.use_cuda:
            self.actor.cuda()

    # agent interact with the environment to collect experience
    def interact(self):
        super(DQN, self)._take_one_step()

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
        current_q = self.actor(states_var).gather(1, actions_var)

        # compute V(s_{t+1}) for all next states and all actions,
        # and we then take max_a { V(s_{t+1}) }
        next_state_action_values = self.actor(next_states_var).detach()
        next_q = th.max(next_state_action_values, 1)[0].view(-1, 1)
        # compute target q by: r + gamma * max_a { V(s_{t+1}) }
        target_q = self.reward_scale * rewards_var + self.reward_gamma * next_q * (1. - dones_var)

        # update value network
        self.actor_optimizer.zero_grad()
        if self.critic_loss == "huber":
            loss = th.nn.functional.smooth_l1_loss(current_q, target_q)
        else:
            loss = th.nn.MSELoss()(current_q, target_q)
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

    # choose an action based on state with random noise added for exploration in training
    def exploration_action(self, state):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                                  np.exp(-1. * self.n_steps / self.epsilon_decay)
        if np.random.rand() < epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = self.action(state)
        return action

    # choose an action based on state for execution
    def action(self, state):
        state_var = to_tensor_var([state], self.use_cuda)
        state_action_value_var = self.actor(state_var)
        if self.use_cuda:
            state_action_value = state_action_value_var.data.cpu().numpy()[0]
        else:
            state_action_value = state_action_value_var.data.numpy()[0]
        action = np.argmax(state_action_value)
        return action
