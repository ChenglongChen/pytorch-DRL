
import torch as th
from torch import nn
from torch.optim import Adam, RMSprop
import numpy as np

from common.Memory import ReplayMemory
from common.Model import ActorNetwork, CriticNetwork
from common.utils import entropy, index_to_one_hot, to_tensor_var


class MAA2C(object):
    """
    An multi-agent learned with Advantage Actor-Critic
    - Actor takes its local observations as input
    - agent interact with environment to collect experience
    - agent training with experience to update policy

    - training_strategy:
        - cocurrent
            - each agent learns its own individual policy which is independent
            - multiple policies are optimized simultaneously
        - centralized
            - centralized training of decentralized actors
            - centralized critic takes both state and action from all agents as input
            - each actor has its own critic for estimating the value function, which allows
                each actor has different reward structure, e.g., cooperative, competitive,
                mixed cooperative-competitive
    """
    def __init__(self, env, memory_capacity, n_agents, state_dim, action_dim,
                 actor_hidden_size=32, actor_lr=0.001,
                 critic_hidden_size=32, critic_lr=0.001,
                 max_grad_norm=None, entropy_reg=0.01,
                 optimizer_type="rmsprop", alpha=0.99, epsilon=1e-08,
                 use_cuda=True, batch_size=10, n_steps=5,
                 reward_gamma=0.99,
                 done_penalty=None, training_strategy="centralized",
                 epsilon_start=0.9, epsilon_end=0.05,
                 epsilon_decay=200, episodes_before_train=0,
                 critic_loss="huber"):

        assert training_strategy in ["cocurrent", "centralized"]
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_steps = n_steps
        self.env_state = self.env.reset()
        self.n_episodes = 0
        self.done_penalty = done_penalty
        self.reward_gamma = reward_gamma
        self.episodes_before_train = episodes_before_train

        self.n_agents = n_agents
        self.memory = ReplayMemory(memory_capacity)
        
        self.max_grad_norm = max_grad_norm
        self.entropy_reg = entropy_reg
        self.batch_size = batch_size
        self.critic_loss = critic_loss
        self.training_strategy = training_strategy

        # params for epsilon greedy
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.actors = [ActorNetwork(self.state_dim, actor_hidden_size, self.action_dim, nn.functional.softmax)] * self.n_agents
        if self.training_strategy == "cocurrent":
            self.critics = [CriticNetwork(self.state_dim, self.action_dim, critic_hidden_size, 1)] * self.n_agents
        elif self.training_strategy == "centralized":
            critic_state_dim = self.n_agents * self.state_dim
            critic_action_dim = self.n_agents * self.action_dim
            self.critics = [CriticNetwork(critic_state_dim, critic_action_dim, critic_hidden_size, 1)] * self.n_agents
        if optimizer_type == "adam":
            self.actor_optimizers = [Adam(a.parameters(), lr=actor_lr) for a in self.actors]
            self.critic_optimizers = [Adam(c.parameters(), lr=critic_lr) for c in self.critics]
        elif optimizer_type == "rmsprop":
            self.actor_optimizers = [RMSprop(a.parameters(), lr=actor_lr, alpha=alpha, eps=epsilon)
                                        for a in self.actors]
            self.critic_optimizers = [RMSprop(c.parameters(), lr=critic_lr, alpha=alpha, eps=epsilon)
                                        for c in self.critics]
        self.use_cuda = use_cuda and th.cuda.is_available()
        self.FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        if self.use_cuda:
            for a in self.actors:
                a.cuda()
            for c in self.critics:
                c.cuda()

    # discount rewards
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
            actions.append([index_to_one_hot(a, self.action_dim) for a in action])
            rewards.append(reward)
            final_state = next_state
            self.env_state = next_state
            if done[0]:
                self.env_state = self.env.reset()
                break
        # discount reward
        if done[0]:
            final_r = [0.0] * self.n_agents
            self.n_episodes += 1
            self.episode_done = True
        else:
            self.episode_done = False
            final_action = self.action(final_state)
            one_hot_action = [index_to_one_hot(a, self.action_dim) for a in final_action]
            final_r = self.value(final_state, one_hot_action)

        rewards = np.array(rewards)
        for agent_id in range(self.n_agents):
            rewards[:,agent_id] = self._discount_reward(rewards[:,agent_id], final_r[agent_id])
        rewards = rewards.tolist()

        self.memory.push(states, actions, rewards)

    # train on a roll out batch
    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            pass

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        actions_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, self.n_agents, 1)
        whole_states_var = states_var.view(-1, self.n_agents*self.state_dim)
        whole_actions_var = actions_var.view(-1, self.n_agents*self.action_dim)

        for agent_id in range(self.n_agents):
            # update actor network
            self.actor_optimizers[agent_id].zero_grad()
            softmax_actions = self.actors[agent_id](states_var[:,agent_id,:])
            if self.training_strategy == "cocurrent":
                values = self.critics[agent_id](states_var[:,agent_id,:], actions_var[:,agent_id,:]).detach()
            elif self.training_strategy == "centralized":
                values = self.critics[agent_id](whole_states_var, whole_actions_var).detach()
            advantages = rewards_var[:,agent_id,:] - values
            neg_logloss = - th.sum(softmax_actions * actions_var[:,agent_id,:], 1)
            pg_loss = th.mean(neg_logloss * advantages)
            entropy_loss = th.mean(entropy(softmax_actions))
            actor_loss = pg_loss + entropy_loss * self.entropy_reg
            actor_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm(self.actors[agent_id].parameters(), self.max_grad_norm)
            self.actor_optimizers[agent_id].step()

            # update critic network
            self.critic_optimizers[agent_id].zero_grad()
            target_values = rewards_var[:,agent_id,:]
            if self.training_strategy == "cocurrent":
                values = self.critics[agent_id](states_var[:,agent_id,:], actions_var[:,agent_id,:])
            elif self.training_strategy == "centralized":
                values = self.critics[agent_id](whole_states_var, whole_actions_var)
            if self.critic_loss == "huber":
                critic_loss = nn.functional.smooth_l1_loss(values, target_values)
            else:
                critic_loss = nn.MSELoss()(values, target_values)
            critic_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm(self.critics[agent_id].parameters(), self.max_grad_norm)
            self.critic_optimizers[agent_id].step()

    # predict softmax action based on state
    def _softmax_action(self, state):
        state_var = to_tensor_var([state], self.use_cuda)
        softmax_action = np.zeros((self.n_agents, self.action_dim), dtype=np.float64)
        for agent_id in range(self.n_agents):
            softmax_action_var = self.actors[agent_id](state_var[:,agent_id,:])
            if self.use_cuda:
                softmax_action[agent_id] = np.argmax(softmax_action_var.data.cpu().numpy()[0])
            else:
                softmax_action[agent_id] = np.argmax(softmax_action_var.data.numpy()[0])
        return softmax_action

    # predict action based on state, added random noise for exploration in training
    def exploration_action(self, state):
        softmax_action = self._softmax_action(state)
        actions = [0]*self.n_agents
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                                     np.exp(-1. * self.n_steps / self.epsilon_decay)
        for agent_id in range(self.n_agents):
            if np.random.rand() < epsilon:
                actions[agent_id] = np.random.choice(self.action_dim, p=softmax_action[agent_id])
            else:
                actions[agent_id] = np.argmax(softmax_action[agent_id])
        return actions

    # predict action based on state for execution
    def action(self, state):
        softmax_actions = self._softmax_action(state)
        actions = np.argmax(softmax_actions, axis=1)
        return actions

    # evaluate value
    def value(self, state, action):
        state_var = to_tensor_var([state], self.use_cuda)
        action_var = to_tensor_var([action], self.use_cuda)
        whole_state_var = state_var.view(-1, self.n_agents*self.state_dim)
        whole_action_var = action_var.view(-1, self.n_agents*self.action_dim)
        values = [0]*self.n_agents
        for agent_id in range(self.n_agents):
            if self.training_strategy == "cocurrent":
                value_var = self.critics[agent_id](state_var[:,agent_id,:], action_var[:,agent_id,:])
            elif self.training_strategy == "centralized":
                value_var = self.critics[agent_id](whole_state_var, whole_action_var)
            if self.use_cuda:
                values[agent_id] = value_var.data.cpu().numpy()[0]
            else:
                values[agent_id] = value_var.data.numpy()[0]
        return values
    
    # evaluation
    def evaluation(self, env, eval_episodes=10):
        rewards = np.zeros(self.n_agents)
        for i in range(eval_episodes):
            state = env.reset()
            action = self.action(state)
            state, reward, done, _ = env.step(action)
            rewards += np.array(reward)
            while not done:
                action = self.action(state)
                state, reward, done, _ = env.step(action)
                rewards += np.array(reward)
        rewards /= float(eval_episodes)
        return rewards
