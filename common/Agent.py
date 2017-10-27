
class Agent(object):
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
