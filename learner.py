import numpy as np
import math
import random

class ExplorationStrategy:
    NONE = 0
    SOFTMAX = 1
    EPSILON_GREEDY = 2

class LinearApproximatedOnlineLearner:

    def __init__(self, features, actions, alpha=.001, gamma=.99):
        self.features = features
        self.actions = actions
        self.theta = np.zeros(self.features.size())
        self.alpha = alpha
        self.gamma = gamma
       
    
    def _softmax(self, x, l):
        return np.exp(np.multiply(l,x-np.max(x)))/np.sum(np.exp(np.multiply(l,x-np.max(x))), axis=0)

    def act(self, state, exploration=ExplorationStrategy.NONE, exploration_param=1):
        Q = [self.Q(state, action) for action in range(0, self.actions.size(state))]
        if exploration == ExplorationStrategy.NONE:
            max_action = 0
            max_value = -float("inf")
            for action in range(0, self.actions.size(state)):
                if max_value < Q[action]:
                    max_value = Q[action]
                    max_action = action
            return max_action
        elif exploration == ExplorationStrategy.SOFTMAX:
            p = self._softmax(Q, exploration_param)
            #print(p)
            return np.where(np.random.multinomial(1, p, size=1)[0] == 1)[0][0]
        elif exploration == ExplorationStrategy.EPSILON_GREEDY:
            if random.random() < exploration_param:
                return random.randint(0, self.actions.size(state) - 1)
            else:
                return self.act(state)
        else:
            return # Error

    def U(self, state):
        Q_max = -float("inf")
        for i in range(0, self.actions.size(state)):
            Q_max = max(Q_max, self.Q(state, i))
        return Q_max

    def Q(self, state, action):
        return np.dot(self.theta, self.features.compute(state, action))

    def update(self, state, action, reward, next_state):
        #print(self.U(next_state))#self.theta)
        td_error = reward + self.gamma * self.U(next_state) - self.Q(state, action)
        self.theta = np.add(self.theta, self.alpha * td_error * self.features.compute(state, action))

