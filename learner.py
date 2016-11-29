import numpy as np
from enum import Enum
import math

class ExplorationStrategy(Enum):
    NONE = 0
    SOFTMAX = 1


class LinearApproximatedOnlineLearner:

    def __init__(self, features, actions, alpha=.01, gamma=.99):
        self.features = features
        self.actions = actions
        self.theta = np.zeros(self.features.size())
        self.alpha = alpha
        self.gamma = gamma
       
    
    def _softmax(self, x, l):
        return np.exp(l*x)/np.sum(np.exp(l*x), axis=0)

    def act(self, state, exploration=ExplorationStrategy.NONE, exploration_param=1):
        Q = [self.Q(state, action) for action in range(0, self.actions.size())]
        if exploration == ExplorationStrategy.NONE:
            max_action = 0
            max_value = -math.inf
            for action in range(0, self.actions.size()):
                if max_value < Q[action]:
                    max_value = Q[action]
                    max_action = action
            return max_action
        elif exploration == ExplorationStrategy.SOFTMAX:
            p = self._softmax(Q, exploration_param)
            return np.random.multinomial(1, p, size=1)[0]
        else:
            return # Error

    def U(self, state):
        Q_max = -math.inf
        for i in range(0, self.actions.size()):
            Q_max = max(Q_max, self.Q(state, i))
        return Q_max

    def Q(self, state, action):
        return np.dot(self.theta, self.features.compute(state, action))

    def update(self, state, action, reward, next_state):
        td_error = reward + self.gamma * self.U(next_state) - self.Q(state, action)
        self.theta = np.add(self.theta, alpha * td_error * self.features.compute(state, action))

