import gym
import gym_soccer
import numpy
import math

from actions_soccer import ActionSpaceSoccerSimple 
from actions_soccer import SoccerDirection
from features_soccer import FeaturesSoccerSmall
from learner import ExplorationStrategy
from learner import LinearApproximatedOnlineLearner

NUM_EPISODES = 10#500

turn_dirs = [SoccerDirection.BALL]
dash_powers = [0,50,100] 
kick_powers = [10,25]
kick_dirs = [SoccerDirection.GOAL_CENTER] # SoccerDirection.FORWARD

actions = ActionSpaceSoccerSimple(turn_dirs, dash_powers, kick_powers, kick_dirs)
features = FeaturesSoccerSmall(actions)
exploration = ExplorationStrategy.EPSILON_GREEDY #ExplorationStrategy.EPSILON_GREEDY #ExplorationStrategy.SOFTMAX
learner = LinearApproximatedOnlineLearner(features, actions)

env = gym.make('SoccerEmptyGoal-v0')

for _ in range(NUM_EPISODES):
    observation = env.reset()
    done = False
    t = 0
    while not done:
        #env.render()
        action = learner.act(observation, exploration, 0.1)
        observation_next, reward, done, info = env.step(actions.env_action(observation, action))
        if not done:
            learner.update(observation, action, reward, observation_next)
        
            print(str(t))
            print("State:")
            print(features.state_feature_str(observation))
            print("Action: " + actions.env_action_str(observation, action))
            print(str(t) + " Reward: " + str(reward) + "\n\n")

            observation = observation_next
            t = t + 1


#for _ in range(1000):
#    env.render()
#    env.step(env.action_space.sample()) 

#from gym import envs
#print(envs.registry.all())
