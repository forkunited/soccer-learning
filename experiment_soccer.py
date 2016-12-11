# Code for running the Q-learning experiments on the
# soccer environment.  This code runs the experiments
# documented in the paper

import gym
import gym_soccer
import numpy
import math

from actions_soccer import ActionSpaceSoccerSimple 
from actions_soccer import SoccerDirection
from features_soccer import FeaturesSoccerSmall
from learner import ExplorationStrategy
from learner import LinearApproximatedOnlineLearner

class ExperimentSoccer:

    def __init__(self, episodes, train_iters, test_iters, test_freq, actions, exploration, exploration_param, interact_feats=True, goalie=False):
        self.episodes = episodes
        self.train_iters = train_iters
        self.test_iters = test_iters
        self.test_freq = test_freq
        self.actions = actions
        self.exploration = exploration
        self.exploration_param = exploration_param
        self.goalie = goalie
        self.features = FeaturesSoccerSmall(actions, interact_feats, goalie) 

    def make_env(self):
        if self.goalie:
            return gym.make('SoccerAgainstKeeper-v0')
        else:
            return gym.make('SoccerEmptyGoal-v0')

    # Aggregate the results of the experiment
    def aggregate(self, results):
        agg = []
        for i in range(len(results[0])):
            agg.append([results[0][i][0],0,0])   

        for i in range(len(results)):
            for j in range(len(results[i])):
                agg[j][1] = agg[j][1] + results[i][j][1]/len(results) # Add to mean
 
        for i in range(len(results)):
            for j in range(len(results[i])):
                agg[j][2] = agg[j][2] + (results[i][j][1]-agg[j][1])*(results[i][j][1]-agg[j][1])/(len(results) - 1.0)

        for i in range(len(agg)):
            agg[i][2] = math.sqrt(agg[i][2])/math.sqrt(len(results))

        return agg

    # Get a latex string representation of the results
    def str_aggregate(self, results):
        agg = self.aggregate(results)
        s = ""
        for i in range(len(agg)):
            s = s + "(" + str(agg[i][0]) + "," + str(agg[i][1]) + ")" + "+-" + "(0," + str(agg[i][2]/2) + ")\n"

        return s


    def run(self):
        print("Experiment running " + str(self.episodes) + " episodes.")
        results = []
        env = self.make_env()
        for i in range(self.episodes):
            results.append(self.run_episode(i, env))
        return self.str_aggregate(results)

    def run_episode(self, episode_num, env):
        print("Running episode " + str(episode_num) + "...")
        learner = LinearApproximatedOnlineLearner(self.features, self.actions)
        test_num = 0
        results = []
        for i in range(self.train_iters):
            if i % self.test_freq == 0:
                results.append(self.run_test(test_num, i, learner, env))
                test_num = test_num + 1

            self.run_train_iter(i, learner, env)
        return results


    def run_train_iter(self, iter_num, learner, env):
        print("Running training iter " + str(iter_num) + "...")
        observation = env.reset()
        done = False
        t = 0
        while not done:
            action = learner.act(observation, self.exploration, self.exploration_param)
            observation_next, reward, done, info = env.step(self.actions.env_action(observation, action))
            # Possibly add condition here to check valid rewards 
            if (not done) or reward > 1:
                learner.update(observation, action, reward, observation_next)
                #if done:
                #print(str(t))
                #print("State:")
                #print(features.state_feature_str(observation))
                #print("Action: " + actions.env_action_str(observation, action))
                #print(str(t) + " Reward: " + str(reward) + "\n\n")

            observation = observation_next
            t = t + 1

    def run_test(self, test_num, train_iter, learner, env):
        print("Running test " + str(test_num) + "...")
        avg_reward = 0
        for i in range(self.test_iters):
            observation = env.reset()
            done = False
            while not done:
                action = learner.act(observation)
                observation, reward, done, info = env.step(self.actions.env_action(observation, action)) 
                avg_reward = avg_reward + reward
        avg_reward = avg_reward/self.test_iters
        return [train_iter, avg_reward]

