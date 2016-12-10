import gym
import gym_soccer
import numpy
import math

from actions_soccer import ActionSpaceSoccerSimple 
from actions_soccer import SoccerDirection
from learner import ExplorationStrategy
from experiment_soccer import ExperimentSoccer

EPISODES = 10
TRAIN_ITERS = 101
TEST_ITERS = 10
TEST_FREQ = 10

ACTION_TURN_DIRS = [SoccerDirection.BALL]
ACTION_KICK_DIRS_DEFAULT = [SoccerDirection.GOAL_CENTER]
ACTION_KICK_DIRS_GOALIE = [SoccerDirection.GOAL_CENTER, SoccerDirection.GOALIE_LEFT, SoccerDirection.GOALIE_RIGHT]



# Simplest
#explore = ExplorationStrategy.EPSILON_GREEDY
#explore_param = 0.1
#dash_powers = [0,50]
#kick_powers = [10]
#actions = ActionSpaceSoccerSimple(ACTION_TURN_DIRS, dash_powers, kick_powers, ACTION_KICK_DIRS_DEFAULT)
#interact_feats = True
#goalie = False
#e = ExperimentSoccer(EPISODES, TRAIN_ITERS, TEST_ITERS, TEST_FREQ, actions, explore, explore_param, interact_feats, goalie)
#results = e.run()
#print(results)
#(0,-0.000814766159879)+-(0,0.0055527559558)
#(10,4.26315060474)+-(0,0.102753590201)
#(20,3.56539393759)+-(0,0.138693081872)
#(30,4.121938397)+-(0,0.183225443109)
#(40,4.85534530964)+-(0,0.166365707801)
#(50,4.41841162037)+-(0,0.109481613714)
#(60,4.32011032639)+-(0,0.0994841259775)
#(70,4.01205929493)+-(0,0.116822058032)
#(80,3.90162751505)+-(0,0.110558224404)
#(90,4.06619576837)+-(0,0.144115013309)
#(100,4.02593671208)+-(0,0.150053712366)


# Add a few dash and kick options
#explore = ExplorationStrategy.EPSILON_GREEDY
#explore_param = 0.1
#dash_powers = [0,50,100] 
#kick_powers = [10,25]
#actions = ActionSpaceSoccerSimple(ACTION_TURN_DIRS, dash_powers, kick_powers, ACTION_KICK_DIRS_DEFAULT)
#interact_feats = True
#goalie = False
#e = ExperimentSoccer(EPISODES, TRAIN_ITERS, TEST_ITERS, TEST_FREQ, actions, explore, explore_param, interact_feats, goalie)
#results = e.run()
#print(results)
#(0,0.0139213616313)+-(0,0.00605800643234)
#(10,5.21402583654)+-(0,0.187084871842)
#(20,5.25896926465)+-(0,0.142880516312)
#(30,5.33567953492)+-(0,0.161937560097)
#(40,4.90326815329)+-(0,0.194554987329)
#(50,5.27385891907)+-(0,0.152507988304)
#(60,4.97618586938)+-(0,0.193071627117)
#(70,4.99458909049)+-(0,0.156493108997)
#(80,4.91115058516)+-(0,0.207643011157)
#(90,5.34108563205)+-(0,0.163342665673)
#(100,5.23276049037)+-(0,0.169805301043)


# Many dash powers
explore = ExplorationStrategy.EPSILON_GREEDY
explore_param = 0.1
dash_powers = [0,10,25,50,100]
kick_powers = [10,25]
actions = ActionSpaceSoccerSimple(ACTION_TURN_DIRS, dash_powers, kick_powers, ACTION_KICK_DIRS_DEFAULT)
interact_feats = True
goalie = False
e = ExperimentSoccer(EPISODES, TRAIN_ITERS, TEST_ITERS, TEST_FREQ, actions, explore, explore_param, interact_feats, goalie)
results = e.run()
print(results)


# Many powers
#explore = ExplorationStrategy.EPSILON_GREEDY
#explore_param = 0.1
#dash_powers = [0,10,25,50,100]
#kick_powers = [10,25,50,100]
#actions = ActionSpaceSoccerSimple(ACTION_TURN_DIRS, dash_powers, kick_powers, ACTION_KICK_DIRS_DEFAULT)
#interact_feats = True
#goalie = False
#e = ExperimentSoccer(EPISODES, TRAIN_ITERS, TEST_ITERS, TEST_FREQ, actions, explore, explore_param, interact_feats, goalie)
#results = e.run()
#print(results)
# (0,0.0129882913367)+-(0,0.00684361117802)
# (10,2.08055943596)+-(0,0.402293659012)
# (20,1.97003436795)+-(0,0.355154507838)
# (30,2.55872762439)+-(0,0.429800309704)
# (40,2.04461856576)+-(0,0.390824856237)
# (50,2.15929128561)+-(0,0.414352693603)
# (60,2.21235553938)+-(0,0.408530060595)
# (70,2.46500122008)+-(0,0.391114373194)
# (80,2.17663702742)+-(0,0.407359136958)
# (90,2.16252911184)+-(0,0.376946042705)
# (100,2.24595822585)+-(0,0.385793581283)


# More kick directions
# Increase exploration
# Change to softmax?
# Give higher kick power
# Give large action space
# Turn off interactions 

################## GOALIE ##################



