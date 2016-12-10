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


#name = "Many dash powers\n"
#explore = ExplorationStrategy.EPSILON_GREEDY
#explore_param = 0.1
#dash_powers = [0,10,25,50,100]
#kick_powers = [10,25]
#actions = ActionSpaceSoccerSimple(ACTION_TURN_DIRS, dash_powers, kick_powers, ACTION_KICK_DIRS_DEFAULT)
#interact_feats = True
#goalie = False
#e = ExperimentSoccer(EPISODES, TRAIN_ITERS, TEST_ITERS, TEST_FREQ, actions, explore, explore_param, interact_feats, goalie)
#results = e.run()
#print(name + results)
#print("\n")
#(0,-0.0110938651699)+-(0,0.00752244894382)
#(10,2.54324215814)+-(0,0.435110749047)
#(20,2.83977689712)+-(0,0.452071366696)
#(30,3.0273139208)+-(0,0.44008587956)
#(40,2.99566643039)+-(0,0.461459603739)
#(50,3.44626365762)+-(0,0.395226427443)
#(60,3.28091710456)+-(0,0.377975662213)
#(70,3.14896683447)+-(0,0.446466915329)
#(80,3.31562318158)+-(0,0.411528412018)
#(90,3.12509470702)+-(0,0.46397125091)
#(100,3.04605201108)+-(0,0.442692573042)


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

#name = "Increased exploration\n"
#explore = ExplorationStrategy.EPSILON_GREEDY
#explore_param = 0.2
#dash_powers = [0,50,100]
#kick_powers = [10,25]
#actions = ActionSpaceSoccerSimple(ACTION_TURN_DIRS, dash_powers, kick_powers, ACTION_KICK_DIRS_DEFAULT)
#interact_feats = True
#goalie = False
#e = ExperimentSoccer(EPISODES, TRAIN_ITERS, TEST_ITERS, TEST_FREQ, actions, explore, explore_param, interact_feats, goalie)
#results = e.run()
#print(name + results)
#print("\n")
#(0,0.00980409289447)+-(0,0.00579201265907)
#(10,4.59483112006)+-(0,0.267620626087)
#(20,5.33835827381)+-(0,0.144417326733)
#(30,5.11908148844)+-(0,0.159450696309)
#(40,5.46894926491)+-(0,0.104931341139)
#(50,5.09892080248)+-(0,0.200000830391)
#(60,5.23329679777)+-(0,0.163763580552)
#(70,5.43004449107)+-(0,0.126677901855)
#(80,4.96025467361)+-(0,0.216469126061)
#(90,5.14550423478)+-(0,0.197584409878)
#(100,5.16483061982)+-(0,0.184497231464)

#name = "No interactions\n"
#explore = ExplorationStrategy.EPSILON_GREEDY
#explore_param = 0.1
#dash_powers = [0,50,100]
#kick_powers = [10,25]
#actions = ActionSpaceSoccerSimple(ACTION_TURN_DIRS, dash_powers, kick_powers, ACTION_KICK_DIRS_DEFAULT)
#interact_feats = False
#goalie = False
#e = ExperimentSoccer(EPISODES, TRAIN_ITERS, TEST_ITERS, TEST_FREQ, actions, explore, explore_param, interact_feats, goalie)
#results = e.run()
#print(name + results)
#print("\n") 
#(0,0.026653308659)+-(0,0.00639176096557)
#(10,4.61915316765)+-(0,0.143657232992)
#(20,4.36605370177)+-(0,0.185510275589)
#(30,4.66516471978)+-(0,0.151326359313)
#(40,4.62845695004)+-(0,0.197052849)
#(50,4.42432950539)+-(0,0.160316629962)
#(60,4.73398611037)+-(0,0.155812835791)
#(70,4.35467003213)+-(0,0.184873624838)
#(80,4.37352688111)+-(0,0.202703451003)
#(90,4.18137995282)+-(0,0.18811481112)
#(100,4.2270549526)+-(0,0.188178624871)


################## GOALIE ##################

# Simplest
#explore = ExplorationStrategy.EPSILON_GREEDY
#explore_param = 0.1
#dash_powers = [0,50,100]
#kick_powers = [10,25]
#actions = ActionSpaceSoccerSimple(ACTION_TURN_DIRS, dash_powers, kick_powers, ACTION_KICK_DIRS_DEFAULT)
#interact_feats = True
#goalie = True
#e = ExperimentSoccer(EPISODES, TRAIN_ITERS, TEST_ITERS, TEST_FREQ, actions, explore, explore_param, interact_feats, goalie)
#results = e.run()
#print(results)
#(0,0.99807580535)+-(0,0.00127515280448)
#(10,1.00320760976)+-(0,0.00252425190463)
#(20,1.00118723814)+-(0,0.00171546319834)
#(30,1.00701183964)+-(0,0.00103459701444)
#(40,1.00579941608)+-(0,0.0020888771673)
#(50,1.01034369401)+-(0,0.00187612719103)
#(60,0.997938768393)+-(0,0.00189923386335)
#(70,1.00688532105)+-(0,0.00191217905626)
#(80,1.00813319659)+-(0,0.0015787103528)
#(90,1.00710613018)+-(0,0.00147285785506)
#(100,1.00644518867)+-(0,0.00161577566995)


# Goalie actions
#explore = ExplorationStrategy.EPSILON_GREEDY
#explore_param = 0.1
#dash_powers = [0,50,100]
#kick_powers = [10,25]
#actions = ActionSpaceSoccerSimple(ACTION_TURN_DIRS, dash_powers, kick_powers, ACTION_KICK_DIRS_GOALIE)
#interact_feats = True
#goalie = True
#e = ExperimentSoccer(EPISODES, TRAIN_ITERS, TEST_ITERS, TEST_FREQ, actions, explore, explore_param, interact_feats, goalie)
#results = e.run()
#print(results)
#(0,1.00477180715)+-(0,0.00180814464105)
#(10,1.00441971269)+-(0,0.00205323227901)
#(20,1.00527562274)+-(0,0.0019244520305)
#(30,1.00320094951)+-(0,0.00182312596748)
#(40,1.05105961981)+-(0,0.0257725816971)
#(50,1.00052416643)+-(0,0.00138779558689)
#(60,1.00565126883)+-(0,0.00139365099579)
#(70,1.00171884688)+-(0,0.00200670555854)
#(80,1.00479638847)+-(0,0.00187969300886)
#(90,1.0069708861)+-(0,0.00252194854878)
#(100,1.00459237222)+-(0,0.00222398134039)


# Kick harder
#explore = ExplorationStrategy.EPSILON_GREEDY
#explore_param = 0.1
#dash_powers = [0,50,100]
#kick_powers = [10,25,100]
#actions = ActionSpaceSoccerSimple(ACTION_TURN_DIRS, dash_powers, kick_powers, ACTION_KICK_DIRS_GOALIE)
#interact_feats = True
#goalie = True
#e = ExperimentSoccer(EPISODES, TRAIN_ITERS, TEST_ITERS, TEST_FREQ, actions, explore, explore_param, interact_feats, goalie)
#results = e.run()
#print(results)
#(0,0.997542101225)+-(0,0.00182305901399)
#(10,1.01551149991)+-(0,0.00209568416854)
#(20,1.01743445989)+-(0,0.003209899663)
#(30,1.06115232509)+-(0,0.0253523644688)
#(40,1.07173604296)+-(0,0.0255654443934)
#(50,1.05943218932)+-(0,0.0267464165468)
#(60,1.06897262634)+-(0,0.0262479448175)
#(70,1.06108485644)+-(0,0.0254651639777)
#(80,1.02246129606)+-(0,0.00184776526857)
#(90,1.0126397431)+-(0,0.00233037718414)
#(100,1.01540701183)+-(0,0.00252328769452)
