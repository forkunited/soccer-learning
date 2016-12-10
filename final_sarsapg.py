# QAC with Sarsa for the Soccer environment
import gym
import gym_soccer
import numpy as np
import math

def sarsa_gradient():
	env = gym.make('SoccerEmptyGoal-v0') 	# create env
	num_features = 58 	# all state features are used here, so num_features = 15
	num_ang = 15 	# angles in radians are extracted as arccos(cos angle ) * sign(sin angle), this is the number of angles
	total_features = num_features + num_ang + 1 	# +1 from bias unit

	th_kick = np.zeros(total_features) 	# initialize value function approximators
	th_kick[12] = 1		# kicking value depends on kickable
	th_dash = np.zeros(total_features)
	th_turn = np.zeros(total_features)
	th_turn[73] = -1	# turn is biased to be less valuable than other actions

	mu_p_d = np.zeros(total_features) 	# mean of dash power approximator
	mu_p_d[7] = 50 		# initialized as a linear function of stamina
	mu_p_d[73] = 50
	mu_a_d = np.zeros(total_features) 	# mean of dash angle approximator
	mu_a_d[71] = 360 	# initialized to the angle towards ball
	mu_a_t = np.zeros(total_features) 	# mean of turn angle approximator
	mu_a_t[71] = 360 	# initialized to the angle towards ball 
	mu_p_k = np.zeros(total_features)	# mean of kick power approximator
	mu_p_k[73] = 20		# biased to 10
	mu_a_k = np.zeros(total_features)	# mean of kick angle approximator
	mu_a_k[60] = 360	# initialized to the angle towards goal
	# sigmas chosen small and constant - we consider an almost deterministic policy in this version, and sigma is not updated
	sigma_a_d = 0.1
	sigma_a_k = 0.1
	sigma_t = 0.1
	sigma_p_k = 0.1
	sigma_p_d = 0.1


	alpha = 10**-6	# learning rate, different for policy and value
	val_pol = 50
	alpha_p = val_pol * alpha
	epsilon = 0.1 		# parameter for epsilon greedy search
	rewards = []		# reward vector initialization
	for i_episode in range(500): # for 500 episodes 
		observation = env.reset()	# reset environment at the beginning of each episode
		flag = 1			# flag that prevents update in the first step
		total_reward = 0		# total reward in episode initialized to 0
		if i_episode == 399:		# no learning in last 100 episodes, only measuring performance
			epsilon = 0
		for t in range(500):		# max number of time steps
			if i_episode >= 399:	# visualize last 100 episodes to observe what has been learned		
				env.render() # visualization
			ang_feat = extract_ang(observation) # extract angles + bias and append them to the original observation vector
			observation = np.append(observation, ang_feat)
			value_kick = np.dot(observation,th_kick) # compute values via dot product
			value_dash = np.dot(observation,th_dash)
			value_turn = np.dot(observation,th_turn)
			if observation[12] <= 0:
				value_kick = -1000000		# to increase convergence speed, set kick value very small when ball is not kickable
			u = np.random.uniform()
			if u <= epsilon:	# epsilon greedy search
				u = np.random.uniform()
				if observation[12] > 0: # if kickable
					a = 2 # kick
				 	n = np.random.normal()
					ang = np.dot(observation,mu_a_k) + sigma_a_k * n	# sample angle from Gaussian policy
					if ang < -180:
						ang = -180
					elif ang > 180:
						ang = 180
					n = np.random.normal()
					powa = np.dot(observation, mu_p_k) + sigma_p_k * n	# sample power from Gaussian policy
					if powa < 0:
						powa = 0
					elif powa > 100:
						powa = 100
					action = [a, [0], [0], [0], [powa], [ang]]	# create action vector
					Q_current = value_kick
				else:
					if u < (1/2):
						a = 0 # dash
						n = np.random.normal()
						ang = np.dot(observation,mu_a_d) + sigma_a_d * n	# sample angle from Gaussian policy
						if ang < -180:
							ang = -180
						elif ang > 180:
							ang = 180
				
						n = np.random.normal()
						powa = np.dot(observation, mu_p_d) + sigma_a_d * n	# sample power from Gaussian policy
						if powa < -100:
							powa = -100
						elif powa > 100:
							powa = 100
						action = [ a, [powa], [ang], [0], [0], [0]] 	# create action vector
						Q_current = value_dash
					else:
						a = 1 # turn
						n = np.random.normal()
						ang = np.dot(observation,mu_a_t) + sigma_t * n		# sample angle from Gaussian policy
						if ang < -180:
							ang = -180
						elif ang > 180:
							ang = 180
						action =[ a, [0], [0], [ang], [0], [0]]		# create action vector
						Q_current = value_turn
			elif u > epsilon:	# exploit
				best = max(value_dash, value_turn, value_kick)		# best action
				if best ==  value_dash:		# find best action and execute
					a = 0 # dash
					n = np.random.normal() 
					ang = np.dot(observation,mu_a_d) + sigma_a_d * n
					if ang < -180:
						ang = -180
					elif ang > 180:
						ang = 180
				
					n = np.random.normal()
					powa = np.dot(observation, mu_p_d) + sigma_a_d * n
					if powa < 0:
						powa = 0
					elif powa > 100:
						powa = 100
					action = [ a, [powa], [ang], [0], [0], [0]]
					Q_current = value_dash
				elif best == value_turn:
					a = 1 # turn
					n = np.random.normal()
					ang = np.dot(observation,mu_a_t) + sigma_t * n
					if ang < -180:
						ang = -180
					elif ang > 180:
						ang = 180
					action =[ a, [0], [0], [ang], [0], [0]]
					Q_current = value_turn
				else:
					a = 2 # kick
				 	n = np.random.normal()
					ang = np.dot(observation,mu_a_k) + sigma_a_k * n
					if ang < -180:
						ang = -180
					elif ang > 180:
						ang = 180
					n = np.random.normal()
					powa = np.dot(observation, mu_p_k) + sigma_p_k * n
					if powa < 0:
						powa = 0
					elif powa > 100:
						powa = 100
					action = [a, [0], [0], [0], [powa], [ang]]
					Q_current = value_kick
	
			if flag == 0: # check whether first step or not
				previous_a = previous_action[0]		# remember the action from previous time step
				delta = previous_r + Q_current - Q_previous # TD error
				if previous_a == 0:	 # update parameters according to the update equations
					previous_pow = previous_action[1]
					previous_ang = previous_action[2]
					mu_a_d += np.multiply(alpha * ( previous_ang - np.dot(previous_obs, mu_a_d)) * delta, previous_obs) / sigma_a_d**2
					mu_p_d += np.multiply(alpha * ( previous_pow - np.dot(previous_obs, mu_p_d)) * delta, previous_obs) / sigma_p_d**2
					th_dash += np.multiply(alpha_p * delta, previous_obs) 
				elif previous_a == 1:	
					previous_ang = previous_action[3]
					mu_a_t += np.multiply(alpha * ( previous_ang - np.dot(previous_obs, mu_a_t)) * delta, previous_obs) / sigma_t**2
					th_turn += np.multiply(alpha_p * delta, previous_obs)
				else:
					previous_pow = previous_action[4]
					previous_ang = previous_action[5]
					mu_a_k += np.multiply(alpha * ( previous_ang - np.dot(previous_obs, mu_a_k)) * delta, previous_obs) / sigma_a_k**2
					mu_p_k += np.multiply(alpha * ( previous_pow - np.dot(previous_obs, mu_p_k)) * delta, previous_obs) / sigma_p_k**2
					th_kick += np.multiply(alpha_p * delta, previous_obs) 

						
			flag = 0
			previous_action = action 	# store action, observation, Q and reward to be used in the update in next time step
			previous_obs = observation
			Q_previous = Q_current
			observation, reward, done, info = env.step(action)	# execute action and observe next state and reward
			previous_r = reward
			total_reward += reward		# update total reward in episode
			if done:	# if current episode is finished
				delta = reward		# update is slightly different when we are at terminal state
				if reward > 4.5:		# if a goal was scored
					if previous_a == 0: # update parameters
						previous_pow = previous_action[1]
						previous_ang = previous_action[2]
						mu_a_d += np.multiply(alpha * ( previous_ang - np.dot(previous_obs, mu_a_d)) * delta, previous_obs) / sigma_a_d**2
						mu_p_d += np.multiply(alpha * ( previous_pow - np.dot(previous_obs, mu_p_d)) * delta, previous_obs) / sigma_p_d**2
						th_dash += np.multiply(alpha_p * delta, previous_obs) 
					elif previous_a == 1:
						previous_ang = previous_action[3]
						mu_a_t += np.multiply(alpha * ( previous_ang - np.dot(previous_obs, mu_a_t)) * delta, previous_obs) / sigma_t**2
						th_turn += np.multiply(alpha_p * delta, previous_obs)
					else:
						previous_pow = previous_action[4]
						previous_ang = previous_action[5]
						mu_a_k += np.multiply(alpha * ( previous_ang - np.dot(previous_obs, mu_a_k)) * delta, previous_obs) / sigma_a_k**2
						mu_p_k += np.multiply(alpha * ( previous_pow - np.dot(previous_obs, mu_p_k)) * delta, previous_obs) / sigma_p_k**2
						th_kick += np.multiply(alpha_p * delta, previous_obs) 				
				
            			print("Episode finished after {} timesteps".format(t+1))
				rewards = np.append(rewards, total_reward)	# append reward from this episode to the rewards vector
            			break
				

	print rewards
	programPause = raw_input("Press the <ENTER> key to continue...")
	return th_dash, th_turn, th_kick, mu_a_d, mu_p_d, mu_a_t, mu_a_k, mu_p_k

def extract_ang(obs):	 # extracts normalized angular features in radians	
	ang_features = []
	ang = np.arccos(obs[3]) * np.sign(obs[2]) / (math.pi)
	ang_features.append(ang) 	# self vel ang
	ang = np.arccos(obs[6]) * np.sign(obs[5]) / (math.pi)
	ang_features.append(ang) 	# self ang
	ang = np.arccos(obs[14]) * np.sign(obs[13]) / (math.pi)
	ang_features.append(ang) 	# goal center ang
	ang = np.arccos(obs[17]) * np.sign(obs[16]) / (math.pi) 
	ang_features.append(ang) 	# top goal post
	ang = np.arccos(obs[20]) * np.sign(obs[19]) / (math.pi)
	ang_features.append(ang) 	# bottom goal post
	ang = np.arccos(obs[23]) * np.sign(obs[22]) / (math.pi)
	ang_features.append(ang) 	# center of pen box line
	ang = np.arccos(obs[26]) * np.sign(obs[24]) / (math.pi)
	ang_features.append(ang) 	# top corner of pen box
	ang = np.arccos(obs[29]) * np.sign(obs[27]) / (math.pi)
	ang_features.append(ang) 	# bot corner of pen box
	ang = np.arccos(obs[32]) * np.sign(obs[30]) / (math.pi) 
	ang_features.append(ang) 	# left mid point
	ang = np.arccos(obs[35]) * np.sign(obs[34]) / (math.pi) 
	ang_features.append(ang) 	# top left corner
	ang = np.arccos(obs[38]) * np.sign(obs[37]) / (math.pi)
	ang_features.append(ang) 	# top right corner
	ang = np.arccos(obs[41]) * np.sign(obs[40]) / (math.pi)
	ang_features.append(ang) 	# bot right corner
	ang = np.arccos(obs[44]) * np.sign(obs[43]) / (math.pi) 
	ang_features.append(ang) 	# bot left corner
	ang = np.arccos(obs[52]) * np.sign(obs[51]) / (math.pi)
	ang_features.append(ang) 	# angle to ball
	ang = np.arccos(obs[57]) * np.sign(obs[56]) / (math.pi)
	ang_features.append(ang) 	# ball vel angle
	ang_features.append(1) 		# bias unit
	return ang_features


	

th_dash, th_turn, th_kick, mu_a_d, mu_p_d, mu_a_t, mu_a_k, mu_p_k = sarsa_gradient()


