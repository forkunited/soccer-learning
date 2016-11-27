import gym
import gym_soccer
import numpy as np
import math
# this probably has some bugs because I was trying a couple of new things but should still give you a baseline (and you can always remove them). I couldn't find my backup, I may have accidentally overwritten :(
def sarsa_gradient (): # th_x is the value function estimation parameters, mu_x_y is the set of parameters for the mean of the Gaussian for the x parameter of yth action (angle of kick for ex)
	env = gym.make('SoccerEmptyGoal-v0') # create env
	num_features = 58
	num_ang = 15 # I was trying to extract angles as arccos(cos angle ) * sign(sin angle), this is the number of angles
	th_kick = np.random.normal(0,1,num_features + num_ang)
	th_kick[12] = 3 # I was experimenting with a clever initialization (13 should be the feature that states whether ball is kickable or not)
	th_dash = np.random.normal(0,1,num_features + num_ang)
	th_turn = np.random.normal(0,1,num_features + num_ang)
	mu_a_k = np.zeros(num_ang) # again, experimenting with the angle extraction + clever initialization
	mu_a_k[2] = 0.5 # these should be the angles corresponding to the agent's angle with the goal
	mu_a_k[3] = 0.25
	mu_a_k[4] = 0.25
	mu_a_d = np.zeros(num_ang)
	mu_a_d[13] = 1 # this should be agent's angle with the ball
	mu_a_t = np.zeros(num_ang)
	mu_a_t[13] = 1
	mu_p_d = np.random.normal(0,1,num_features + num_ang) # initialize all else randomly (we can also do zeros)
	mu_p_k = np.random.normal(0,1,num_features + num_ang)
	sigma_a_d = np.random.normal(0,1,num_features + num_ang)
	sigma_a_t = np.random.normal(0,1,num_features + num_ang)
	sigma_a_k = np.random.normal(0,1,num_features + num_ang)
	sigma_p_d = np.random.normal(0,1,num_features + num_ang)
	sigma_p_k = np.random.normal(0,1,num_features + num_ang)
	alpha = 0.001 # learning rate, different for policy and value
	val_pol = 30
	alpha_p = val_pol * alpha
	for i_episode in range(1000): # for 1000 episodes 
		observation = env.reset()
		flag = 1
		for t in range(200): # 200 = max. number of time steps I used for each episode. you go out of time if you don't touch the ball for 100 instants anyway (which I rarely did) so this was fine
			# env.render() # visualization
			ang_feat = extract_ang(observation) # extract angles and append them to the original observation vector
			observation = np.append(observation, ang_feat) 
			value_kick = np.dot(observation,th_kick) # compute values via dot product
			value_dash = np.dot(observation,th_dash)
			value_turn = np.dot(observation,th_turn)
			denom = np.exp(value_kick) + np.exp(value_dash) + np.exp(value_turn) # softmax

			soft_actions = []			
			soft_actions.append(np.exp(value_dash) / denom)
			soft_actions.append(np.exp(value_turn) / denom)
			soft_actions.append(np.exp(value_kick) / denom)
			u = np.random.uniform()
			# print state_t.shape 
			if u < soft_actions[0]:
				a = 0 # dash
				n = np.random.normal() # sample from the Gaussians
				ang = np.dot(ang_feat,mu_a_d) + np.dot(observation, sigma_a_d) * n
				n = np.random.normal()
				powa = np.dot(observation, mu_p_d) + np.dot(observation, sigma_p_d) * n
				action = [ a, [powa], [ang], [0], [0], [0]] # create the action vector
				Q_current = value_dash
				print action
			elif u < soft_actions[0] + soft_actions[1]:
				a = 1 # turn
				n = np.random.normal()
				ang = np.dot(ang_feat,mu_a_t) + np.dot(observation, sigma_a_t) * n
				action =[ a, [0], [0], [ang], [0], [0]]
				Q_current = value_turn
				print action
			else:
				a = 2 # kick
			 	n = np.random.normal()
				ang = np.dot(ang_feat,mu_a_k) + np.dot(observation, sigma_a_k) * n
				n = np.random.normal()
				powa = np.dot(observation, mu_p_k) + np.dot(observation, sigma_p_k) * n
				action = [a, [0], [0], [0], [powa], [ang]]
				Q_current = value_kick

			if flag == 0: # I am using Sarsa. There is no update in the first time step of each episode, this checks that.
				previous_a = previous_action[0]
				delta = previous_r + Q_current - Q_previous # TD error
				if previous_a == 0: # update the parameters corresponding to the action chosen in the previous time step
					previous_pow = previous_action[1]
					previous_ang = previous_action[2]
					sigma_a_d += np.multiply(alpha / (np.dot(previous_obs, sigma_a_d)) * ((previous_ang - np.dot(previous_af, mu_a_d))**2 - np.dot(previous_obs, sigma_a_d)**2) * delta, previous_obs) # again these should be different for the vanilla version, I used np.multiply to multiply a scalar with a vector
			
					mu_a_d += np.multiply(alpha * ( previous_ang - np.dot(previous_af, mu_a_d)) * delta, previous_af)
					sigma_p_d += np.multiply(alpha / (np.dot(previous_obs, sigma_p_d)) * ((previous_pow - np.dot(previous_obs, mu_p_d))**2 - np.dot(previous_obs, sigma_p_d)**2) * delta, previous_obs)
			
					mu_p_d += np.multiply(alpha * ( previous_pow - np.dot(previous_obs, mu_p_d)) * delta, previous_obs)
					th_dash += np.multiply(alpha_p * delta, previous_obs) 
				elif previous_a == 1:
					previous_ang = previous_action[3]
					sigma_a_t += np.multiply(alpha / (np.dot(previous_obs, sigma_a_t)) * ((previous_ang - np.dot(previous_af, mu_a_t))**2 - np.dot(previous_obs, sigma_a_t)**2) * delta, previous_obs)
			
					mu_a_t += np.multiply(alpha * ( previous_ang - np.dot(previous_af, mu_a_t)) * delta, previous_af)

					th_turn += np.multiply(alpha_p * delta, previous_obs)
				else:
					previous_pow = previous_action[4]
					previous_ang = previous_action[5]
					sigma_a_k += np.multiply(alpha / (np.dot(previous_obs, sigma_a_k)) * ((previous_ang - np.dot(previous_af, mu_a_k))**2 - np.dot(previous_obs, sigma_a_k)**2) * delta, previous_obs)
			
					mu_a_k += np.multiply(alpha * ( previous_ang - np.dot(previous_af, mu_a_k)) * delta, previous_af)
					sigma_p_k += np.multiply(alpha / (np.dot(previous_obs, sigma_p_k)) * ((previous_pow - np.dot(previous_obs, mu_p_k))**2 - np.dot(previous_obs, sigma_p_k)**2) * delta, previous_obs)
			
					mu_p_k += np.multiply(alpha * ( previous_pow - np.dot(previous_obs, mu_p_k)) * delta, previous_obs)
					th_kick += np.multiply(alpha_p * delta, previous_obs) 

						
			flag = 0
			previous_action = action # store action, obs, Q, r etc. as they will be used for update in the next step
			previous_obs = observation
			previous_af = ang_feat
			Q_previous = Q_current
			# previous_state_t = state_t

			observation, reward, done, info = env.step(action)
			previous_r = reward
			# print reward
			if done:
            			print("Episode finished after {} timesteps".format(t+1))
            			break
			# elif t%20 == 0:
				# print t
	average_reward = 0
	# this part is just to evaluate how good a policy is after learning. there is no update here, other than that everything should be the same. 
	for i_episode in range(100):
		observation = env.reset()
		episodic_reward = 0
    		for t in range(200):
        		env.render() # I visualized to be able to diagnose errors
        		value_kick = np.dot(observation,th_kick)
			value_dash = np.dot(observation,th_dash)
			value_turn = np.dot(observation,th_turn)
			denom = np.exp(value_kick) + np.exp(value_dash) + np.exp(value_turn)

			soft_actions = []			
			soft_actions.append(np.exp(value_dash) / denom)
			soft_actions.append(np.exp(value_turn) / denom)
			soft_actions.append(np.exp(value_kick) / denom)
			u = np.random.uniform()
			# print state_t.shape 
			if u < soft_actions[0]:
				a = 0
				n = np.random.normal()
				ang = np.dot(observation,mu_a_d) + np.dot(observation, sigma_a_d) * n
				n = np.random.normal()
				powa = np.dot(observation, mu_p_d) + np.dot(observation, sigma_p_d) * n
				action = [ a, [powa], [ang], [0], [0], [0]]
				Q_current = value_dash
			elif u < soft_actions[0] + soft_actions[1]:
				a = 1
				n = np.random.normal()
				ang = np.dot(observation,mu_a_t) + np.dot(observation, sigma_a_t) * n
				action =[ a, [0], [0], [ang], [0], [0]]
				Q_current = value_turn
			else:
				a = 2
				n = np.random.normal()
				ang = np.dot(observation,mu_a_k) + np.dot(observation, sigma_a_k) * n
				n = np.random.normal()
				powa = np.dot(observation, mu_p_k) + np.dot(observation, sigma_p_k) * n
				action = [a, [0], [0], [0], [powa], [ang]]
				Q_current = value_kick
        		
			# print action
        		observation, reward, done, info = env.step(action)
			episodic_reward += reward
			if done:
            			print("Episode finished after {} timesteps".format(t+1))
            			break
		average_reward += episodic_reward / 100
	programPause = raw_input("Press the <ENTER> key to continue...") # I paused to have more time analyzing the learned policy
        		
	print average_reward
	
	return th_dash, th_turn, th_kick, mu_a_d, sigma_a_d, mu_p_d, sigma_p_d, mu_a_t, sigma_a_t, mu_a_k, sigma_a_k, mu_p_k, sigma_p_k

def extract_ang(obs): # extracts angular features as previously mentioned
	ang_features = []
	ang = np.arccos(obs[3]) * np.sign(obs[2]) / (2*math.pi) * 360
	ang_features.append(ang) # self vel ang
	ang = np.arccos(obs[6]) * np.sign(obs[5]) / (2*math.pi) * 360
	ang_features.append(ang) # self ang
	ang = np.arccos(obs[14]) * np.sign(obs[13]) / (2*math.pi) * 360
	ang_features.append(ang) # goal center ang
	ang = np.arccos(obs[17]) * np.sign(obs[16]) / (2*math.pi) * 360
	ang_features.append(ang) # top goal post
	ang = np.arccos(obs[20]) * np.sign(obs[19]) / (2*math.pi) * 360
	ang_features.append(ang) # bottom goal post
	ang = np.arccos(obs[23]) * np.sign(obs[22]) / (2*math.pi) * 360
	ang_features.append(ang) # center of pen box line
	ang = np.arccos(obs[26]) * np.sign(obs[24]) / (2*math.pi) * 360
	ang_features.append(ang) # top corner of pen box
	ang = np.arccos(obs[29]) * np.sign(obs[27]) / (2*math.pi) * 360
	ang_features.append(ang) # bot corner of pen box
	ang = np.arccos(obs[32]) * np.sign(obs[30]) / (2*math.pi) * 360
	ang_features.append(ang) # left mid point
	ang = np.arccos(obs[35]) * np.sign(obs[34]) / (2*math.pi) * 360
	ang_features.append(ang) # top left corner
	ang = np.arccos(obs[38]) * np.sign(obs[37]) / (2*math.pi) * 360
	ang_features.append(ang) # top right corner
	ang = np.arccos(obs[41]) * np.sign(obs[40]) / (2*math.pi) * 360
	ang_features.append(ang) # bot right corner
	ang = np.arccos(obs[44]) * np.sign(obs[43]) / (2*math.pi) * 360
	ang_features.append(ang) # bot left corner
	ang = np.arccos(obs[52]) * np.sign(obs[51]) / (2*math.pi) * 360
	ang_features.append(ang) # angle to ball
	ang = np.arccos(obs[57]) * np.sign(obs[56]) / (2*math.pi) * 360
	ang_features.append(ang) # ball vel angle
	# deal with against keeper later.
	return ang_features

th_dash, th_turn, th_kick, mu_a_d, sigma_a_d, mu_p_d, sigma_p_d, mu_a_t, sigma_a_t, mu_a_k, sigma_a_k, mu_p_k, sigma_p_k = sarsa_gradient()

