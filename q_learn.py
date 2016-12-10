# Q-learning
import gym
import gym_soccer
import numpy as np
import math

def q_gradient():
	env = gym.make('SoccerEmptyGoal-v0') # create env
	num_features = 58
	num_ang = 15 # I was trying to extract angles as arccos(cos angle ) * sign(sin angle), this is the number of angles
	total_features = num_features + num_ang + 1
	# th_dash = np.zeros(num_features + num_ang)

	th_kick = np.zeros(total_features)
	# th_kick = 1
	th_dash = np.zeros(total_features)
	th_turn = np.zeros(total_features)
	th_turn[73] = -10

	mu_p_d = np.zeros(total_features)
	mu_p_d[73] = 70
	# mu_a_d = np.zeros(num_ang)
	# mu_a_t = np.zeros(num_ang)
	mu_a_d = np.zeros(total_features)
	mu_a_d
	mu_a_d[71] = 360
	mu_a_t = np.zeros(total_features)
	mu_a_t[71] = 360
	mu_p_k = np.zeros(total_features)
	mu_p_k[73] = 10
	mu_a_k = np.zeros(total_features)
	mu_a_k[60] = 360
	# SEPARATE SIGMA FOR POWER AND ANGLE : OTHERWISE IT JUST CAN'T MOVE PROPERLY.
	sigma_a_d = 0
	sigma_a_k = 0 
	sigma_t = 0
	sigma_p_k = 0
	sigma_p_d = 0


	alpha = 10**-6 # learning rate, different for policy and value
	val_pol = 10
	alpha_p = val_pol * alpha
	epsilon = 0.1

	for i_episode in range(300): # for 1000 episodes 
		observation = env.reset()
		flag = 1
		if i_episode == 199:
			epsilon = 0
		for t in range(500): # 200 = max. number of time steps I used for each episode. you go out of time if you don't touch the ball for 100 instants anyway (which I rarely did) so this was fine
			env.render() # visualization
			ang_feat = extract_ang(observation) # extract angles and append them to the original observation vector
			observation = np.append(observation, ang_feat)
			value_kick = np.dot(observation,th_kick) # compute values via dot product
			value_dash = np.dot(observation,th_dash)
			value_turn = np.dot(observation,th_turn)
			# print value_dash, value_kick, value_turn
			if observation[12] <= 0:
				value_kick = -1000000
			# denom = np.exp(value_kick) + np.exp(value_dash) + np.exp(value_turn) # softmax

			# soft_actions = []			
			# soft_actions.append(np.exp(value_dash) / denom)
			# soft_actions.append(np.exp(value_turn) / denom)
			# soft_actions.append(np.exp(value_kick) / denom)
			u = np.random.uniform()
			if u <= epsilon:

				u = np.random.uniform()
				if observation[12] > 0:
					if u < (1/3):
						a = 0 # dash
						n = np.random.normal() # sample from the Gaussians
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
						action = [ a, [powa], [ang], [0], [0], [0]] # create the action vector
						Q_current = value_dash
						# print action
					elif u < (2/3):
						a = 1 # turn
						n = np.random.normal()
						ang = np.dot(observation,mu_a_t) + sigma_t * n
						if ang < -180:
							ang = -180
						elif ang > 180:
							ang = 180
						action =[ a, [0], [0], [ang], [0], [0]]
						Q_current = value_turn
						# print action
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
				else:
					if u < (1/2):
						a = 0 # dash
						n = np.random.normal() # sample from the Gaussians
						ang = np.dot(observation,mu_a_d) + sigma_a_d * n
						if ang < -180:
							ang = -180
						elif ang > 180:
							ang = 180
				
						n = np.random.normal()
						powa = np.dot(observation, mu_p_d) + sigma_a_d * n
						if powa < -100:
							powa = -100
						elif powa > 100:
							powa = 100
						action = [ a, [powa], [ang], [0], [0], [0]] # create the action vector
						Q_current = value_dash
						# print action
					else:
						a = 1 # turn
						n = np.random.normal()
						ang = np.dot(observation,mu_a_t) + sigma_t * n
						if ang < -180:
							ang = -180
						elif ang > 180:
							ang = 180
						action =[ a, [0], [0], [ang], [0], [0]]
						Q_current = value_turn
						# print action
			elif u > epsilon: 
				best = max(value_dash, value_turn, value_kick)
				if best ==  value_dash:
					a = 0 # dash
					n = np.random.normal() # sample from the Gaussians
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
					action = [ a, [powa], [ang], [0], [0], [0]] # create the action vector
					Q_current = value_dash
					# print action
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
					# print action
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
	
			if flag == 0: # I am using Sarsa. There is no update in the first time step of each episode, this checks that.
				previous_a = previous_action[0]
				delta = previous_r + Q_current - Q_previous # TD error
				if previous_a == 0: # update the parameters corresponding to the action chosen in the previous time step
					previous_pow = previous_action[1]
					previous_ang = previous_action[2]
					# sigma_a_d += np.multiply(alpha / (np.dot(previous_obs, sigma_a_d)) * ((previous_ang - np.dot(previous_af, mu_a_d))**2 - np.dot(previous_obs, sigma_a_d)**2) * delta, previous_obs) # again these should be different for the vanilla version, I used np.multiply to multiply a scalar with a vector
			
					mu_a_d += np.multiply(alpha * ( previous_ang - np.dot(previous_obs, mu_a_d)) * delta, previous_obs) # / sigma_a_d**2
					# sigma_p_d += np.multiply(alpha / (np.dot(previous_obs, sigma_p_d)) * ((previous_pow - np.dot(previous_obs, mu_p_d))**2 - np.dot(previous_obs, sigma_p_d)**2) * delta, previous_obs)
			
					mu_p_d += np.multiply(alpha * ( previous_pow - np.dot(previous_obs, mu_p_d)) * delta, previous_obs) # / sigma_p_d**2
					th_dash += np.multiply(alpha_p * delta, previous_obs) 
				elif previous_a == 1:
					previous_ang = previous_action[3]
					# sigma_a_t += np.multiply(alpha / (np.dot(previous_obs, sigma_a_t)) * ((previous_ang - np.dot(previous_af, mu_a_t))**2 - np.dot(previous_obs, sigma_a_t)**2) * delta, previous_obs)
			
					mu_a_t += np.multiply(alpha * ( previous_ang - np.dot(previous_obs, mu_a_t)) * delta, previous_obs) # / sigma_t**2

					th_turn += np.multiply(alpha_p * delta, previous_obs)
				else:
					previous_pow = previous_action[4]
					previous_ang = previous_action[5]
					# sigma_a_k += np.multiply(alpha / (np.dot(previous_obs, sigma_a_k)) * ((previous_ang - np.dot(previous_af, mu_a_k))**2 - np.dot(previous_obs, sigma_a_k)**2) * delta, previous_obs)
			
					mu_a_k += np.multiply(alpha * ( previous_ang - np.dot(previous_obs, mu_a_k)) * delta, previous_obs) # / sigma_a_k**2
					# sigma_p_k += np.multiply(alpha / (np.dot(previous_obs, sigma_p_k)) * ((previous_pow - np.dot(previous_obs, mu_p_k))**2 - np.dot(previous_obs, sigma_p_k)**2) * delta, previous_obs)
			
					mu_p_k += np.multiply(alpha * ( previous_pow - np.dot(previous_obs, mu_p_k)) * delta, previous_obs) # / sigma_p_k**2
					th_kick += np.multiply(alpha_p * delta, previous_obs) 

						
			flag = 0
			# print action
			previous_action = action # store action, obs, Q, r etc. as they will be used for update in the next step
			previous_obs = observation
			# previous_af = ang_feat
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
	# average_reward = evaluate_policy(env, th_dash, th_turn, th_kick, mu_a_d, sigma_d, mu_p_d, mu_a_t, sigma_t, mu_a_k, sigma_k, mu_p_k)
	# print average_reward
	return th_dash, th_turn, th_kick, mu_a_d, sigma_d, mu_p_d, mu_a_t, sigma_t, mu_a_k, sigma_k, mu_p_k

def extract_ang(obs): # extracts angular features as previously mentioned
	ang_features = []
	ang = np.arccos(obs[3]) * np.sign(obs[2]) / (math.pi)
	ang_features.append(ang) # self vel ang
	ang = np.arccos(obs[6]) * np.sign(obs[5]) / (math.pi)
	ang_features.append(ang) # self ang
	ang = np.arccos(obs[14]) * np.sign(obs[13]) / (math.pi)
	ang_features.append(ang) # goal center ang
	ang = np.arccos(obs[17]) * np.sign(obs[16]) / (math.pi) 
	ang_features.append(ang) # top goal post
	ang = np.arccos(obs[20]) * np.sign(obs[19]) / (math.pi)
	ang_features.append(ang) # bottom goal post
	ang = np.arccos(obs[23]) * np.sign(obs[22]) / (math.pi)
	ang_features.append(ang) # center of pen box line
	ang = np.arccos(obs[26]) * np.sign(obs[24]) / (math.pi)
	ang_features.append(ang) # top corner of pen box
	ang = np.arccos(obs[29]) * np.sign(obs[27]) / (math.pi)
	ang_features.append(ang) # bot corner of pen box
	ang = np.arccos(obs[32]) * np.sign(obs[30]) / (math.pi) 
	ang_features.append(ang) # left mid point
	ang = np.arccos(obs[35]) * np.sign(obs[34]) / (math.pi) 
	ang_features.append(ang) # top left corner
	ang = np.arccos(obs[38]) * np.sign(obs[37]) / (math.pi)
	ang_features.append(ang) # top right corner
	ang = np.arccos(obs[41]) * np.sign(obs[40]) / (math.pi)
	ang_features.append(ang) # bot right corner
	ang = np.arccos(obs[44]) * np.sign(obs[43]) / (math.pi) 
	ang_features.append(ang) # bot left corner
	ang = np.arccos(obs[52]) * np.sign(obs[51]) / (math.pi)
	ang_features.append(ang) # angle to ball
	ang = np.arccos(obs[57]) * np.sign(obs[56]) / (math.pi)
	ang_features.append(ang) # ball vel angle
	ang_features.append(1) # bias unit
	# deal with against keeper later.
	return ang_features

def evaluate_policy(env, th_dash, th_turn, th_kick, mu_a_d, sigma_d, mu_p_d, mu_a_t, sigma_t, mu_a_k, sigma_k, mu_p_k):
	th_dash, th_turn, th_kick, mu_a_d, sigma_d, mu_p_d, mu_a_t, sigma_t, mu_a_k, sigma_k, mu_p_k
	average_reward = 0
	# this part is just to evaluate how good a policy is after learning. there is no update here, other than that everything should be the same. 
	for i_episode in range(100):
		observation = env.reset()
		episodic_reward = 0
    		for t in range(500):
        		env.render() # I visualized to be able to diagnose errors
			# ang_feat = extract_ang(observation) # extract angles and append them to the original observation vector
			observation = np.append(observation, 1) 
			aug_s = np.transpose(np.matrix([observation, observation]))
			vector_dash = np.transpose(th_dash)
			vector_turn = np.transpose(th_turn)
			vector_kick = np.transpose(th_kick)

        		
			n1 = np.random.normal()
			dash_ang = np.dot(mu_a_d, observation) + n1 * sigma_d
			if dash_ang < -180:
				dash_ang = -180
				n1 = (-180 - np.dot(mu_a_d, observation)) / sigma_d
				# dash_angf = 0 # like relu (maybe leaky in the future)
			elif dash_ang > 180:
				dash_ang = 180
				n1 = (180 - np.dot(mu_a_d, observation)) / sigma_d
				# dash_angf = 0

			n2 = np.random.normal()
			dash_powa = np.dot(mu_p_d, observation) + n2 * sigma_d
			if dash_powa < -100:
				dash_powa = -100
				n2 = (-100 - np.dot(mu_p_d, observation)) / sigma_d
				# dash_powf = 0
			elif dash_powa > 100:
				dash_powa = 100
				n2 = (100 - np.dot(mu_p_d, observation)) / sigma_d
				# dash_powf = 0
 
			value_dash = np.dot( np.dot(vector_dash, aug_s) , [n1 / sigma_d, n2 / sigma_d])
			
			n3 = np.random.normal()
			turn_ang = np.dot(mu_a_t, observation) + n3 * sigma_t

			if turn_ang < -180:
				turn_ang = -180
				n3 = (-180 - np.dot(mu_a_t, observation)) / sigma_t
				# dash_angf = 0 # like relu (maybe leaky in the future)
			elif turn_ang > 180:
				turn_ang = 180
				n3 = (180 - np.dot(mu_a_t, observation)) / sigma_t
			value_turn = np.dot(vector_turn, np.transpose(np.matrix(observation))) * n3 / sigma_t

			n4 = np.random.normal()
			kick_ang = np.dot(mu_a_k, observation) + n4 * sigma_k

			if kick_ang < -180:
				kick_ang = -180
				n4 = (-180 - np.dot(mu_a_k, observation)) / sigma_k
				# dash_angf = 0 # like relu (maybe leaky in the future)
			elif kick_ang > 180:
				kick_ang = 180
				n4 = (180 - np.dot(mu_a_k, observation)) / sigma_k


			n5 = np.random.normal()
			kick_powa = np.dot(mu_p_k, observation) + n5 * sigma_k

			if kick_powa < 0:
				kick_powa = 0
				n5 = (0 - np.dot(mu_p_k, observation)) / sigma_k
				# dash_powf = 0
			elif kick_powa > 100:
				kick_powa = 100
				n5 = (100 - np.dot(mu_p_k, observation)) / sigma_k			

			value_kick = np.dot( np.dot(vector_kick, aug_s) , [n4 / sigma_k, n5 / sigma_k])
			if observation[12] > 0:
				best = max(value_dash, value_turn, value_kick)				

				if best == value_dash:
					a = 0
					action = [a, [dash_powa], [dash_ang], 0, 0, 0]
					Q_current = value_dash
					ns = np.transpose(np.matrix([n1 / sigma_d, n2 / sigma_d]))

				elif best == value_kick:
					a = 2
					action = [a, 0, 0, 0, [kick_powa], [kick_ang]]
					Q_current = value_kick
					ns = np.transpose(np.matrix([n4 / sigma_k, n5 / sigma_k]))
				
				else:
					a = 1
					action = [a, 0, 0, [turn_ang], 0, 0]
					Q_current = value_turn
					ns = np.matrix([n3/sigma_t])
			else:
				best = max(value_dash, value_turn)
				if best == value_dash:
					a = 0
					action = [a, [dash_powa], [dash_ang], 0, 0, 0]
					Q_current = value_dash
					ns = np.transpose(np.matrix([n1 / sigma_d, n2 / sigma_d]))
				else:
					a = 1
					action = [a, 0, 0, [turn_ang], 0, 0]
					Q_current = value_turn
					ns = np.matrix([n3/sigma_t])
			
        		
			# print action
        		observation, reward, done, info = env.step(action)
			episodic_reward += reward
			if done:
            			print("Episode finished after {} timesteps".format(t+1))
            			break
		average_reward += episodic_reward / 100
	programPause = raw_input("Press the <ENTER> key to continue...")
	return average_reward
	

th_dash, th_turn, th_kick, mu_a_d, sigma_d, mu_p_d, mu_a_t, sigma_t, mu_a_k, sigma_k, mu_p_k = q_gradient()


