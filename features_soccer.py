import numpy as np
import math

class SoccerFeatureIndex:
    VELOCITY_ANGLE_SINE = 2         # Sine of velocity angle
    VELOCITY_ANGLE_COSINE = 3       # Cosine of velocity angle
    VELOCITY_MAGNITUDE = 4          # Velocity magnitude
    BODY_ANGLE_SINE = 5             # Sine of agent body angle
    BODY_ANGLE_COSINE = 6           # Cosine of agent body angle
    STAMINA = 7
    COLLIDING_WITH_BALL = 9         # Indicator of collision with ball
    COLLIDING_WITH_GOAL_POST = 11   # Indicator of collision with goal post
    KICKABLE = 12                   # Indicator of whether ball is kickable
    GOAL_CENTER_SINE = 13           # Sine of angle to goal center
    GOAL_CENTER_COSINE = 14         # Cosine of angle to goal center
    GOAL_CENTER_DISTANCE = 15       # Distance to goal center
    TOP_POST_SINE = 16
    TOP_POST_COSINE = 17
    TOP_POST_DISTANCE = 18
    BOTTOM_POST_SINE = 19
    BOTTOM_POST_COSINE = 20
    BOTTOM_POST_DISTANCE = 21
    BALL_ANGLE_SINE = 51            # Sine of angle between agent and ball
    BALL_ANGLE_COSINE = 52          # Cosine of angle between agent and ball
    BALL_DISTANCE = 53              # Distance to ball
    BALL_VELOCITY_MAGNITUDE = 55    # Ball velocity magnitude
    BALL_VELOCITY_ANGLE_SINE = 56   # Ball velocity angle sine
    BALL_VELOCITY_ANGLE_COSINE = 57 # Ball velocity angle cosine
    GOALIE_SINE = 58		
    GOALIE_COSINE = 59
    GOALIE_DISTANCE = 60
    GOALIE_VELOCITY_MAGNITUDE = 63
    GOALIE_VELOCITY_SINE = 64
    GOALIE_VELOCITY_COSINE = 65

    
class SoccerAngleFeature:
    AGENT_VELOCITY = 0
    AGENT_BODY = 1
    GOAL_CENTER = 2
    BALL = 3
    BALL_VELOCITY = 4
    TOP_POST = 5
    BOTTOM_POST = 6
    GOALIE = 7
    GOALIE_VELOCITY = 8

class SoccerState:
    def __init__(self):
        pass

    @staticmethod
    def get_feature(state, index):
        return state[index]

    @staticmethod
    def is_kickable(state):
        return state[SoccerFeatureIndex.KICKABLE] != -1

    @staticmethod
    def get_angle_radians(state, angle_feature):
        if angle_feature == SoccerAngleFeature.BALL:
            return SoccerState.get_angle_radians_helper(state, SoccerFeatureIndex.BALL_ANGLE_COSINE, SoccerFeatureIndex.BALL_ANGLE_SINE)
        elif angle_feature == SoccerAngleFeature.GOAL_CENTER:
            return SoccerState.get_angle_radians_helper(state, SoccerFeatureIndex.GOAL_CENTER_COSINE, SoccerFeatureIndex.GOAL_CENTER_SINE)
        elif angle_feature == SoccerAngleFeature.AGENT_VELOCITY:
            return SoccerState.get_angle_radians_helper(state, SoccerFeatureIndex.VELOCITY_ANGLE_COSINE, SoccerFeatureIndex.VELOCITY_ANGLE_SINE)
        elif angle_feature == SoccerAngleFeature.AGENT_BODY:
            return SoccerState.get_angle_radians_helper(state, SoccerFeatureIndex.BODY_ANGLE_COSINE, SoccerFeatureIndex.BODY_ANGLE_SINE)
        elif angle_feature == SoccerAngleFeature.BALL_VELOCITY:
            return SoccerState.get_angle_radians_helper(state, SoccerFeatureIndex.BALL_VELOCITY_ANGLE_COSINE, SoccerFeatureIndex.BALL_VELOCITY_ANGLE_SINE)
	elif angle_feature == SoccerAngleFeature.TOP_POST:
	    return SoccerState.get_angle_radians_helper(state, SoccerFeatureIndex.TOP_POST_COSINE, SoccerFeatureIndex.TOP_POST_SINE)
	elif angle_feature == SoccerAngleFeature.BOTTOM_POST:
	    return SoccerState.get_angle_radians_helper(state, SoccerFeatureIndex.BOTTOM_POST_COSINE, SoccerFeatureIndex.BOTTOM_POST_SINE)
	elif angle_feature == SoccerAngleFeature.GOALIE:
	    return SoccerState.get_angle_radians_helper(state, SoccerFeatureIndex.GOALIE_COSINE, SoccerFeatureIndex.GOALIE_SINE)
	elif angle_feature == SoccerAngleFeature.GOALIE_VELOCITY:
	    return SoccerState.get_angle_radians_helper(state, SoccerFeatureIndex.GOALIE_VELOCITY_COSINE, SoccerFeatureIndex.GOALIE_VELOCITY_SINE)
        else:
            return # Error

    @staticmethod
    def get_angle_degrees(state, angle_feature):
        if angle_feature == SoccerAngleFeature.BALL:
            return SoccerState.get_angle_degrees_helper(state, SoccerFeatureIndex.BALL_ANGLE_COSINE, SoccerFeatureIndex.BALL_ANGLE_SINE)
        elif angle_feature == SoccerAngleFeature.GOAL_CENTER:
            return SoccerState.get_angle_degrees_helper(state, SoccerFeatureIndex.GOAL_CENTER_COSINE, SoccerFeatureIndex.GOAL_CENTER_SINE)
        elif angle_feature == SoccerAngleFeature.AGENT_VELOCITY:
            return SoccerState.get_angle_degrees_helper(state, SoccerFeatureIndex.VELOCITY_ANGLE_COSINE, SoccerFeatureIndex.VELOCITY_ANGLE_SINE)
        elif angle_feature == SoccerAngleFeature.AGENT_BODY:
            return SoccerState.get_angle_degrees_helper(state, SoccerFeatureIndex.BODY_ANGLE_COSINE, SoccerFeatureIndex.BODY_ANGLE_SINE)
        elif angle_feature == SoccerAngleFeature.BALL_VELOCITY:
            return SoccerState.get_angle_degrees_helper(state, SoccerFeatureIndex.BALL_VELOCITY_ANGLE_COSINE, SoccerFeatureIndex.BALL_VELOCITY_ANGLE_SINE)
	elif angle_feature == SoccerAngleFeature.TOP_POST:
	    return SoccerState.get_angle_degrees_helper(state, SoccerFeatureIndex.TOP_POST_COSINE, SoccerFeatureIndex.TOP_POST_SINE)
	elif angle_feature == SoccerAngleFeature.BOTTOM_POST:
	    return SoccerState.get_angle_degrees_helper(state, SoccerFeatureIndex.BOTTOM_POST_COSINE, SoccerFeatureIndex.BOTTOM_POST_SINE)
	elif angle_feature == SoccerAngleFeature.GOALIE:
	    return SoccerState.get_angle_degrees_helper(state, SoccerFeatureIndex.GOALIE_COSINE, SoccerFeatureIndex.GOALIE_SINE)
	elif angle_feature == SoccerAngleFeature.GOALIE_VELOCITY:
	    return SoccerState.get_angle_degrees_helper(state, SoccerFeatureIndex.GOALIE_VELOCITY_COSINE, SoccerFeatureIndex.GOALIE_VELOCITY_SINE)
        else:
            return # Error

    @staticmethod
    def get_angle_radians_helper(state, cos_index, sin_index):
        angle_cos = state[cos_index]
        angle_sin = state[sin_index]
        return np.arccos(angle_cos)*np.sign(angle_sin)

    @staticmethod
    def get_angle_degrees_helper(state, cos_index, sin_index):
        return SoccerState.get_angle_radians_helper(state, cos_index, sin_index)*360/(2*math.pi)


class FeaturesSoccerSmall:

    def __init__(self, env_name, actions):
        theta_v = lambda s : SoccerState.get_angle_radians(s, SoccerAngleFeature.AGENT_VELOCITY) # Velocity angle
        m_v = lambda s : s[SoccerFeatureIndex.VELOCITY_MAGNITUDE]                                # Velocity magnitude
        theta_a = lambda s : SoccerState.get_angle_radians(s, SoccerAngleFeature.AGENT_BODY)     # Agent Body angle
        collide_b = lambda s : s[SoccerFeatureIndex.COLLIDING_WITH_BALL]                         # Colliding with ball
        collide_g = lambda s : s[SoccerFeatureIndex.COLLIDING_WITH_GOAL_POST]                    # Colliding with goal post
        kickable = lambda s : s[SoccerFeatureIndex.KICKABLE]                                     # Kickable
        theta_gc = lambda s : SoccerState.get_angle_radians(s, SoccerAngleFeature.GOAL_CENTER)   # Goal center angle
        d_gc = lambda s : s[SoccerFeatureIndex.GOAL_CENTER_DISTANCE]                             # Goal center distance
        theta_b = lambda s : SoccerState.get_angle_radians(s, SoccerAngleFeature.BALL)           # Ball angle
        d_b = lambda s : s[SoccerFeatureIndex.BALL_DISTANCE]                                     # Ball distance
        m_bv = lambda s : s[SoccerFeatureIndex.BALL_VELOCITY_MAGNITUDE]                          # Ball velocity magnitude
        theta_bv = lambda s : SoccerState.get_angle_radians(s, SoccerAngleFeature.BALL_VELOCITY) # Ball velocity angle
	stamina = lambda s : s[SoccerFeatureIndex.STAMINA]
	bias = lambda s : 1 
	if env_name == 'AgainstKeeper':
		theta_tp = lambda s : SoccerState.get_angle_radians(s, SoccerAngleFeature.TOP_POST)
		d_tp = lambda s : s[SoccerFeatureIndex.TOP_POST_DISTANCE]
		theta_bp = lambda s : SoccerState.get_angle_radians(s, SoccerAngleFeature.BOTTOM_POST)
		d_bp = lambda s : s[SoccerFeatureIndex.BOTTOM_POST_DISTANCE]
		theta_gk = lambda s : SoccerState.get_angle_radians(s, SoccerAngleFeature.GOALIE)
		d_gk = lambda s : s[SoccerFeatureIndex.GOALIE_DISTANCE]
		theta_gkv = lambda s : SoccerState.get_angle_radians(s, SoccerAngleFeature.GOALIE_VELOCITY)
		m_gkv = lambda s: s[SoccerFeatureIndex.GOALIE_VELOCITY_MAGNITUDE]
	# maybe time since last kicked ball can be added
               

        theta_v2 = lambda s : theta_v(s)*theta_v(s)
        theta_a2 = lambda s : theta_a(s)*theta_a(s)
        theta_gc2 = lambda s : theta_gc(s)*theta_gc(s)
        theta_b2 = lambda s : theta_b(s)*theta_b(s)
        theta_bv2 = lambda s : theta_bv(s)*theta_bv(s)

        theta_b_bv = lambda s : theta_b(s)*theta_bv(s)
        theta_gc_b = lambda s : theta_gc(s)*theta_b(s)
	if env_name == 'AgainstKeeper':
		theta_tp_b = lambda s : theta_tp(s)*theta_b(s)
		theta_bp_b = lambda s : theta_tp(s)*theta_b(s)
		theta_gk_b = lambda s : theta_gk(s)*theta_b(s)
		theta_gk_gkv = lambda s : theta_gk(s)*theta_gkv(s)
		theta_tp_gc = lambda s : theta_tp(s)*theta_gc(s)
		theta_tp_bp = lambda s : theta_tp(s)*theta_bp(s)
		theta_tp_gk = lambda s : theta_tp(s)*theta_gk(s)
		theta_gc_gk = lambda s : theta_gc(s)*theta_gk(s)
		theta_gc_bp = lambda s : theta_gc(s)*theta_bp(s)
		theta_bp_gk = lambda s : theta_bp(s)*theta_gk(s)

	if env_name == 'EmptyGoal':
	
        	base_features = [theta_v, m_v, theta_a, collide_b, collide_g, kickable, theta_gc, d_gc, theta_b, d_b, m_bv, theta_bv, stamina, bias]
        	base_feature_names = ["theta_v", "m_v", "theta_a", "collide_b", "collide_g", "kickable", "theta_gc", "d_gc", "theta_b", "d_b", "m_bv", "theta_bv", "stamina", "bias"]
		interaction_features = [theta_b_bv, theta_gc_b]
        	interaction_feature_names = ["theta_b_bv", "theta_gc_b"]
	else:

		base_features = [theta_v, m_v, theta_a, collide_b, collide_g, kickable, theta_gc, d_gc, theta_tp, d_tp, theta_bp, d_bp, theta_b, d_b, m_bv, theta_bv, theta_gk, d_gk, m_gkv, theta_gkv, stamina, bias]
        	base_feature_names = ["theta_v", "m_v", "theta_a", "collide_b", "collide_g", "kickable", "theta_gc", "d_gc", "theta_tp", "d_tp", "theta_bp", "d_bp", "theta_b", "d_b", "m_bv", "theta_bv", "theta_gk", "d_gk", "m_gkv", "theta_gkv", "stamina", "bias"]
		interaction_features = [theta_b_bv, theta_gc_b, theta_tp_b, theta_bp_b, theta_gk_b, theta_gk_gkv, theta_tp_gc, theta_tp_bp, theta_tp_gk, theta_gc_gk, theta_gc_bp, theta_bp_gk]
        	interaction_feature_names = ["theta_b_bv", "theta_gc_b", "theta_tp_b", "theta_bp_b", "theta_gk_b", "theta_gk_gkv", "theta_tp_gc", "theta_tp_bp", "theta_tp_gk", "theta_gc_gk", "theta_gc_bp", "theta_bp_gk"]
        
        #squared_features = [theta_v2, theta_a2, theta_gc2, theta_b2, theta_bv2]
        #squared_feature_names = ["theta_v2", "theta_a2", "theta_gc2", "theta_b2", "theta_bv2"]

        

	

	
        self.features = base_features + interaction_features #+ squared_features
        self.feature_names = base_feature_names + interaction_feature_names #+ squared_feature_names

        self.actions = actions


    def size(self):
        return self.actions.size() * len(self.features)
 

    def compute(self, state, action):
        B = np.zeros(self.size())     

        for i in range(0, len(self.features)):
            B[action*len(self.features)+i] = self.features[i](state)        

        return B

    def state_feature_str(self, state):
        s = ""
        for i in range(0, len(self.features)):
            s = s + self.feature_names[i] + ": " + str(self.features[i](state)) + "\n"
        return s

