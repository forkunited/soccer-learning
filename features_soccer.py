# This file contains code for representing the
# soccer feature set for Q-learning

import numpy as np
import math

# Indices into the soccer environment observation
# vector
class SoccerFeatureIndex:
    VELOCITY_ANGLE_SINE = 2         # Sine of velocity angle
    VELOCITY_ANGLE_COSINE = 3       # Cosine of velocity angle
    VELOCITY_MAGNITUDE = 4          # Velocity magnitude
    BODY_ANGLE_SINE = 5             # Sine of agent body angle
    BODY_ANGLE_COSINE = 6           # Cosine of agent body angle
    STAMINA = 7                     # Agent stamina
    COLLIDING_WITH_BALL = 9         # Indicator of collision with ball
    COLLIDING_WITH_GOAL_POST = 11   # Indicator of collision with goal post
    KICKABLE = 12                   # Indicator of whether ball is kickable
    GOAL_CENTER_SINE = 13           # Sine of angle to goal center
    GOAL_CENTER_COSINE = 14         # Cosine of angle to goal center
    GOAL_CENTER_DISTANCE = 15       # Distance to goal center
    TOP_POST_SINE = 16              # Sine of angle to top goal post
    TOP_POST_COSINE = 17            # Cosine of angle to top goal post
    TOP_POST_DISTANCE = 18          # Distance to top goal post
    BOTTOM_POST_SINE = 19           # Sine of angle to bottom goal post
    BOTTOM_POST_COSINE = 20         # Cosine of angle to bottom goal post
    BOTTOM_POST_DISTANCE = 21       # Distance to bottom goal post
    BALL_ANGLE_SINE = 51            # Sine of angle between agent and ball
    BALL_ANGLE_COSINE = 52          # Cosine of angle between agent and ball
    BALL_DISTANCE = 53              # Distance to ball
    BALL_VELOCITY_MAGNITUDE = 55    # Ball velocity magnitude
    BALL_VELOCITY_ANGLE_SINE = 56   # Ball velocity angle sine
    BALL_VELOCITY_ANGLE_COSINE = 57 # Ball velocity angle cosine
    GOALIE_SINE = 58                # Sine of angle to goalie
    GOALIE_COSINE = 59              # Cosine of angle to goalie
    GOALIE_DISTANCE = 60            # Distance to goalie
    GOALIE_VELOCITY_MAGNITUDE = 63  # Goalie velocity magnitude
    GOALIE_VELOCITY_SINE = 64       # Sine of angle to goalie
    GOALIE_VELOCITY_COSINE = 65     # Cosine of angle to goalie

# Soccer angle feature types
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

# Helper static methods for pulling information from
# a soccer environment observation
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

# A soccer feature set for the Q-learning algorithm
class FeaturesSoccerSmall:
    # Initialize the feature set for a given set of actions.  Optionally include goalie and
    # interaction features
    def __init__(self, actions, interactions=True, goalie=False):
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
        stamina = lambda s : s[SoccerFeatureIndex.STAMINA]                                       # Stamina
        bias = lambda s : 1        

        theta_b_bv = lambda s : theta_b(s)*theta_bv(s)
        theta_gc_b = lambda s : theta_gc(s)*theta_b(s)

        base_features = [theta_v, m_v, theta_a, collide_b, collide_g, kickable, theta_gc, d_gc, theta_b, d_b, m_bv, theta_bv, stamina, bias]
        base_feature_names = ["theta_v", "m_v", "theta_a", "collide_b", "collide_g", "kickable", "theta_gc", "d_gc", "theta_b", "d_b", "m_bv", "theta_bv", "stamina", "bias"]

        goalie_features = []
        goalie_feature_names =[]
        if goalie:
            theta_tp = lambda s : SoccerState.get_angle_radians(s, SoccerAngleFeature.TOP_POST)         # Top post angle
            d_tp = lambda s : s[SoccerFeatureIndex.TOP_POST_DISTANCE]                                   # Top post distance
            theta_bp = lambda s : SoccerState.get_angle_radians(s, SoccerAngleFeature.BOTTOM_POST)      # Bottom post angle
            d_bp = lambda s : s[SoccerFeatureIndex.BOTTOM_POST_DISTANCE]                                # Bottom post distance
            theta_gk = lambda s : SoccerState.get_angle_radians(s, SoccerAngleFeature.GOALIE)           # Goalie angle
            d_gk = lambda s : s[SoccerFeatureIndex.GOALIE_DISTANCE]                                     # Goalie distance
            theta_gkv = lambda s : SoccerState.get_angle_radians(s, SoccerAngleFeature.GOALIE_VELOCITY) # Goalie velocity angle
            m_gkv = lambda s: s[SoccerFeatureIndex.GOALIE_VELOCITY_MAGNITUDE]                           # Goalie velocity magnitude
            
            goalie_features = [theta_tp, d_tp, theta_bp, d_bp, theta_gk, d_gk, theta_gkv, m_gkv]
            goalie_feature_names = ["theta_tp", "d_tp", "theta_bp", "d_bp", "theta_gk", "d_gk", "theta_gkv", "m_gkv"]
            
        interaction_features = []
        interaction_feature_names = []
        if interactions:
            theta_b_bv = lambda s : theta_b(s)*theta_bv(s)
            theta_gc_b = lambda s : theta_gc(s)*theta_b(s)        
 
            interaction_features = [theta_b_bv, theta_gc_b]
            interaction_feature_names = ["theta_b_bv", "theta_gc_b"]

        goalie_interaction_features = []
        goalie_interaction_feature_names = []
        if goalie and interactions:
            theta_tp_b = lambda s : theta_tp(s)*theta_b(s)
            theta_bp_b = lambda s : theta_bp(s)*theta_b(s)
            theta_gk_b = lambda s : theta_gk(s)*theta_b(s)
            theta_gk_gkv = lambda s : theta_gk(s)*theta_gkv(s)
            theta_bp_gk = lambda s : theta_bp(s)*theta_gk(s)
            theta_tp_gk = lambda s : theta_tp(s)*theta_gk(s)
            theta_gc_gk = lambda s : theta_gc(s)*theta_gk(s)

            goalie_interaction_features = [theta_tp_b, theta_bp_b, theta_gk_b, theta_gk_gkv, theta_bp_gk, theta_tp_gk, theta_gc_gk]
            goalie_interaction_feature_names = ["theta_tp_b", "theta_bp_b", "theta_gk_b", "theta_gk_gkv", "theta_bp_gk", "theta_tp_gk", "theta_gc_gk"]


        self.features = base_features + interaction_features + goalie_features + goalie_interaction_features
        self.feature_names = base_feature_names + interaction_feature_names + goalie_feature_names + goalie_interaction_feature_names

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

