import numpy as np
from enum import Enum
import math

class SoccerFeatureIndex(Enum):
    VELOCITY_ANGLE_SINE = 2         # Sine of velocity angle
    VELOCITY_ANGLE_COSINE = 3       # Cosine of velocity angle
    VELOCITY_MAGNITUDE = 4          # Velocity magnitude
    BODY_ANGLE_SINE = 5             # Sine of agent body angle
    BODY_ANGLE_COSINE = 6           # Cosine of agent body angle
    COLLIDING_WITH_BALL = 9         # Indicator of collision with ball
    COLLIDING_WITH_GOAL_POST = 11   # Indicator of collision with goal post
    KICKABLE = 12                   # Indicator of whether ball is kickable
    GOAL_CENTER_SINE = 13           # Sine of angle to goal center
    GOAL_CENTER_COSINE = 14         # Cosine of angle to goal center
    GOAL_CENTER_DISTANCE = 15       # Distance to goal center
    BALL_ANGLE_SINE = 51            # Sine of angle between agent and ball
    BALL_ANGLE_COSINE = 52          # Cosine of angle between agent and ball
    BALL_DISTANCE = 53              # Distance to ball
    BALL_VELOCITY_MAGNITUDE = 55    # Ball velocity magnitude
    BALL_VELOCITY_ANGLE_SINE = 56   # Ball velocity angle sine
    BALL_VELOCITY_ANGLE_COSINE = 57 # Ball velocity angle cosine
    
class SoccerAngleFeature(Enum):
    AGENT_VELOCITY = 0
    AGENT_BODY = 1
    GOAL_CENTER = 2
    BALL = 3
    BALL_VELOCITY = 4

class SoccerState:
    def __init__(self):
        pass

    @staticmethod
    def get_feature(state, index):
        return state[index]

    @staticmethod
    def get_angle_radians(state, angle_feature):
        if angle_feature == SoccerAngleFeature.BALL:
            return SoccerState.get_angle_radians(state, SoccerFeatureIndex.BALL_ANGLE_COSINE, SoccerFeatureIndex.BALL_ANGLE_SINE)
        elif angle_feature == SoccerAngleFeature.GOAL_CENTER:
            return SoccerState.get_angle_radians(state, SoccerFeatureIndex.GOAL_CENTER_COSINE, SoccerFeatureIndex.GOAL_CENTER_SINE)
        elif angle_feature == SoccerAngleFeature.AGENT_VELOCITY:
            return SoccerState.get_angle_radians(state, SoccerFeatureIndex.VELOCITY_ANGLE_COSINE, SoccerFeatureIndex.VELOCITY_ANGLE_SINE)
        elif angle_feature == SoccerAngleFeature.AGENT_BODY:
            return SoccerState.get_angle_radians(state, SoccerFeatureIndex.BODY_ANGLE_COSINE, SoccerFeatureIndex.BODY_ANGLE_SINE)
        elif angle_feature == SoccerAngleFeature.BALL_VELOCITY:
            return SoccerState.get_angle_radians(state, SoccerFeatureIndex.BALL_VELOCITY_ANGLE_COSINE, SoccerFeatureIndex.BALL_VELOCITY_ANGLE_SINE)
        else:
            return # Error

    @staticmethod
    def get_angle_degrees(state, angle_feature):
        if angle_feature == SoccerAngleFeature.BALL:
            return SoccerState.get_angle_degrees(state, SoccerFeatureIndex.BALL_ANGLE_COSINE, SoccerFeatureIndex.BALL_ANGLE_SINE)
        elif angle_feature == SoccerAngleFeature.GOAL_CENTER:
            return SoccerState.get_angle_degrees(state, SoccerFeatureIndex.GOAL_CENTER_COSINE, SoccerFeatureIndex.GOAL_CENTER_SINE)
        elif angle_feature == SoccerAngleFeature.AGENT_VELOCITY:
            return SoccerState.get_angle_degrees(state, SoccerFeatureIndex.VELOCITY_ANGLE_COSINE, SoccerFeatureIndex.VELOCITY_ANGLE_SINE)
        elif angle_feature == SoccerAngleFeature.AGENT_BODY:
            return SoccerState.get_angle_degrees(state, SoccerFeatureIndex.BODY_ANGLE_COSINE, SoccerFeatureIndex.BODY_ANGLE_SINE)
        elif angle_feature == SoccerAngleFeature.BALL_VELOCITY:
            return SoccerState.get_angle_degrees(state, SoccerFeatureIndex.BALL_VELOCITY_ANGLE_COSINE, SoccerFeatureIndex.BALL_VELOCITY_ANGLE_SINE)
        else:
            return # Error

    @staticmethod
    def get_angle_radians(state, cos_index, sin_index):
        angle_cos = state[cos_index]
        angle_sin = state[sin_index]
        return np.arccos(angle_cos)*np.sign(angle_sin)

    @staticmethod
    def get_angle_degrees(state, cos_index, sin_index):
        return self.get_angle_radians(state, cos_index, sin_index)*360/(2*math.pi)


class FeaturesSoccerSmall:

    def __init__(self, actions):
        theta_v = lambda s : SoccerState.get_angle_degrees(s, SoccerAngleFeature.AGENT_VELOCITY) # Velocity angle
        m_v = lambda s : s[SoccerFeatureIndex.VELOCITY_MAGNITUDE]                                # Velocity magnitude
        theta_a = lambda s : SoccerState.get_angle_degrees(s, SoccerAngleFeature.AGENT_BODY)     # Agent Body angle
        collide_b = lambda s : s[SoccerFeatureIndex.COLLIDING_WITH_BALL]                         # Colliding with ball
        collide_g = lambda s : s[SoccerFeatureIndex.COLLIDING_WITH_GOAL_POST]                    # Colliding with goal post
        kickable = lambda s : s[SoccerFeatureIndex.KICKABLE]                                     # Kickable
        theta_gc = lambda s : SoccerState.get_angle_degrees(s, SoccerAngleFeature.GOAL_CENTER)   # Goal center angle
        d_gc = lambda s : s[SoccerFeatureIndex.GOAL_CENTER_DISTANCE]                             # Goal center distance
        theta_b = lambda s : SoccerState.get_angle_degrees(s, SoccerAngleFeature.BALL)           # Ball angle
        d_b = lambda s : s[SoccerFeatureIndex.BALL_DISTANCE]                                     # Ball distance
        m_bv = lambda s : s[SoccerFeatureIndex.BALL_VELOCITY_MAGNITUDE]                          # Ball velocity magnitude
        theta_bv = lambda s : SoccerState.get_angle_degrees(s, SoccerAngleFeature.BALL_VELOCITY) # Ball velocity angle
        bias = lambda s : 1        

        self.features = [theta_v, m_v, theta_a, collide_b, collide_g, kickable, theta_gc, d_gc, theta_b, d_b, m_bv, theta_bv, bias]
        self.feature_names = ["theta_v", "m_v", "theta_a", "collide_b", "collide_g", "kickable", "theta_gc", "d_gc", "theta_b", "d_b", "m_bv", "theta_bv", "bias"]
        self.actions = actions


    def size(self):
        return self.actions.size() * self.features.size()
 

    def compute(self, state, action):
        B = np.zeros(self.size())     

        for i in range(0, self.features.size()):
            B[action*self.features.size()+i] = self.features[i](state)        

        return B

    def state_feature_str(self, state):
        s = ""
        for i in range(0, self.features.size()):
            s = s + self.feature_names[i] + ": " + str(self.features[i](state)) + "\n"
        return s

