# This code represents the Q-learning finite action spaces
# for the soccer environment

import numpy as np
from features_soccer import SoccerState
from features_soccer import SoccerAngleFeature

class SoccerActionType:
    DASH = 0
    TURN = 1
    KICK = 2

class SoccerActionParameter:
    POWER = 1
    ANGLE = 0

class SoccerDirection:
    LEFT = 0
    RIGHT = 1
    FORWARD = 2
    BACKWARD = 3
    GOAL_CENTER = 4
    BALL = 5
    GOALIE_LEFT = 6
    GOALIE_RIGHT = 7

# This represents an action space given a finite set of turn directions,
# dash powers, kick powers, and kick directions
class ActionSpaceSoccerSimple:
    
    def __init__(self, turn_dirs, dash_powers, kick_powers, kick_dirs):
        # Angles in [-180, 180]
        # Kick power in [0, 100]
        # Dash power in [-100, 100]

        self.turn_dirs = turn_dirs
        self.dash_powers = dash_powers
        self.kick_powers = kick_powers
        self.kick_dirs = kick_dirs

        # This comment is a bit outdated in that it assumes that there
        # is only one turn action.  But other than that, it gives the basic
        # idea for how actions are indexed
        # 0                               : Turn toward ball
        # 1...|dashPowers|                : Dash left
        # |dashPowers|+1...2|dashPowers|  : Dash right
        # 2|dashPowers|+1...3|dashPowers| : Dash forward
        # 3|dashPowers|+1...4|dashPowers| : Dash backward
        # 4|dashPowers|+1...4|dashPowers|+|kickPowers| : First kick direction
        # ...
        # 4|dashPowers|+|kickPowers|(|kickDirections| - 1) + 1...  : Last kick direction
  
    def valid_range(self, state):
        if SoccerState.is_kickable(state):
            return range(len(self.turn_dirs)+len(self.dash_powers)*4, self.size())
        else:
            return range(0, len(self.turn_dirs)+len(self.dash_powers)*4)

    def size(self, state = None):
        if state == None or SoccerState.is_kickable(state):
            return len(self.turn_dirs) + len(self.dash_powers)*4 + len(self.kick_powers)*len(self.kick_dirs)
        else:
            return len(self.turn_dirs) + len(self.dash_powers)*4

    def get_action_type(self, action_index):
        if action_index < len(self.turn_dirs):
            return SoccerActionType.TURN
        elif action_index < len(self.turn_dirs) + 4*len(self.dash_powers):
            return SoccerActionType.DASH
        else:
            return SoccerActionType.KICK  


    def get_action_param_values(self, state, action_index):
        action_type = self.get_action_type(action_index) 
        if action_type == SoccerActionType.TURN:
            turn_dir = self.turn_dirs[action_index]
            return [self.get_turn_angle(state, turn_dir)]
        elif action_type == SoccerActionType.DASH:
            dash_index = self.get_dash_index(action_index)
            dash_power = self.get_dash_power(dash_index)
            dash_angle = self.get_dash_angle(state, dash_index)
            return [dash_angle, dash_power]
        else:
            kick_index = self.get_kick_index(action_index)
            kick_power = self.get_kick_power(kick_index)
            kick_direction = self.get_kick_direction(kick_index)
            kick_angle = self.get_kick_angle(state, kick_direction)
            return [kick_angle, kick_power]


    def get_dash_index(self, action_index):
        return action_index - len(self.turn_dirs)


    def get_dash_power(self, dash_index):
        power_index = dash_index % len(self.dash_powers)
        return self.dash_powers[power_index]


    def get_dash_angle(self, state, dash_index):
        # Left, right, forward, backward
        dir_index = dash_index // len(self.dash_powers) 
        rel_dir = SoccerState.get_angle_degrees(state, SoccerAngleFeature.BALL)
        if dir_index == SoccerDirection.LEFT:
            return -90+rel_dir
        elif dir_index == SoccerDirection.RIGHT:
            return 90+rel_dir
        elif dir_index == SoccerDirection.FORWARD:
            return 0+rel_dir
        elif dir_index == SoccerDirection.BACKWARD:
            return 180+rel_dir
        else:
            return # Error

            
    def get_turn_angle(self, state, direction):
        if direction == SoccerDirection.BALL:
            return SoccerState.get_angle_degrees(state, SoccerAngleFeature.BALL)
        else:
            return # Error


    def get_kick_index(self, action_index):
        return action_index - len(self.turn_dirs) - 4*len(self.dash_powers)
    
    def get_kick_power(self, kick_index):
        power_index = kick_index % len(self.kick_powers)
        return self.kick_powers[power_index]

    def get_kick_direction(self, kick_index):
        dir_index = kick_index // len(self.kick_powers)
        return self.kick_dirs[dir_index]

    def get_kick_angle(self, state, direction):
        if direction == SoccerDirection.FORWARD:
            return 0
        elif direction == SoccerDirection.GOAL_CENTER:
            return SoccerState.get_angle_degrees(state, SoccerAngleFeature.GOAL_CENTER)
        elif direction == SoccerDirection.GOALIE_LEFT:
            return 0.8 * SoccerState.get_angle_degrees(state, SoccerAngleFeature.TOP_POST) + 0.2 * SoccerState.get_angle_degrees(state, SoccerAngleFeature.GOALIE)
        elif direction == SoccerDirection.GOALIE_RIGHT:
            return 0.8 * SoccerState.get_angle_degrees(state, SoccerAngleFeature.BOTTOM_POST) + 0.2 * SoccerState.get_angle_degrees(state, SoccerAngleFeature.GOALIE)
        else:
            return # Error

 
    def env_action(self, state, action_index):
        action_type = self.get_action_type(action_index)
        params = self.get_action_param_values(state, action_index)
        
        if action_type == SoccerActionType.DASH: 
            return [action_type, [params[SoccerActionParameter.POWER]],[params[SoccerActionParameter.ANGLE]],[0],[0],[0]]
        elif action_type == SoccerActionType.TURN:
            return [action_type, [0],[0],[params[SoccerActionParameter.ANGLE]],[0],[0]]
        else:
            return [action_type, [0],[0],[0],[params[SoccerActionParameter.POWER]],[params[SoccerActionParameter.ANGLE]]]

    def env_action_str(self, state, action_index):
        action_type = self.get_action_type(action_index)
        params = self.get_action_param_values(state, action_index)

        s = ""
        if action_type == SoccerActionType.DASH:
            s = "Dash(power=" + str(params[SoccerActionParameter.POWER]) + ", angle=" + str(params[SoccerActionParameter.ANGLE]) + ")"
        elif action_type == SoccerActionType.TURN:
            s = "Turn(theta=" + str(params[SoccerActionParameter.ANGLE]) + ")"
        else:
            s = "Kick(power=" + str(params[SoccerActionParameter.POWER]) + ", angle=" + str(params[SoccerActionParameter.ANGLE]) + ")"

        return s

