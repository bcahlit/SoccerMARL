import math
import os  # 动作
import signal
import socket
import subprocess
import time
from contextlib import closing

import gym
import numpy as np
import ray
from gym import error, spaces, utils
from gym.utils import seeding

try:
    import hfo_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you can install HFO dependencies with 'pip install gym[soccer].')".format(e))

import logging

logger = logging.getLogger(__name__)

@ray.remote
class SingleAgentEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config, port):
        print("single agent",port)
        self.server_port = port
        self.hfo_path = hfo_py.get_hfo_path()
        self.env = hfo_py.HFOEnvironment()
        if  "feature_set" in config :
            self.env.connectToServer( feature_set=config['feature_set'], config_dir=hfo_py.get_config_path(), server_port=self.server_port)
        else :
            self.env.connectToServer( config_dir=hfo_py.get_config_path(), server_port=self.server_port)
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=((self.env.getStateSize(),)), dtype=np.float32)
        print("single agent init",self.observation_space)
        self.action_space = spaces.Discrete(14)

        self.status = hfo_py.IN_GAME
        self._seed = -1
        self.old_ball_prox = 0
        self.old_kickable = 0
        self.old_ball_dist_goal = 0
        self.got_kickable_reward = False
        self.first_step = True
        self.unum = self.env.getUnum()  # uniform number (identifier) of our lone agent

    def get_observation_space(self):
        return self.observation_space
    def get_action_space(self):
        return self.action_space
    # def __del__(self):
    #     self.env.act(hfo_py.QUIT)
    #     self.env.step()
    #     os.kill(self.server_process.pid, signal.SIGINT)
    #     if self.viewer is not None:
    #         os.kill(self.viewer.pid, signal.SIGKILL)
    
    @ray.method(num_returns=4)
    def step(self, action):
        self._take_action(action)
        self.status = self.env.step()
        reward = self._get_reward()
        ob = self.env.getState()
        episode_over = self.status != hfo_py.IN_GAME
        return ob, reward, episode_over, {'status': self.status}

    def _take_action(self, action):
        """ Converts the action space into an HFO action. """
        action_type = ACTION_LOOKUP[action]
        self.env.act(action_type)

    #  def _get_reward(self):
        #  """ Reward is given for scoring a goal. """
        #  if self.status == hfo_py.GOAL:
            #  return 1
        #  else:
            #  return 0

    def _get_reward(self):
        """
        Agent is rewarded for minimizing the distance between itself and
        the ball, minimizing the distance between the ball and the goal,
        and scoring a goal.
        """
        current_state = self.env.getState()
        #print("State =",current_state)
        #print("len State =",len(current_state))
        ball_proximity = current_state[53]
        goal_proximity = current_state[15]
        ball_dist = 1.0 - ball_proximity
        goal_dist = 1.0 - goal_proximity
        kickable = current_state[12]
        ball_ang_sin_rad = current_state[51]
        ball_ang_cos_rad = current_state[52]
        ball_ang_rad = math.acos(ball_ang_cos_rad)
        if ball_ang_sin_rad < 0:
            ball_ang_rad *= -1.
        goal_ang_sin_rad = current_state[13]
        goal_ang_cos_rad = current_state[14]
        goal_ang_rad = math.acos(goal_ang_cos_rad)
        if goal_ang_sin_rad < 0:
            goal_ang_rad *= -1.
        alpha = max(ball_ang_rad, goal_ang_rad) - min(ball_ang_rad,
                                                      goal_ang_rad)
        ball_dist_goal = math.sqrt(ball_dist * ball_dist +
                                   goal_dist * goal_dist - 2. * ball_dist *
                                   goal_dist * math.cos(alpha))
        # Compute the difference in ball proximity from the last step
        if not self.first_step:
            ball_prox_delta = ball_proximity - self.old_ball_prox
            kickable_delta = kickable - self.old_kickable
            ball_dist_goal_delta = ball_dist_goal - self.old_ball_dist_goal
        self.old_ball_prox = ball_proximity
        self.old_kickable = kickable
        self.old_ball_dist_goal = ball_dist_goal
        reward = 0
        if not self.first_step:
            mtb = self.__move_to_ball_reward(kickable_delta, ball_prox_delta)
            ktg = 3. * self.__kick_to_goal_reward(ball_dist_goal_delta)
            eot = self.__EOT_reward()
            reward = mtb + ktg + eot
            #print("mtb: %.06f ktg: %.06f eot: %.06f"%(mtb,ktg,eot))

        self.first_step = False
        #print("r =",reward)
        return reward

    def __move_to_ball_reward(self, kickable_delta, ball_prox_delta):
        reward = 0.
        if self.env.playerOnBall().unum < 0 or self.env.playerOnBall(
        ).unum == self.unum:
            reward += ball_prox_delta
        if kickable_delta >= 1 and not self.got_kickable_reward:
            reward += 1.
            self.got_kickable_reward = True
        return reward

    def __kick_to_goal_reward(self, ball_dist_goal_delta):
        if (self.env.playerOnBall().unum == self.unum):
            return -ball_dist_goal_delta
        elif self.got_kickable_reward == True:
            return 0.2 * -ball_dist_goal_delta
        return 0.

    def __EOT_reward(self):
        if self.status == hfo_py.GOAL:
            return 5.
        # elif self.status == hfo_py.CAPTURED_BY_DEFENSE:
        #     return -1.
        return 0.

    def reset(self):
        """ Repeats NO-OP action until a new episode begins. """
        while self.status == hfo_py.IN_GAME:
            self.env.act(hfo_py.NOOP)
            self.status = self.env.step()
        while self.status != hfo_py.IN_GAME:
            self.env.act(hfo_py.NOOP)
            self.status = self.env.step()
            # prevent infinite output when server dies
            if self.status == hfo_py.SERVER_DOWN:
                raise ServerDownException("HFO server down!")
        return self.env.getState()


class ServerDownException(Exception):
    """
    Custom error so agents can catch it and exit cleanly if the server dies unexpectedly.
    """
    pass
  

ACTION_LOOKUP = {
    #  0 : hfo_py.GO_TO_BALL,
    #  1 : hfo_py.MOVE,
    #  2 : hfo_py.SHOOT,
    #  3 : hfo_py.DRIBBLE, # Used on defense to slide tackle the ball
    #  4 : hfo_py.CATCH,  # Used only by goalie to catch the ball   
    0: hfo_py.NOOP,
    1: hfo_py.TOLEFT,
    2: hfo_py.TOPLEFT,
    3: hfo_py.TOP,
    4: hfo_py.TOPRIGHT,
    5: hfo_py.TORIGHT,
    6: hfo_py.BOTTOMRIGHT,
    7: hfo_py.BOTTOM,
    8: hfo_py.BOTTOMLEFT,

    9: hfo_py.AUTOPASS,
    10: hfo_py.AUTOTACKLE,

    11: hfo_py.GO_TO_BALL,
    12: hfo_py.MOVE,
    13: hfo_py.SHOOT,
    14: hfo_py.DRIBBLE
}

STATUS_LOOKUP = {
    hfo_py.IN_GAME: 'IN_GAME',
    hfo_py.SERVER_DOWN: 'SERVER_DOWN',
    hfo_py.GOAL: 'GOAL',
    hfo_py.OUT_OF_BOUNDS: 'OUT_OF_BOUNDS',
    hfo_py.OUT_OF_TIME: 'OUT_OF_TIME',
    hfo_py.CAPTURED_BY_DEFENSE: 'CAPTURED_BY_DEFENSE',
}
