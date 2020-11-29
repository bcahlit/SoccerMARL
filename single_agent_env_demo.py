import os, subprocess, time, signal#动作
import numpy as np
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

import socket
from contextlib import closing

try:
    import hfo_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you can install HFO dependencies with 'pip install gym[soccer].')".format(e))

import logging
logger = logging.getLogger(__name__)


class SingleAgentEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self, config, port):
        print("single agent",port)
        self.server_port = port
        self.hfo_path = hfo_py.get_hfo_path()
        self.env = hfo_py.HFOEnvironment()
        print("___28",port)
        if  "feature_set" in config :
            self.env.connectToServer( feature_set=config['feature_set'], config_dir=hfo_py.get_config_path(), server_port=self.server_port)
        else :
            print("___30",port)
            self.env.connectToServer( config_dir=hfo_py.get_config_path(), server_port=self.server_port)
            print("___32",port)
        print("___35",port)
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=((self.env.getStateSize(),)), dtype=np.float32)
        print("single agent init",self.observation_space)
        self.action_space = spaces.Discrete(4)

        self.status = hfo_py.IN_GAME
        self._seed = -1

    def __del__(self):
        self.env.act(hfo_py.QUIT)
        self.env.step()
        os.kill(self.server_process.pid, signal.SIGINT)
        if self.viewer is not None:
            os.kill(self.viewer.pid, signal.SIGKILL)


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

    def _get_reward(self):
        """ Reward is given for scoring a goal. """
        if self.status == hfo_py.GOAL:
            return 1
        else:
            return 0

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
    0 : hfo_py.GO_TO_BALL,
    1 : hfo_py.MOVE,
    2 : hfo_py.SHOOT,
    3 : hfo_py.DRIBBLE, # Used on defense to slide tackle the ball
    4 : hfo_py.CATCH,  # Used only by goalie to catch the ball
}

STATUS_LOOKUP = {
    hfo_py.IN_GAME: 'IN_GAME',
    hfo_py.SERVER_DOWN: 'SERVER_DOWN',
    hfo_py.GOAL: 'GOAL',
    hfo_py.OUT_OF_BOUNDS: 'OUT_OF_BOUNDS',
    hfo_py.OUT_OF_TIME: 'OUT_OF_TIME',
    hfo_py.CAPTURED_BY_DEFENSE: 'CAPTURED_BY_DEFENSE',
}