from ray.rllib.env.multi_agent_env import MultiAgentEnv
import os, subprocess, time, signal
import numpy as np
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from soccer_env.single_agent_env import SingleAgentEnv
import socket
from contextlib import closing
import ray
import psutil

import time

try:
    import hfo_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you can install HFO dependencies with 'pip install gym[soccer].')".format(e))

import logging
logger = logging.getLogger(__name__)

def find_free_port():
    """Find a random free port. Does not guarantee that the port will still be free after return.
    Note: HFO takes three consecutive port numbers, this only checks one.

    Source: https://github.com/crowdAI/marLo/blob/master/marlo/utils.py

    :rtype:  `int`
    """

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]
def make_multiagent(env_name_or_creator):

    class MultiEnv(MultiAgentEnv):
        def __init__(self, config):
            self.viewer = None
            self.server_process = None
            self.rcspid = None
            self.server_port = None
            self.hfo_path = hfo_py.get_hfo_path()
            #print(self.hfo_path)
            self._configure_environment(config)
            self.env = hfo_py.HFOEnvironment()
            self.one_hot_state_encoding = config.get("one_hot_state_encoding",
                                                     False)
            # num = config.pop("num_agents", 1)
            self.num = config["server_config"]["offense_agents"]
            
            self.agents = []
            for i in range(self.num):
                self.agents.append(env_name_or_creator.remote(config, self.server_port))
                time.sleep(2)
            self.dones = set()
            # self.observation_space = ray.get(self.agents[0].get_observation_space.remote())
            # self.action_space = ray.get(self.agents[0].get_action_space.remote())


        
        def _configure_environment(self, config):

            self._start_hfo_server(**config['server_config'])

        def _start_hfo_server(self, frames_per_trial=500,
                            #untouched_time=1000, 
                            untouched_time=100, 
                            offense_agents=1,
                            defense_agents=0, offense_npcs=0,
                            defense_npcs=0, sync_mode=True, port=None,
                            offense_on_ball=0, fullstate=True, seed=-1,
                            ball_x_min=0.0, ball_x_max=0.2,
                            verbose=False, log_game=False,
                            log_dir="log"):
        
            if port is None:
                port = find_free_port()
            self.server_port = port
            '''cmd = self.hfo_path + \
                    " --headless --frames-per-trial %i --untouched-time %i --offense-agents %i"\
                    " --defense-agents %i --offense-npcs %i --defense-npcs %i"\
                    " --port %i --offense-on-ball %i --seed %i --ball-x-min %f"\
                    " --ball-x-max %f --log-dir %s"\
                    % (frames_per_trial, untouched_time, 
                        offense_agents,
                        defense_agents, offense_npcs, defense_npcs, port,
                        offense_on_ball, seed, ball_x_min, ball_x_max,
                        log_dir)'''
            cmd = self.hfo_path + \
                    " --headless --frames-per-trial %i --offense-agents %i"\
                    " --defense-agents %i --offense-npcs %i --defense-npcs %i"\
                    " --port %i --offense-on-ball %i --seed %i --ball-x-min %f"\
                    " --ball-x-max %f --log-dir %s"\
                    % (frames_per_trial,
                        offense_agents,
                        defense_agents, offense_npcs, defense_npcs, port,
                        offense_on_ball, seed, ball_x_min, ball_x_max,
                        log_dir)
            if not sync_mode: cmd += " --no-sync"
            if fullstate:     cmd += " --fullstate"
            if verbose:       cmd += " --verbose"
            if not log_game:  cmd += " --no-logging"
            print('Starting server with command: %s' % cmd)
            self.server_process = subprocess.Popen(cmd.split(' '), shell=False)
            time.sleep(1) # Wait for server to startup before connecting a player
            print("server_process", psutil.Process(self.server_process.pid).children(recursive=True))
            self.rcspid = psutil.Process(self.server_process.pid).children(recursive=True)[0].pid
            print("rcssserver_process.pid", self.rcspid)
            time.sleep(2)

        def __del__(self):#note
            # not be used
            os.kill(self.server_process.pid, signal.SIGINT)
            # for i in range(num):
            #     self.agents[i].__del__.remote()

        def reset(self):
            self.dones = set()
            returned = {i: ray.get(stats_id) for i, stats_id in enumerate([a.reset.remote() for a in self.agents])}
            if self.one_hot_state_encoding:
                returned = {
                    0: {"obs": returned[0]},
                    1: {"obs": returned[1]}
                }
            return returned

        def step(self, action_dict):
            obs, rew, done, info = {}, {}, {}, {}
            setps = [self.agents[i].step.remote(action) for i, action in action_dict.items()]
            for i, _step in enumerate(setps):
                obs[i], rew[i], done[i], info[i] = ray.get(_step)
                if done[i]:
                    self.dones.add(i)
            done["__all__"] = len(self.dones) == len(self.agents)
            if self.one_hot_state_encoding:
                obs = {
                    0: {"obs": obs[0]},
                    1: {"obs": obs[1]}
                }
            return obs, rew, done, info

        def close(self):
            # if self.server_process is not None:
            #     try:
            os.kill(self.rcspid, signal.SIGKILL)
            os.kill(self.server_process.pid, signal.SIGKILL)
            
                # except Exception:
                #     pass

        def _start_viewer(self):
        
            cmd = hfo_py.get_viewer_path() +\
                    " --connect --port %d" % (self.server_port)
            self.viewer = subprocess.Popen(cmd.split(' '), shell=False)

        def _render(self, mode='human', close=False):
        
            if close:
                if self.viewer is not None:
                    os.kill(self.viewer.pid, signal.SIGKILL)
            else:
                if self.viewer is None:
                    self._start_viewer()
    return MultiEnv

MultiAgentSoccer = make_multiagent(SingleAgentEnv)
