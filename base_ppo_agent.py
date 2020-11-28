#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import grid_search
import hfo_py

from soccer_env.high_action_soccer_env import HighActionSoccerEnv

def on_episode_end(info):
    episode = info["episode"]
    episode.custom_metrics["goal_rate"] = int(episode.last_info_for()['status'] == hfo_py.GOAL)

stop = {
       "timesteps_total": 100000,
       "episode_reward_mean": 0.89
       }
results = tune.run(PPOTrainer, config={
    "env": HighActionSoccerEnv,
    "lr": 0.001,
    "num_workers": 1,
    "env_config": {
        "server_config":{
            "defense_npcs": 1,
        },
        " feature_set": hfo_py.LOW_LEVEL_FEATURE_SET ,
    },
    # "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
    "callbacks": {
        "on_episode_end": on_episode_end,
    },
    "framework": 'torch'
}, stop=stop)  # "log_level": "INFO" for verbose,

