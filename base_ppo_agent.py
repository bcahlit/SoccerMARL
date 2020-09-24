#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import grid_search

from soccer_env.high_action_soccer_env import HighActionSoccerEnv

stop = {
       "timesteps_total": 100000,
       "episode_reward_mean": 0.89
       }
results = tune.run(PPOTrainer, config={
    "env": HighActionSoccerEnv,
    "lr": 0.001,
    "num_workers": 1,
    "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
    "framework": 'torch'
}, stop=stop)  # "log_level": "INFO" for verbose,

