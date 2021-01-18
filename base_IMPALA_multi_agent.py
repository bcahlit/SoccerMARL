#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ray import tune
from ray.rllib.agents.impala import ImpalaTrainer
from ray.tune import grid_search
from gym import spaces
import numpy as np
import hfo_py
import torch

from soccer_env.mult_agent_env import MultiAgentSoccer

env_config = {
    "server_config": {
        "defense_npcs": 1,
        "offense_agents": 2
    },
    " feature_set": hfo_py.LOW_LEVEL_FEATURE_SET,
}


def on_episode_end(info):
    episode = info["episode"]
    # print("episode.last_info_for()", episode.last_info_for(0))
    episode.custom_metrics["goal_rate"] = int(
        episode.last_info_for(0)['status'] == hfo_py.GOAL)


server_config = env_config["server_config"]
obs_space_size = 59 + 9 * (
    server_config["defense_npcs"] + server_config["offense_agents"] - 1)
obs_space = spaces.Box(
    low=-1, high=1, shape=((obs_space_size, )), dtype=np.float32)
act_space = spaces.Discrete(14)


def gen_policy(_):
    return (None, obs_space, act_space, {})


# Setup PPO with an ensemble of `num_policies` different policies
policies = {
    'policy_{}'.format(i): gen_policy(i)
    for i in range(server_config["offense_agents"])
}
policy_ids = list(policies.keys())

stop = {"timesteps_total": 20000000, "episode_reward_mean": 10}
results = tune.run(
    ImpalaTrainer,
    config={
        "env": MultiAgentSoccer,
        "env_config": env_config,
        'multiagent': {
            'policies': policies,
            'policy_mapping_fn': lambda agent_id: policy_ids[agent_id],
        },
        "model": {
            "fcnet_hiddens":
            grid_search([[256, 256], [512, 512], [512, 256, 128],
                         [1024, 512, 256], [2048, 1024, 512, 256, 128]]),
            "fcnet_activation":
            grid_search(["swish", "relu", "tanh"]),
            "use_lstm":
            grid_search([False, True])
        },
        "callbacks": {
            "on_episode_end": on_episode_end,
        },
        "lr": grid_search([0.0003,0.0005,0.0007,0.0001]),
        "num_gpus": 1 if torch.cuda.is_available() else 0,
        "num_workers": 5,
        "framework": 'torch'
    },
    stop=stop)
