#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import grid_search
from gym import spaces
import numpy as np
import hfo_py

from mult_agent_env_demo import MultiAgentSoccer

obs_space = spaces.Box(low=-1, high=1,
                                            shape=((68,)), dtype=np.float32)
act_space = spaces.Discrete(4)

def gen_policy(_):
    return (None, obs_space, act_space, {})

# Setup PPO with an ensemble of `num_policies` different policies
policies = {
    'policy_{}'.format(i): gen_policy(i) for i in range(2)
}
policy_ids = list(policies.keys())

stop = {
       "timesteps_total": 100000,
       "episode_reward_mean": 0.89
       }
results = tune.run(PPOTrainer, config={
    "env": MultiAgentSoccer,
    "env_config": {
        "server_config":{
            "defense_npcs": 0,
            "offense_agents":2 
        },
        " feature_set": hfo_py.LOW_LEVEL_FEATURE_SET ,
    },
    'multiagent': {
        'policies': policies,
        'policy_mapping_fn': tune.function(
            lambda agent_id: policy_ids[int(agent_id[6:])]),
    },
    "lr": 0.001,
    "num_gpus" : 0.8,
    "num_workers": 1,
    "lr": 1e-4,
    "framework": 'torch'
}, stop=stop)  
