#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import grid_search, register_env
from gym.spaces import Tuple, MultiDiscrete, Dict, Discrete
from gym import spaces
import torch
import numpy as np
import hfo_py

from soccer_env.mult_agent_env import MultiAgentSoccer

env_config = {
    "one_hot_state_encoding": True,
    "server_config": {
        "defense_npcs": 0,
        "offense_agents": 2
    },
    " feature_set": hfo_py.LOW_LEVEL_FEATURE_SET,
}


#  def on_episode_end(info):
    #  episode = info["episode"]
    #  episode.custom_metrics["goal_rate"] = int(
        #  episode.last_info_for(0)['status'] == hfo_py.GOAL)


server_config = env_config["server_config"]
obs_space_size = 59 + 9 * (server_config["defense_npcs"] +
                           server_config["offense_agents"] - 1)
obs_space = spaces.Box(low=-1,
                       high=1,
                       shape=((obs_space_size, )),
                       dtype=np.float32)
act_space = spaces.Discrete(14)

# 必须要添加智能体组 先用两个
grouping = {
    "group_1": [0, 1],
}
obs_space = Tuple([
    Dict({
        "obs": obs_space
    }),
    Dict({
        "obs": obs_space,
    }),
])
act_space = Tuple([
    act_space,
    act_space,
])

register_env(
    "grouped_soccer", lambda config: MultiAgentSoccer(config).
    with_agent_groups(grouping, obs_space=obs_space, act_space=act_space))

config = {
    "env": "grouped_soccer",
    # 我不知道这两个参数的作用
    #  "rollout_fragment_length": 4,
    #  "train_batch_size": 32,
    "exploration_config": {
        "epsilon_timesteps": 5000,
        "final_epsilon": 0.05,
    },
    "num_workers": 1,
    #  "mixer": grid_search([None, "qmix", "vdn"]),
    "mixer": "qmix",
    "env_config": env_config,
    #  "callbacks": {
        #  "on_episode_end": on_episode_end,
    #  },
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus": 1 if torch.cuda.is_available() else 0,
    "framework": "torch",
}
group = True

#  stop = {
#  "episode_reward_mean": args.stop_reward,
#  "timesteps_total": args.stop_timesteps,
#  }

stop = {"timesteps_total": 100000, "episode_reward_mean": 0.89}

results = tune.run("QMIX", stop=stop, config=config, verbose=1)

#  if args.as_test:
#  check_learning_achieved(results, args.stop_reward)
