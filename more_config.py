import hfo_py
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
    "env_config": {
            "defense_npcs": 1,
           # " feature_set": hfo_py.LOW_LEVEL_FEATURE_SET ,
        },
    "lr": 0.001,
    "num_workers": 1,
    "lr": 1e-4,  # try different lrs
    "framework": 'torch'
}, stop=stop)  # "log_level": "INFO" for verbose,