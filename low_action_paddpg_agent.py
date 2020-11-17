import hfo_py
from ray import tune
from agents.paddpg import PADDPGTrainer
from ray.tune import grid_search

from soccer_env.low_action_score_goal import LowActionSoccerEnv

stop = {"timesteps_total": 2000000, "episode_reward_mean": 0.89}
results = tune.run(
    PADDPGTrainer,
    config={
        "env": LowActionSoccerEnv,
        "env_config": {
            "server_config": {
                "defense_npcs": 0,
            },
            "feature_set": hfo_py.LOW_LEVEL_FEATURE_SET,
        },
        "model":{
            "fcnet_hiddens": [2048,1024,512,256,128],
            "fcnet_activation": "swish"
        },
        "exploration_config": {
           "type": "EpsilonGreedy",
           "initial_epsilon": 1.0,
           "final_epsilon": 0.02,
           "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
        },
        "lr": 0.001,
        "num_workers": 1,
        "num_gpus": 0,
        "lr": 1e-4,  # try different lrs
        "framework": 'torch'
    },
    stop=stop)  # "log_level": "INFO" for verbose,
