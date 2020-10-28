import hfo_py
from ray import tune
from ray.rllib.agents.marwil import MARWILTrainer
from ray.tune import grid_search

from soccer_env.low_action_score_goal import LowActionSoccerEnv

stop = {"timesteps_total": 2000000, "episode_reward_mean": 0.89}
results = tune.run(
    MARWILTrainer,
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
        "evaluation_num_workers": 1,
        "evaluation_interval": 1,
        "evaluation_config":{
            "input": "sampler"},
        "beta": 1.0,  # Compare to behavior cloning (beta=0.0).
        "lr": 0.001,
        "num_workers": 1,
        "num_gpus": 1,
        "framework": 'torch',
        "input": "filePath"
    },
    stop=stop)  # "log_level": "INFO" for verbose,
