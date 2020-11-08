import hfo_py
from ray import tune
from ray.rllib.agents.marwil import MARWILTrainer
from agents.marwil_soccer import MARWILSTrainer, MARWILSModel
from ray.tune import grid_search

from soccer_env.low_action_score_goal import LowActionSoccerEnv

stop = {"timesteps_total": 2000000, "episode_reward_mean": 0.89}
results = tune.run(
    # MARWILTrainer,
    MARWILSTrainer,
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
            # "fcnet_activation": "swish"
            "custom_model": MARWILSModel,
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {},
        },
        "evaluation_num_workers": 1,
        "evaluation_interval": 1,
        "evaluation_config":{
            "input": "sampler"},
        "lr":0.005,
        "beta": 1.0,  # Compare to behavior cloning (beta=0.0).
        "num_gpus": 1,
        "framework": 'torch',
        "input": "filePath"
    },
    stop=stop)  # "log_level": "INFO" for verbose,
