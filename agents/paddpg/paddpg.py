import torch
from ray.rllib.agents.ddpg.ddpg import DDPGTrainer, \
    DEFAULT_CONFIG as DDPG_CONFIG
from ray.rllib.agents.ddpg.ddpg_tf_policy import postprocess_advantages
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing
ray.rllib.agents.ddpg.ddpg_torch_policy import DDPGTorchPolicy
from ray.rllib.utils.typing import TrainerConfigDict
from ray.rllib.utils.torch_ops import explained_variance

def paddpg_loss(policy, model, dist_class, train_batch):


PADDPGTorchPolicy = MARWILTorchPolicy.with_updates(
    loss_fn=paddpg_loss,
)

def get_policy_class(config):
    return PADDPGTorchPolicy

PADDPGTrainer = MARWILTrainer.with_updates(
    name="PADDPG",
    default_config=MARWIL_CONFIG,
    get_policy_class=get_policy_class,
    default_policy=PADDPGTorchPolicy,
)