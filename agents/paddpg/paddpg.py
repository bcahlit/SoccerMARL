import torch
import numpy as np
from ray.rllib.agents.ddpg.ddpg import DDPGTrainer, \
    DEFAULT_CONFIG as DDPG_CONFIG
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.agents.ddpg.ddpg_torch_policy import DDPGTorchPolicy
from ray.rllib.agents.ddpg.noop_model import TorchNoopModel
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.typing import TrainerConfigDict
from ray.rllib.models.torch.torch_action_dist import TorchMultiActionDistribution
from ray.rllib.utils.torch_ops import explained_variance
from agents.paddpg.paddpg_model import PADDPGTorchModel

def build_paddpg_models(policy, observation_space, action_space, config):
    if policy.config["use_state_preprocessor"]:
        default_model = None  # catalog decides
        num_outputs = 256  # arbitrary
        config["model"]["no_final_linear"] = True
    else:
        default_model = TorchNoopModel if config["framework"] == "torch" \
            else NoopModel
        num_outputs = int(np.product(observation_space.shape))

    policy.model = ModelCatalog.get_model_v2(
        obs_space=observation_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        model_interface=(PADDPGTorchModel
                         if config["framework"] == "torch" else DDPGTFModel),
        default_model=default_model,
        name="ddpg_model",
        actor_hidden_activation=config["actor_hidden_activation"],
        actor_hiddens=config["actor_hiddens"],
        critic_hidden_activation=config["critic_hidden_activation"],
        critic_hiddens=config["critic_hiddens"],
        twin_q=config["twin_q"],
        add_layer_norm=(policy.config["exploration_config"].get("type") ==
                        "ParameterNoise"),
    )

    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=observation_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        model_interface=(PADDPGTorchModel
                         if config["framework"] == "torch" else DDPGTFModel),
        default_model=default_model,
        name="target_ddpg_model",
        actor_hidden_activation=config["actor_hidden_activation"],
        actor_hiddens=config["actor_hiddens"],
        critic_hidden_activation=config["critic_hidden_activation"],
        critic_hiddens=config["critic_hiddens"],
        twin_q=config["twin_q"],
        add_layer_norm=(policy.config["exploration_config"].get("type") ==
                        "ParameterNoise"),
    )

    return policy.model

def get_distribution_inputs_and_class(policy,
                                      model,
                                      obs_batch,
                                      *,
                                      explore=True,
                                      is_training=False,
                                      **kwargs):
    model_out, _ = model({
        "obs": obs_batch,
        "is_training": is_training,
    }, [], None)
    dist_inputs = model.get_policy_output(model_out)
    return dist_inputs, TorchMultiActionDistribution, []  # []=state out

PADDPGTorchPolicy = DDPGTorchPolicy.with_updates(
    # loss_fn=paddpg_loss,
    validate_spaces=None,
    action_distribution_fn=get_distribution_inputs_and_class,
    make_model_and_action_dist=None,
    make_model=build_paddpg_models,
)

def get_policy_class(config):
    return PADDPGTorchPolicy

PADDPGTrainer = DDPGTrainer.with_updates(
    name="PADDPG",
    default_config=DDPG_CONFIG,
    get_policy_class=get_policy_class,
    default_policy=PADDPGTorchPolicy,
)