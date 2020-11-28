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


# def action_sampler_fn(policy, model, input_dict, state, explore, timestep):
#     """Action sampler function has two phases. During the prefill phase,
#     actions are sampled uniformly [-1, 1]. During training phase, actions
#     are evaluated through DreamerPolicy and an additive gaussian is added
#     to incentivize exploration.
#     """
#     obs = input_dict["obs"]

#     # Custom Exploration
#     if timestep <= policy.config["prefill_timesteps"]:
#         logp = [0.0]
#         # Random action in space [-1.0, 1.0]
#         action = 2.0 * torch.rand(1, model.action_space.shape[0]) - 1.0
#         state = model.get_initial_state()
#     else:
#         # Weird RLLib Handling, this happens when env rests
#         if len(state[0].size()) == 3:
#             # Very hacky, but works on all envs
#             state = model.get_initial_state()
#         action, logp, state = model.policy(obs, state, explore)

#     policy.global_timestep += policy.config["action_repeat"]

#     return action, logp, state

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
    dist_class, logit_dim = ModelCatalog.get_action_dist(
                    model.action_space, policy.config["model"], framework="torch")
    return dist_inputs, dist_class, []  # []=state out

PADDPGTorchPolicy = DDPGTorchPolicy.with_updates(
    # loss_fn=paddpg_loss,
    # action_sampler_fn=action_sampler_fn,
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