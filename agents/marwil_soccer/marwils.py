import torch
from ray.rllib.agents.marwil.marwil import MARWILTrainer, \
    DEFAULT_CONFIG as MARWIL_CONFIG
from ray.rllib.agents.marwil.marwil_tf_policy import postprocess_advantages
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.agents.marwil.marwil_torch_policy import \
            MARWILTorchPolicy
from ray.rllib.utils.typing import TrainerConfigDict
from ray.rllib.utils.torch_ops import explained_variance

def marwil_loss(policy, model, dist_class, train_batch):
    model_out, _ = model.from_batch(train_batch)
    print("model_out.shape",model_out.shape)
    state_values = model.value_function()
    advantages = train_batch[Postprocessing.ADVANTAGES]
    actions = train_batch[SampleBatch.ACTIONS]
    print("actions", actions[0:2])
    for i in range(0, model_out.shape[0]):
        if actions[i][0] == 0:
            model_out[i][-6:]=0
        elif actions[i][0] == 1:
            model_out[i][-4:]=0
            model_out[i][3:7]=0
        elif actions[i][0] == 2:
            model_out[i][3:-4]=0
        else:
            pass
    print("model_out", model_out[0:2])
    action_dist = dist_class(model_out, model)
    print("action_dist", action_dist.input_lens)

    # Value loss.
    policy.v_loss = 0.5 * torch.mean(torch.pow(state_values - advantages, 2.0))

    # Policy loss.
    # Advantage estimation.
    adv = advantages - state_values
    # Update averaged advantage norm.
    policy.ma_adv_norm.add_(
        1e-6 * (torch.mean(torch.pow(adv, 2.0)) - policy.ma_adv_norm))
    # #xponentially weighted advantages.
    exp_advs = torch.exp(policy.config["beta"] *
                         (adv / (1e-8 + torch.pow(policy.ma_adv_norm, 0.5))))
    # log\pi_\theta(a|s)
    logprobs = action_dist.logp(actions)
    policy.p_loss = -1.0 * torch.mean(exp_advs.detach() * logprobs)

    # Combine both losses.
    policy.total_loss = policy.p_loss + policy.config["vf_coeff"] * \
        policy.v_loss
    explained_var = explained_variance(advantages, state_values)
    policy.explained_variance = torch.mean(explained_var)

    return policy.total_loss

MARWILSTorchPolicy = MARWILTorchPolicy.with_updates(
    loss_fn=marwil_loss,
    # postprocess_fn=postprocess_advantages
)

def get_policy_class(config):
    return MARWILSTorchPolicy

MARWILSTrainer = MARWILTrainer.with_updates(
    name="MARWILS",
    default_config=MARWIL_CONFIG,
    get_policy_class=get_policy_class,
    default_policy=MARWILSTorchPolicy,
)