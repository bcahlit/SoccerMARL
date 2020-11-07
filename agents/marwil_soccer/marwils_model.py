import logging
import numpy as np

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, \
    normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

class MultiActionFC(nn.Module):
    """Simple PyTorch version of `linear` function"""

    def __init__(self,
                 in_size,
                 out_size,
                 out_lens,
                 at_hiddens,
                 ap_hiddens,
                 initializer=None,
                 activation=None,
                 use_bias=True,
                 bias_init=0.0):
        super(MultiActionFC, self).__init__()
        assert sum(out_lens) == out_size
        prev_vf_layer_size=in_size
        at_layers = []
        # 动作类型，可以有激活函数
        for size in at_hiddens:
            at_layers.append(
                SlimFC(
                    in_size=prev_vf_layer_size,
                    out_size=size,
                    activation_fn=activation,
                    initializer=normc_initializer(0.5)))
            prev_vf_layer_size = size
        self._at_branch_separate = nn.Sequential(*at_layers)
        # 动作参数, 最后一层不要激活函数.(因为动作参数比较大.)
        prev_vf_layer_size=in_size
        ap_layers = []
        for size in ap_hiddens[:-1]:
            ap_layers.append(
                SlimFC(
                    in_size=prev_vf_layer_size,
                    out_size=size,
                    activation_fn=activation,
                    initializer=normc_initializer(0.5)))
            prev_vf_layer_size = size
        ap_layers.append(
                SlimFC(
                    in_size=prev_vf_layer_size,
                    out_size=ap_hiddens[-1],
                    activation_fn=None,
                    initializer=normc_initializer(0.5)))
        self._ap_branch_separate = nn.Sequential(*ap_layers)

    def forward(self, x):
        at = self._at_branch_separate(x)
        ap = self._ap_branch_separate(x)
        # print("at.shape", at.shape)
        # print("ap.shape", ap.shape)
        return torch.cat([at, ap],1)

class MARWILSModel(TorchModelV2, nn.Module):
    """Generic fully connected network."""
    '''
    instance = model_cls(obs_space, action_space, num_outputs,
                        model_config, name,
                        **customized_model_kwargs)
    '''

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        activation = model_config.get("fcnet_activation")
        hiddens = model_config.get("fcnet_hiddens")
        no_final_linear = model_config.get("no_final_linear")
        self.vf_share_layers = model_config.get("vf_share_layers")
        self.free_log_std = model_config.get("free_log_std")

        # Generate free-floating bias variables for the second half of
        # the outputs.
        if self.free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two", num_outputs)
            num_outputs = num_outputs // 2

        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # Create layers 0 to second-last.
        for size in hiddens:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation))
            prev_layer_size = size


        if num_outputs:
            self._logits = MultiActionFC(
                in_size=prev_layer_size,
                out_size=num_outputs,
                out_lens=[3, 10],
                at_hiddens=[32, 3],
                ap_hiddens=[32, 10],
                initializer=normc_initializer(0.01),
                activation=activation)
        else:
            self.num_outputs = (
                [int(np.product(obs_space.shape))] + hiddens[-1:])[-1]

        # Layer to add the log std vars to the state-dependent means.
        if self.free_log_std and self._logits:
            self._append_free_log_std = AppendBiasLayer(num_outputs)

        self._hidden_layers = nn.Sequential(*layers)

        self._value_branch_separate = None
        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            prev_vf_layer_size = int(np.product(obs_space.shape))
            vf_layers = []
            for size in hiddens:
                vf_layers.append(
                    SlimFC(
                        in_size=prev_vf_layer_size,
                        out_size=size,
                        activation_fn=activation,
                        initializer=normc_initializer(1.0)))
                prev_vf_layer_size = size
            self._value_branch_separate = nn.Sequential(*vf_layers)

        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=normc_initializer(1.0),
            activation_fn=None)
        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        logits = self._logits(self._features) if self._logits else \
            self._features
        if self.free_log_std:
            logits = self._append_free_log_std(logits)
        # print("logits:",logits.shape)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            return self._value_branch(
                self._value_branch_separate(self._last_flat_in)).squeeze(1)
        else:
            return self._value_branch(self._features).squeeze(1)