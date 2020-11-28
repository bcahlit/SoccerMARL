import logging
import numpy as np
import gym

from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, \
    normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch, get_activation_fn
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


def get_action_dim(action_space: gym.Space):
    """Returns action dim,

    Args:
        action_space (Space): Action space of the target gym env.
    Returns:
        
    """

    if isinstance(action_space, gym.spaces.Discrete):
        return action_space.n
    elif isinstance(action_space, (gym.spaces.Box)):
        return  np.product(action_space.shape)*2 # 对角高斯
    elif isinstance(action_space, (gym.spaces.Tuple, gym.spaces.Dict)):
        flat_action_space = flatten_space(action_space)
        size = 0
        all_discrete = True
        for i in range(len(flat_action_space)):
            size += get_action_dim(flat_action_space[i])
        return size
    else:
        raise NotImplementedError(
            "Action space {} not supported".format(action_space))

class PADDPGTorchModel(TorchModelV2, nn.Module):
    """Extension of standard TorchModelV2 for DDPG.

    Data flow:
        obs -> forward() -> model_out
        model_out -> get_policy_output() -> pi(s)
        model_out, actions -> get_q_values() -> Q(s, a)
        model_out, actions -> get_twin_q_values() -> Q_twin(s, a)

    Note that this class by itself is not a valid model unless you
    implement forward() in a subclass."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 actor_hidden_activation="relu",
                 actor_hiddens=(256, 256),
                 critic_hidden_activation="relu",
                 critic_hiddens=(256, 256),
                 twin_q=False,
                 add_layer_norm=False):
        """Initialize variables of this model.

        Extra model kwargs:
            actor_hidden_activation (str): activation for actor network
            actor_hiddens (list): hidden layers sizes for actor network
            critic_hidden_activation (str): activation for critic network
            critic_hiddens (list): hidden layers sizes for critic network
            twin_q (bool): build twin Q networks.
            add_layer_norm (bool): Enable layer norm (for param noise).

        Note that the core layers for forward() are not defined here, this
        only defines the layers for the output heads. Those layers for
        forward() should be defined in subclasses of DDPGTorchModel.
        """
        nn.Module.__init__(self)
        super(PADDPGTorchModel, self).__init__(obs_space, action_space,
                                             num_outputs, model_config, name)

        self.bounded = False
        # self.bounded = np.logical_and(self.action_space.bounded_above,
        #                               self.action_space.bounded_below).any()
        # low_action = nn.Parameter(
        #     torch.from_numpy(self.action_space.low).float())
        # low_action.requires_grad = False
        # self.register_parameter("low_action", low_action)
        # action_range = nn.Parameter(
        #     torch.from_numpy(self.action_space.high -
        #                      self.action_space.low).float())
        # action_range.requires_grad = False
        # self.register_parameter("action_range", action_range)
        
        self.action_dist_dim = get_action_dim(self.action_space)
        #TODO　这个地方可以更加灵活
        self.action_dim = 6

        # Build the policy network.
        self.policy_model = nn.Sequential()
        ins = num_outputs
        self.obs_ins = ins
        activation = get_activation_fn(
            actor_hidden_activation, framework="torch")
        for i, n in enumerate(actor_hiddens):
            self.policy_model.add_module(
                "action_{}".format(i),
                SlimFC(
                    ins,
                    n,
                    initializer=torch.nn.init.xavier_uniform_,
                    activation_fn=activation))
            # Add LayerNorm after each Dense.
            if add_layer_norm:
                self.policy_model.add_module("LayerNorm_A_{}".format(i),
                                             nn.LayerNorm(n))
            ins = n

        self.policy_model.add_module(
            "action_out",
            SlimFC(
                ins,
                self.action_dist_dim,
                initializer=torch.nn.init.xavier_uniform_,
                activation_fn=None))

        # Use sigmoid to scale to [0,1], but also double magnitude of input to
        # emulate behaviour of tanh activation used in DDPG and TD3 papers.
        # After sigmoid squashing, re-scale to env action space bounds.
        class _Lambda(nn.Module):
            def forward(self_, x):
                sigmoid_out = nn.Sigmoid()(2.0 * x)
                squashed = self.action_range * sigmoid_out + self.low_action
                return squashed

        # Only squash if we have bounded actions.
        if self.bounded:
            self.policy_model.add_module("action_out_squashed", _Lambda())

        # Build the Q-net(s), including target Q-net(s).
        def build_q_net(name_):
            activation = get_activation_fn(
                critic_hidden_activation, framework="torch")
            # For continuous actions: Feed obs and actions (concatenated)
            # through the NN. For discrete actions, only obs.
            q_net = nn.Sequential()
            ins = self.obs_ins + self.action_dim
            print("ins:", ins, self.obs_ins, self.action_dim)
            for i, n in enumerate(critic_hiddens):
                q_net.add_module(
                    "{}_hidden_{}".format(name_, i),
                    SlimFC(
                        ins,
                        n,
                        initializer=torch.nn.init.xavier_uniform_,
                        activation_fn=activation))
                ins = n

            q_net.add_module(
                "{}_out".format(name_),
                SlimFC(
                    ins,
                    1,
                    initializer=torch.nn.init.xavier_uniform_,
                    activation_fn=None))
            return q_net

        self.q_model = build_q_net("q")
        if twin_q:
            self.twin_q_model = build_q_net("twin_q")
        else:
            self.twin_q_model = None

    def get_q_values(self, model_out, actions):
        """Return the Q estimates for the most recent forward pass.

        This implements Q(s, a).

        Args:
            model_out (Tensor): obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].
            actions (Tensor): Actions to return the Q-values for.
                Shape: [BATCH_SIZE, action_dim].

        Returns:
            tensor of shape [BATCH_SIZE].
        """
        print("model_out", model_out.shape)
        print("actions", actions.shape)
        return self.q_model(torch.cat([model_out, actions], -1))

    def get_twin_q_values(self, model_out, actions):
        """Same as get_q_values but using the twin Q net.

        This implements the twin Q(s, a).

        Args:
            model_out (Tensor): obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].
            actions (Optional[Tensor]): Actions to return the Q-values for.
                Shape: [BATCH_SIZE, action_dim].

        Returns:
            tensor of shape [BATCH_SIZE].
        """
        return self.twin_q_model(torch.cat([model_out, actions], -1))

    def get_policy_output(self, model_out):
        """Return the action output for the most recent forward pass.

        This outputs the support for pi(s). For continuous action spaces, this
        is the action directly. For discrete, is is the mean / std dev.

        Args:
            model_out (Tensor): obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].

        Returns:
            tensor of shape [BATCH_SIZE, action_out_size]
        """
        return self.policy_model(model_out)

    def policy_variables(self, as_dict=False):
        """Return the list of variables for the policy net."""
        if as_dict:
            return self.policy_model.state_dict()
        return list(self.policy_model.parameters())

    def q_variables(self, as_dict=False):
        """Return the list of variables for Q / twin Q nets."""
        if as_dict:
            return {
                **self.q_model.state_dict(),
                **(self.twin_q_model.state_dict() if self.twin_q_model else {})
            }
        return list(self.q_model.parameters()) + \
            (list(self.twin_q_model.parameters()) if self.twin_q_model else [])
