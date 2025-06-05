from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

@configclass
class RslRlPpoActorCriticRecurrentCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCriticRecurrent"
    """The policy class name. Default is ActorCriticRecurrent."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

    activation: str = "elu"
    rnn_type: str = "lstm"
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 1