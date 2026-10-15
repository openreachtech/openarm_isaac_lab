# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The RMA teacher: an environment encoder in front of an otherwise ordinary policy.

Liang et al. eq. (1) -- ``a_t = pi(x_t, z_t, g)`` with ``z_t = mu(e, s_t)``, both trained
together by PPO. The bottleneck is the point: in the second phase a depth student has to
reproduce ``z`` from what a camera and a history of joint angles can show, and that is only
possible if ``z`` is small and smooth.

The integration with rsl-rl is deliberately thin. ``ActorCritic`` concatenates the groups
named in ``obs_groups["policy"]`` and passes the result to ``self.actor``
(``rsl_rl/modules/actor_critic.py:166``), so listing the privileged group there and
swapping ``self.actor`` for :class:`RmaActor` is the whole change -- ``act``,
``act_inference``, ``evaluate`` and the critic are untouched. The critic then reads the raw
privileged vector through the same concatenation, which is the asymmetric critic we want:
the 16-dimensional bottleneck should constrain the actor, not the value estimate.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP


class RmaActor(nn.Module):
    """``[x_t | e, s_t]`` in, actions out, with the privileged half squeezed through ``z``.

    The input arrives as one flat tensor because that is what ``ActorCritic`` hands to its
    actor; the privileged block is sliced off the end, which is why the environment config
    has to list the privileged group last.
    """

    def __init__(
        self,
        num_policy_obs: int,
        num_privileged_obs: int,
        num_actions: int,
        category_slice: tuple[int, int],
        category_embed_dim: int,
        latent_dim: int,
        encoder_hidden_dims: list[int],
        policy_hidden_dims: list[int],
        activation: str,
    ) -> None:
        super().__init__()
        self.num_policy_obs = num_policy_obs
        self.num_privileged_obs = num_privileged_obs
        self.category_start, self.category_end = category_slice
        num_categories = self.category_end - self.category_start

        # A bias-free Linear over a one-hot *is* an embedding table: the j-th column is the
        # vector for category j, trained by the same gradient as nn.Embedding(C, d) would
        # be. Written this way so the observation can stay float (no long cast inside the
        # model) and so the table can be read straight off `.weight` afterwards -- which
        # categories the policy has decided are alike is a useful thing to be able to look
        # at when a shape family underperforms.
        self.category_embed = nn.Linear(num_categories, category_embed_dim, bias=False)

        encoder_in = num_privileged_obs - num_categories + category_embed_dim
        self.encoder = MLP(encoder_in, latent_dim, encoder_hidden_dims, activation)
        self.policy = MLP(num_policy_obs + latent_dim, num_actions, policy_hidden_dims, activation)

    def encode(self, privileged: torch.Tensor) -> torch.Tensor:
        """The privileged vector compressed to ``z``. Also the student's regression target."""
        one_hot = privileged[..., self.category_start : self.category_end]
        rest = torch.cat(
            [privileged[..., : self.category_start], privileged[..., self.category_end :]], dim=-1
        )
        return self.encoder(torch.cat([rest, self.category_embed(one_hot)], dim=-1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        policy_obs = obs[..., : self.num_policy_obs]
        privileged = obs[..., self.num_policy_obs :]
        return self.policy(torch.cat([policy_obs, self.encode(privileged)], dim=-1))


class RmaActorCritic(ActorCritic):
    """``ActorCritic`` with :class:`RmaActor` in place of the stock actor MLP.

    The parent builds a plain MLP actor over the concatenated observation first; it is
    replaced here and never used. Cheap, and it keeps everything else -- the action
    distribution, the noise parameterisation, the critic, checkpoint loading -- exactly as
    rsl-rl wrote it.
    """

    def __init__(
        self,
        obs,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        privileged_group: str = "privileged",
        latent_dim: int = 16,
        encoder_hidden_dims: list[int] = [128, 64],
        category_slice: tuple[int, int] = (5, 10),
        category_embed_dim: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(obs, obs_groups, num_actions, **kwargs)

        if privileged_group not in obs_groups["policy"]:
            raise ValueError(
                f"RmaActorCritic needs '{privileged_group}' in obs_groups['policy'] so the"
                " encoder can see it; found " + str(obs_groups["policy"])
            )
        if obs_groups["policy"][-1] != privileged_group:
            raise ValueError(
                f"'{privileged_group}' must be the last group in obs_groups['policy']; the"
                " actor slices it off the end of the concatenated observation."
            )

        num_privileged = obs[privileged_group].shape[-1]
        num_policy = sum(obs[group].shape[-1] for group in obs_groups["policy"]) - num_privileged

        self.actor = RmaActor(
            num_policy_obs=num_policy,
            num_privileged_obs=num_privileged,
            num_actions=num_actions,
            category_slice=category_slice,
            category_embed_dim=category_embed_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            policy_hidden_dims=kwargs.get("actor_hidden_dims", [256, 128, 64]),
            activation=kwargs.get("activation", "elu"),
        )
        print(f"RMA actor: {self.actor}")
