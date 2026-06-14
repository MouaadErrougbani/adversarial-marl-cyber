# src/models/gnn/centralized_critic.py

import torch
from torch import nn

from src.models.gnn.critic import (
    InductiveCriticNetwork,
)

from src.models.gnn.attention import (
    SimpleSelfAttention,
)


class CentralizedCriticNetwork(
    InductiveCriticNetwork
):

    def __init__(
        self,
        in_dim,
        num_agents=5,
        gdim=64,
        hidden1=256,
        **kwargs
    ):
        super().__init__(
            in_dim=in_dim,
            gdim=gdim,
            hidden1=hidden1,
            **kwargs
        )

        self.num_agents = num_agents
        self.gdim = gdim

        # Attention entre les embeddings des agents
        self.agent_attention = SimpleSelfAttention(
            in_dim=gdim,
            h_dim=gdim,
            g_dim=gdim,
        )

        # Tête de valeur finale
        self.value_head = nn.Sequential(
            nn.Linear(gdim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, observations):
        """
        observations:
            [
                obs_agent0,
                obs_agent1,
                obs_agent2,
                obs_agent3,
                obs_agent4,
            ]
        """

        embeddings = []

        for obs in observations:

            (
                x,
                ei,
                global_vec,
                servers,
                n_servers,
                users,
                n_users,
                action_edges,
                multi_subnet,
            ) = obs

            # Encode le graphe local de l'agent
            g = self._encode_graph(
                x,
                ei,
                global_vec,
                servers,
                n_servers,
                users,
                n_users,
            )

            # Cas particulier agent 4 (multi-subnet)
            if multi_subnet:
                g = g.reshape(
                    g.size(0) // 3,
                    3,
                    g.size(-1)
                ).mean(dim=1)

            embeddings.append(g)

        # --------------------------------------------------
        # embeddings :
        # [
        #   [B,64],
        #   [B,64],
        #   [B,64],
        #   [B,64],
        #   [B,64]
        # ]
        # -->
        # [B,5,64]
        # --------------------------------------------------

        agents = torch.stack(
            embeddings,
            dim=1
        )

        mask = torch.ones(
            agents.size(0),
            agents.size(1),
            1,
            dtype=agents.dtype,
            device=agents.device,
        )

        # Fusion via attention
        global_embedding = self.agent_attention(
            agents,
            mask,
        )

        # global_embedding : [B,64]

        value = self.value_head(
            global_embedding
        )

        # value : [B,1]

        return value