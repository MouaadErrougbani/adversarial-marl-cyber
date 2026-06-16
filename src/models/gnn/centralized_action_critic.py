# src/models/gnn/centralized_action_critic.py

import torch

from torch import nn
from torch.nn import functional as F

from src.models.gnn.centralized_critic import (
    CentralizedCriticNetwork,
)


class CentralizedActionCriticNetwork(
    CentralizedCriticNetwork
):
    """
    Critic centralisé pour MADDPG discret.

    Apprend :

        Q(global_state, joint_actions)
    """

    def __init__(
        self,
        in_dim,
        num_agents=5,
        action_dim=243,
        gdim=64,
        hidden1=256,
        **kwargs
    ):
        super().__init__(
            in_dim=in_dim,
            num_agents=num_agents,
            gdim=gdim,
            hidden1=hidden1,
            **kwargs
        )

        self.num_agents = num_agents

        self.action_dim = action_dim

        self.action_embedding_dim = 4

        self.action_embedding = nn.Embedding(
            self.action_dim,
            self.action_embedding_dim,
        )

        self.joint_action_dim = (
            self.num_agents
            * self.action_embedding_dim
        )

        #
        # Réseau Q
        #

        self.q_head = nn.Sequential(

            nn.Linear(
                gdim + self.joint_action_dim,
                256,
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                128,
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                1,
            ),
        )

    def _embed_joint_actions(
        self,
        joint_actions,
    ):
        if not torch.is_tensor(
            joint_actions
        ):
            joint_actions = torch.tensor(
                joint_actions,
                dtype=torch.long,
            )

        embedded = self.action_embedding(
            joint_actions
        )

        embedded = embedded.reshape(
            embedded.size(0),
            -1,
        )
        
        return embedded



    def _encode_single_global_observation(
        self,
        global_observation,
    ):
        """
        Encode une observation globale :

        [
            obs_agent0,
            obs_agent1,
            ...
            obs_agent4
        ]

        -> [1, gdim]
        """

        agent_embeddings = []

        for obs in global_observation:

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

            g = self._encode_graph(
                x,
                ei,
                global_vec,
                servers,
                n_servers,
                users,
                n_users,
            )

            #
            # Cas agent multi-subnet
            #

            if multi_subnet:

                g = g.reshape(
                    g.size(0) // 3,
                    3,
                    g.size(-1),
                ).mean(
                    dim=1
                )

            agent_embeddings.append(
                g.squeeze(0)
            )

        #
        # [num_agents, gdim]
        #

        agents = torch.stack(
            agent_embeddings,
            dim=0,
        )

        #
        # [1, num_agents, gdim]
        #

        agents = agents.unsqueeze(0)

        mask = torch.ones(
            1,
            agents.size(1),
            1,
            dtype=agents.dtype,
            device=agents.device,
        )

        global_embedding = (
            self.agent_attention(
                agents,
                mask,
            )
        )

        #
        # [1, gdim]
        #

        return global_embedding

    def _compute_global_embedding(
        self,
        observations,
    ):
        """
        observations :

        [
            global_obs_t0,
            global_obs_t1,
            ...
        ]

        Retour :

        [B, gdim]
        """

        batch_embeddings = []

        for global_observation in observations:

            embedding = (
                self._encode_single_global_observation(
                    global_observation
                )
            )

            batch_embeddings.append(
                embedding.squeeze(0)
            )

        return torch.stack(
            batch_embeddings,
            dim=0,
        )

    def forward(
        self,
        observations,
        joint_actions,
    ):
        """
        Parameters
        ----------
        observations :

            [
                global_obs_t0,
                global_obs_t1,
                ...
            ]

        joint_actions :

            Tensor[B, num_agents]

        Returns
        -------
        Tensor[B,1]
        """

        global_embedding = (
            self._compute_global_embedding(
                observations
            )
        )

        joint_actions = (
            self._embed_joint_actions(
                joint_actions
            )
        )

        critic_input = torch.cat(
            [
                global_embedding,
                joint_actions,
            ],
            dim=-1,
        )

        q_values = self.q_head(
            critic_input
        )

        return q_values
    
