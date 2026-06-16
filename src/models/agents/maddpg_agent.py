# src/models/agents/maddpg_agent.py

import copy

import torch

from src.models.agents.agent import (
    InductiveGraphAgent,
)

from src.models.memory import (
    MultiMADDPGMemory,
)

from src.models.gnn.centralized_action_critic import (
    CentralizedActionCriticNetwork,
)

from torch import nn


class InductiveGraphMADDPGAgent(
    InductiveGraphAgent
):
    """
    Agent MADDPG discret.
    """

    def __init__(
        self,
        in_dim,
        gamma=0.99,
        tau=0.01,
        bs=128,
        a_kwargs=None,
        c_kwargs=None,
        training=True,
        concat_edges=False,
        num_agents=5,
        actor=None,
        critic=None,
    ):
        a_kwargs = a_kwargs or {}
        c_kwargs = c_kwargs or {}

        critic = critic or (
            CentralizedActionCriticNetwork(
                in_dim=in_dim,
                num_agents=num_agents,
                **c_kwargs,
            )
        )

        super().__init__(
            in_dim=in_dim,
            a_kwargs=a_kwargs,
            c_kwargs=c_kwargs,
            training=training,
            concat_edges=concat_edges,
            actor=actor,
            critic=critic,
        )

        self.external_actor = (
            actor is not None
        )

        self.external_critic = (
            critic is not None
        )

        #
        # Hyperparamètres MADDPG
        #

        self.gamma = gamma
        self.tau = tau
        self.bs = bs

        self.num_agents = num_agents

        #
        # Replay Buffer
        #

        self.memory = (
            MultiMADDPGMemory(
                batch_size=bs,
                num_agents=num_agents,
            )
        )

        #
        # Target Networks
        #

        self.target_actor = copy.deepcopy(
            self.actor
        )

        self.target_critic = copy.deepcopy(
            self.critic
        )

        self.target_actor.eval()
        self.target_critic.eval()

        #
        # Sauvegarde
        #

        self.args = (
            in_dim,
        )

        self.kwargs = dict(
            gamma=gamma,
            tau=tau,
            bs=bs,
            a_kwargs=a_kwargs,
            c_kwargs=c_kwargs,
            training=training,
            concat_edges=concat_edges,
            num_agents=num_agents,
        )

        self._algorithm = "MADDPG"

        self.agents = None

        self.mse = nn.MSELoss()

    @torch.no_grad()
    def get_action(
        self,
        obs,
        *args,
    ):
        """
        Sélectionne une action discrète.
        """

        local_observation, is_blocked = obs

        if is_blocked:
            return None

        distro = self.actor(
            *local_observation
        )

        if self.deterministic:

            action = (
                distro.probs.argmax()
            )

        else:

            action = (
                distro.sample()
            )

        return action.item()

    def remember(
        self,
        agent_idx,
        *args
    ):
        """
        Stocke une transition MADDPG.
        """

        self.memory.remember(
            agent_idx,
            *args
        )
    
    def soft_update(self):
        """
        Mise à jour douce des réseaux cibles.
        """

        #
        # Actor
        #

        for target_param, param in zip(
            self.target_actor.parameters(),
            self.actor.parameters(),
        ):

            target_param.data.copy_(

                self.tau
                * param.data

                +

                (1.0 - self.tau)
                * target_param.data
            )

        #
        # Critic
        #

        for target_param, param in zip(
            self.target_critic.parameters(),
            self.critic.parameters(),
        ):

            target_param.data.copy_(

                self.tau
                * param.data

                +

                (1.0 - self.tau)
                * target_param.data
            )

    def set_agents(
        self,
        agents,
    ):
        """
        Référence vers tous les agents MADDPG.
        """

        self.agents = agents

    def _compute_target_q(
        self,
        rewards,
        terminals,
        next_q_values,
    ):
        """
        Calcule la cible TD.
        """

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
        ).unsqueeze(-1)

        terminals = torch.tensor(
            terminals,
            dtype=torch.float32,
        ).unsqueeze(-1)

        targets = (

            rewards

            +

            self.gamma
            * (1.0 - terminals)
            * next_q_values

        )

        return targets.detach()

    def _compute_critic_loss(
        self,
        q_values,
        target_q_values,
    ):
        """
        Perte du critic.
        """

        return self.mse(
            q_values,
            target_q_values,
        )

    def _compute_actor_loss(
        self,
        q_values,
    ):
        """
        Perte actor MADDPG.
        """

        return -q_values.mean()


    def _predict_joint_actions(
        self,
        batch_joint_local_observations,
        use_target=False,
    ):
        """
        Parameters
        ----------
        batch_joint_local_observations

            [
                [
                    obs_agent0,
                    obs_agent1,
                    ...
                    obs_agent4
                ],
                ...
            ]

        Returns
        -------
        Tensor[B,num_agents]
        """

        batch_actions = []

        for joint_obs in batch_joint_local_observations:

            actions = []

            for agent, obs in zip(
                self.agents,
                joint_obs,
            ):

                actor = (
                    agent.target_actor
                    if use_target
                    else agent.actor
                )

                dist = actor(
                    *obs
                )

                action = (
                    dist.probs.argmax(
                        dim=-1
                    )
                )

                actions.append(
                    action.item()
                )

            batch_actions.append(
                actions
            )

        return torch.tensor(
            batch_actions,
            dtype=torch.long,
        )

    def _build_actor_joint_actions(
        self,
        batch_local_obs,
        batch_joint_actions,
    ):
        """
        Construit les actions jointes
        pour l'update Actor.
        """

        joint_actions = torch.tensor(
            batch_joint_actions,
            dtype=torch.long,
        )

        predicted_actions = []

        for obs in batch_local_obs:

            dist = self.actor(
                *obs
            )

            action = (
                dist.probs.argmax(
                    dim=-1
                )
            )

            predicted_actions.append(
                action.item()
            )

        predicted_actions = torch.tensor(
            predicted_actions,
            dtype=torch.long,
        )

        #
        # Remplace l'action de cet agent
        #

        joint_actions[:, self.agent_id] = (
            predicted_actions
        )

        return joint_actions

    def _build_actor_joint_actions_probs(
        self,
        batch_local_obs,
        batch_joint_actions,
    ):
        """
        Construit les actions jointes pour
        l'update Actor.

        Les autres agents utilisent
        les actions du replay buffer.

        L'agent courant utilise
        ses probabilités d'actions.
        """

        joint_actions = torch.tensor(
            batch_joint_actions,
            dtype=torch.long,
        )

        #
        # [B, num_agents, action_dim]
        #

        joint_actions_probs = torch.nn.functional.one_hot(
            joint_actions,
            num_classes=self.critic.action_dim,
        ).float()

        #
        # Probabilités du policy courant
        #

        actor_probs = []

        for obs in batch_local_obs:

            dist = self.actor(
                *obs
            )

            actor_probs.append(
                dist.probs
            )

        actor_probs = torch.stack(
            actor_probs,
            dim=0,
        )

        #
        # Remplace uniquement
        # les actions de cet agent
        #

        joint_actions_probs[
            :,
            self.agent_id,
            :
        ] = actor_probs

        return joint_actions_probs


    def learn(
        self,
        verbose=False,
    ):
        """
        Entraînement MADDPG discret.
        Version fonctionnelle de validation.
        """

        (
            s,
            g,

            a,
            r,

            ns,
            ng,

            joint_ns,

            t,

            ja,

            batches,

        ) = self.memory.get_batches()

        critic_loss_sum = 0.0
        actor_loss_sum = 0.0

        update_count = 0

        for b in batches:

            b = b.tolist()

            #
            # Batch
            #

            batch_global_obs = [
                g[idx]
                for idx in b
            ]

            batch_next_global_obs = [
                ng[idx]
                for idx in b
            ]

            batch_joint_next_local_obs = [
                joint_ns[idx]
                for idx in b
            ]

            batch_joint_actions = [
                ja[idx]
                for idx in b
            ]

            batch_rewards = [
                r[idx]
                for idx in b
            ]

            batch_terminals = [
                t[idx]
                for idx in b
            ]

            #
            # ---------- TARGET Q ----------
            #

            next_joint_actions = []

            for joint_obs in batch_joint_next_local_obs:

                actions = []

                for agent, obs in zip(
                    self.agents,
                    joint_obs,
                ):

                    with torch.no_grad():

                        dist = (
                            agent.target_actor(
                                *obs
                            )
                        )

                        action = (
                            dist.probs.argmax(
                                dim=-1
                            )
                        )

                        actions.append(
                            int(action)
                        )

                next_joint_actions.append(
                    actions
                )

            next_joint_actions = torch.tensor(
                next_joint_actions,
                dtype=torch.long,
            )

            with torch.no_grad():

                next_q_values = (
                    self.target_critic(
                        batch_next_global_obs,
                        next_joint_actions,
                    )
                )

                target_q_values = (
                    self._compute_target_q(
                        rewards=batch_rewards,
                        terminals=batch_terminals,
                        next_q_values=next_q_values,
                    )
                )

            #
            # ---------- CURRENT Q ----------
            #

            joint_actions_tensor = torch.tensor(
                batch_joint_actions,
                dtype=torch.long,
            )

            current_q_values = (
                self.critic(
                    batch_global_obs,
                    joint_actions_tensor,
                )
            )

            critic_loss = (
                self._compute_critic_loss(
                    current_q_values,
                    target_q_values,
                )
            )

            #
            # Critic update
            #

            self.critic.opt.zero_grad()

            critic_loss.backward()

            self.critic.opt.step()

            critic_loss_sum += (
                critic_loss.item()
            )

            #
            # ---------- ACTOR ----------
            #

            batch_local_obs = [
                s[idx]
                for idx in b
            ]

            actor_joint_actions = (
                self._build_actor_joint_actions(
                    batch_local_obs,
                    batch_joint_actions,
                )
            )

            actor_q_values = (
                self.critic(
                    batch_global_obs,
                    actor_joint_actions,
                )
            )

            actor_loss = (
                self._compute_actor_loss(
                    actor_q_values
                )
            )

            self.actor.opt.zero_grad()

            actor_loss.backward()

            self.actor.opt.step()

            actor_loss_sum += (
                actor_loss.item()
            )

            #
            # Targets
            #

            self.soft_update()

            update_count += 1

            if verbose:

                print(
                    f"Critic Loss: "
                    f"{critic_loss.item():0.6f} "
                    f"Actor Loss: "
                    f"{actor_loss.item():0.6f}"
                )

        self.memory.clear()

        if update_count == 0:

            return {
                "total_loss": 0.0,
                "critic_loss": 0.0,
                "actor_loss": 0.0,
            }

        return {

            "total_loss":
                (
                    critic_loss_sum
                    + actor_loss_sum
                ) / update_count,

            "critic_loss":
                critic_loss_sum
                / update_count,

            "actor_loss":
                actor_loss_sum
                / update_count,
        }

