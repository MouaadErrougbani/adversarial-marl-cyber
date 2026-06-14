# src/models/agents/mappo_agent.py

import torch
from torch import nn

from src.models.agents.ppo_agent import (
    InductiveGraphPPOAgent,
)

from src.models.memory import (
    MultiMAPPOMemory,
)

from src.models.utils import (
    combine_marl_states,build_global_observation
)

from src.models.gnn.centralized_critic import (
    CentralizedCriticNetwork
)

class InductiveGraphMAPPOAgent(
    InductiveGraphPPOAgent
    ):
    """
    Agent MAPPO utilisant un critic centralisé.
    """

    def __init__(
        self,
        in_dim,
        gamma=0.99,
        lmbda=0.95,
        clip=0.1,
        bs=5,
        epochs=6,
        a_kwargs=None,
        c_kwargs=None,
        training=True,
        concat_edges=False,
        num_agents=5,
        device="cpu",
        actor=None,
        critic=None,
    ):
        a_kwargs = a_kwargs or {}
        c_kwargs = c_kwargs or {}
        c = critic or CentralizedCriticNetwork(
            in_dim,
            **c_kwargs
        )
        super().__init__(
            in_dim=in_dim,
            gamma=gamma,
            lmbda=lmbda,
            clip=clip,
            bs=bs,
            epochs=epochs,
            a_kwargs=a_kwargs,
            c_kwargs=c_kwargs,
            training=training,
            concat_edges=concat_edges,
            num_agents=num_agents,
            device=device,
            actor=actor,
            critic=c,
        )

        self.external_actor  = actor is not None
        self.external_critic  = critic is not None
        
        
        self.memory = MultiMAPPOMemory(
            batch_size=bs,
            num_agents=num_agents,
        )

        self._algorithm = "MAPPO"


    def remember(
        self,
        agent_idx,
        local_observation,
        global_observation,
        action,
        value,
        log_prob,
        reward,
        terminal,
    ):
        """
        Sauvegarde une transition MAPPO dans la mémoire.
        """

        self.memory.remember(
            agent_idx,
            local_observation,
            global_observation,
            action,
            value,
            log_prob,
            reward,
            terminal,
        )

    @torch.no_grad()
    def get_action(self, obs, *args):
        """
        Sélectionne une action à partir de l'observation locale
        et évalue cette action avec une observation globale.
        """

        local_observation, global_observation, is_blocked = obs

        if is_blocked:
            return None

        local_observation = self._move_to_device(
            local_observation
        )

        global_observation = self._move_to_device(
            global_observation
        )

        distro = self.actor(
            *local_observation
        )

        if self.deterministic:
            action = distro.probs.argmax()
        else:
            action = distro.sample()

        if not self.training:
            return action.item()

        value = self.critic(
            global_observation
        )

        log_prob = distro.log_prob(
            action
        )

        return (
            action.item(),
            value.item(),
            log_prob.item(),
        )

    @staticmethod
    def build_global_observation(
        observations,
    ):
        return build_global_observation(
            observations
        )

    def learn(self, verbose=False):
        total_loss_sum = 0.0
        actor_loss_sum = 0.0
        critic_loss_sum = 0.0
        update_count = 0

        for e in range(self.epochs):
            s, g, a, v, p, r, t, batches = self.memory.get_batches()

            returns = self._compute_returns(r, t)
            advantages = self._compute_advantages(returns, v)

            device = self.device

            for b in batches:
                b = b.tolist()

                s_ = [s[idx] for idx in b]
                g_ = [g[idx] for idx in b]
                a_ = [a[idx] for idx in b]

                batched_states = combine_marl_states(s_)
                batched_states = self._move_to_device(batched_states)

                batch_returns = returns[b].to(device)
                a_t = advantages[b].to(device)

                actions = torch.tensor(
                    a_,
                    dtype=torch.long,
                    device=device,
                )

                old_log_probs = torch.tensor(
                    [p[idx] for idx in b],
                    dtype=torch.float32,
                    device=device,
                )

                self._zero_grad()

                dist = self.actor(*batched_states)

                new_log_probs = dist.log_prob(actions)

                actor_loss = self._compute_actor_loss(
                    new_log_probs=new_log_probs,
                    old_log_probs=old_log_probs,
                    advantages=a_t,
                )

                entropy_loss = self._compute_entropy_loss(dist)

                if not self.external_critic:
                    critic_vals = []

                    for global_state in g_:
                        global_state = self._move_to_device(global_state)
                        critic_vals.append(
                            self.critic(global_state)
                        )

                    critic_vals = torch.cat(
                        critic_vals,
                        dim=0,
                    )

                    critic_loss = self._compute_critic_loss(
                        critic_values=critic_vals,
                        returns=batch_returns,
                    )

                    total_loss = (
                        actor_loss
                        + 0.5 * critic_loss
                        - 0.01 * entropy_loss
                    )

                else:
                    critic_loss = torch.tensor(
                        0.0,
                        dtype=torch.float32,
                        device=device,
                    )

                    total_loss = (
                        actor_loss
                        - 0.01 * entropy_loss
                    )

                total_loss.backward()
                self._step()

                total_loss_sum += float(
                    total_loss.detach().cpu().item()
                )

                actor_loss_sum += float(
                    actor_loss.detach().cpu().item()
                )

                critic_loss_sum += float(
                    critic_loss.detach().cpu().item()
                )

                update_count += 1

                if verbose:
                    print(
                        f"[{e}] "
                        f"C-Loss: {0.5 * critic_loss.item():0.4f} "
                        f"A-Loss: {actor_loss.item():0.4f} "
                        f"E-loss: {-entropy_loss.item() * 0.01:0.4f}",
                        flush=True,
                    )

        self.memory.clear()

        if update_count == 0:
            return {
                "total_loss": 0.0,
                "actor_loss": 0.0,
                "critic_loss": 0.0,
            }

        return {
            "total_loss": total_loss_sum / update_count,
            "actor_loss": actor_loss_sum / update_count,
            "critic_loss": critic_loss_sum / update_count,
        }

