# src/models/agents/ppo_agent.py
import torch

from torch import nn

from src.models.memory import (
    MultiPPOMemory, 
)

from src.models.utils import (
    combine_marl_states,
)

from src.models.agents.agent import InductiveGraphAgent


class InductiveGraphPPOAgent(InductiveGraphAgent):
    '''
    Class to manage agents' memories and learning (when training)
    When training is complete, uses the InductiveActorNetwork to decide
    which action to take
    '''
    def __init__(self, in_dim, gamma=0.99, lmbda=0.95, clip=0.1, bs=5, epochs=6,
                 a_kwargs=None, c_kwargs=None, training=True, concat_edges=False, agent_count=5, device="cpu"):
        a_kwargs = a_kwargs or {}
        c_kwargs = c_kwargs or {}
        super().__init__(in_dim, a_kwargs, c_kwargs, training, concat_edges, device=device)

        self.memory = MultiPPOMemory(bs, agents=agent_count)

        self.args = (in_dim,)
        self.kwargs = dict(
            gamma=gamma, lmbda=lmbda, clip=clip, bs=bs, epochs=epochs,
            a_kwargs=a_kwargs, c_kwargs=c_kwargs, training=training, concat_edges=concat_edges, device=device, agent_count=agent_count
        )

        # PPO Hyperparams
        self.gamma = gamma
        self.lmbda = lmbda
        self.clip = clip
        self.bs = bs
        self.epochs = epochs

        self.mse = nn.MSELoss()

    def remember(self, idx, s, a, v, p, r, t):
        '''
        Save an observation to the agent's memory buffer
        '''
        self.memory.remember(idx, s,a,v,p,r,t)

    def _compute_returns(self, rewards, terminals):
        returns = []
        discounted_return = 0

        for reward, is_terminal in zip(reversed(rewards), reversed(terminals)):
            if is_terminal:
                discounted_return = 0

            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)

        returns = torch.tensor(returns, dtype=torch.float32)

        return returns

    def _compute_advantages(self, returns, values):
        values = torch.tensor(values, dtype=torch.float32)

        advantages = returns - values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages.detach()

    def _compute_actor_loss(self, new_log_probs, old_log_probs, advantages):
        ratio = torch.exp(new_log_probs - old_log_probs)

        clipped_ratio = torch.clamp(
            ratio,
            min=1 - self.clip,
            max=1 + self.clip
        )

        unclipped_objective = ratio * advantages
        clipped_objective = clipped_ratio * advantages

        actor_loss = -torch.min(
            unclipped_objective,
            clipped_objective
        ).mean()

        return actor_loss
    
    def _compute_critic_loss(self, critic_values, returns):
        return self.mse(
            critic_values,
            returns.unsqueeze(-1)
        )
    
    def _compute_entropy_loss(self, dist):
        return dist.entropy().mean()

    def _compute_total_loss(self, actor_loss, critic_loss, entropy_loss):
        total_loss = (
            actor_loss
            + 0.5 * critic_loss
            - 0.01 * entropy_loss
        )

        return total_loss

    def learn(self, verbose=False):
        '''        
        This runs the PPO update algorithm on memories stored in self.memory 
        Assumes that an external process is adding memories to the buffer
        '''
        

        for e in range(self.epochs):
            s,a,v,p,r,t, batches = self.memory.get_batches()

            returns = self._compute_returns(r, t)
            
            advantages = self._compute_advantages(returns, v)

            device = self.device
            # Optimize for clipped advantage for each minibatch 
            for b_idx,b in enumerate(batches):
                b = b.tolist()

                # Combine graphs from minibatches so GNN is called once
                s_ = [s[idx] for idx in b]
                a_ = [a[idx] for idx in b]
                batched_states = combine_marl_states(s_)
                batched_states = self._move_to_device(batched_states)

                self._zero_grad()

                # Forward pass 
                dist = self.actor(*batched_states)
                critic_vals = self.critic(*batched_states)

                actions = torch.tensor(a_, dtype=torch.long, device=device)
                new_log_probs  = dist.log_prob(actions)
                old_log_probs = torch.tensor([p[i] for i in b], dtype=torch.float32, device=device)

                a_t = advantages[b].to(device)
                batch_returns = returns[b].to(device)

                actor_loss = self._compute_actor_loss(
                    new_log_probs=new_log_probs,
                    old_log_probs=old_log_probs,
                    advantages=a_t
                )
                
                critic_loss = self._compute_critic_loss(
                    critic_values=critic_vals,
                    returns=batch_returns
                )

                entropy_loss = self._compute_entropy_loss(dist)

                total_loss = self._compute_total_loss(
                    actor_loss=actor_loss,
                    critic_loss=critic_loss,
                    entropy_loss=entropy_loss
                )

                total_loss.backward()
                self._step()

                # Print loss for each minibatch if verbose 
                # (aggregate loss is printed regardless)
                if verbose:
                    print(f'[{e}] C-Loss: {0.5*critic_loss.item():0.4f}  A-Loss: {actor_loss.item():0.4f} E-loss: {-entropy_loss.item()*0.01:0.4f}', flush=True)



        # After we have sampled our minibatches e times, clear the memory buffer
        self.memory.clear()
        return total_loss.item()

def load(in_f, device="cpu"):
    data = torch.load(in_f, map_location="cpu")
    args, kwargs = data["agent"]

    kwargs["device"] = device

    agent = InductiveGraphPPOAgent(*args, **kwargs)
    agent.actor.load_state_dict(data["actor"])
    agent.critic.load_state_dict(data["critic"])
    agent.to(device)

    agent.eval()
    return agent

