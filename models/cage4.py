import torch
from torch import nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from torch_geometric.nn import GCNConv

from models.memory_buffer import MultiPPOMemory
from models.utils import combine_marl_states

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except Exception:
    XLA_AVAILABLE = False

MAX_SERVERS = 6
MAX_USERS = 10
MAX_EDGES = 8


def pad_sequence(seq, lens, padding):
    padded = torch.zeros(lens.size(0), padding, seq.size(-1), device=seq.device)
    mask = torch.ones(padded.size(0), padded.size(1), device=seq.device)

    offset = 0
    for i,len in enumerate(lens):
        st = offset
        en = offset+len

        padded[i][:len] = seq[st:en]
        mask[i][len:] = 0
        offset += len

    return padded, mask.unsqueeze(-1)

def extract_hosts(x, servers, n_servers, users, n_users):
    srv = x[servers]
    srv,s_mask = pad_sequence(srv, n_servers, MAX_SERVERS)  # B x MAX_s x d

    usr = x[users]
    usr,u_mask = pad_sequence(usr, n_users, MAX_USERS)      # B x MAX_u x d

    hosts = torch.cat([srv,usr], dim=1)
    mask = torch.cat([s_mask, u_mask], dim=1)
    return hosts, mask

class SimpleSelfAttention(nn.Module):
    '''
    Implimenting global-node self-attention from
        https://arxiv.org/pdf/2009.12462.pdf
    '''
    def __init__(self, in_dim, h_dim, g_dim):
        super().__init__()

        self.att = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.Softmax(dim=-1)
        )
        self.feat = nn.Linear(in_dim, h_dim)
        self.glb = nn.Sequential(
            nn.Linear(h_dim+g_dim, g_dim),
            nn.Tanh()
        )

        self.g_dim = g_dim
        self.h_dim = h_dim

    def forward(self, v, mask, g=None):
        '''
        Inputs:
            v:      B x N x d tensor
            mask:   B x N x 1 tensor of 1s or 0s
            g:      B x d tensor
        '''
        if g is None:
            g = torch.zeros((v.size(0), self.g_dim), device=v.device)

        att = self.att(v)                   # B x N x h
        feat = self.feat(v)                 # B x N x h
        out = (att*feat*mask).sum(dim=1)    # B x h

        g_ = self.glb(torch.cat([out,g], dim=-1))  # B x g
        return g + g_                              # Short-circuit


class InductiveActorNetwork(nn.Module):
    def __init__(self, in_dim, global_state_space=3,
                 node_action_space=4, edge_action_space=2, global_action_space=1,
                 hidden1=256, hidden2=64, gdim=64, lr=0.0003, concat_edges=False):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)

        self.g0_attn = SimpleSelfAttention(in_dim, hidden1, gdim)
        self.g1_attn = SimpleSelfAttention(hidden1, hidden1, gdim)
        self.g2_attn = SimpleSelfAttention(hidden2, hidden2, gdim)

        # Just learn a good parameter to encode each phase as
        self.global_net = nn.Linear(
            global_state_space, gdim
        )

        self.node_actions = nn.Sequential(
            nn.Linear(hidden2+gdim, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden2 // 2),
            nn.ReLU(),
            nn.Linear(hidden2 // 2, node_action_space)
        )

        # If edges should be processed as
        # f(src * dst) or f(src || dst)
        self.concat_edges = concat_edges
        self.edge_actions = nn.Sequential(
            nn.Linear(hidden2 if not concat_edges else hidden2*2, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden2 // 2),
            nn.ReLU(),
        )
        self.edge_out = nn.Linear(hidden2 // 2 + gdim, edge_action_space)

        self.global_out = nn.Sequential(
            nn.Linear(gdim, gdim//2),
            nn.ReLU(),
            nn.Linear(gdim//2, global_action_space)
        )

        self.sm = nn.Softmax(dim=1)
        self.opt = Adam(self.parameters(), lr)

        self.node_action_space = node_action_space
        self.edge_action_space = edge_action_space

    def forward(self, x, ei, global_vec, servers, n_servers, users, n_users, action_edges, multi_subnet):
        # Always come in groups of 9
        rtrs = action_edges.unique(sorted=True).squeeze(-1)
        bs = rtrs.size(0) // 9
        rtrs = rtrs.reshape(bs, 9)
        if multi_subnet:
            rtrs = rtrs.repeat_interleave(3,0)

        rtr_mask = torch.ones(rtrs.size(0), 9, 1, device=x.device)

        # Global init features
        g0 = self.global_net(global_vec)

        v,mask = extract_hosts(x, servers, n_servers, users, n_users)
        rtr = x[rtrs]
        v = torch.cat([v, rtr], dim=1)
        mask = torch.cat([mask, rtr_mask], dim=1)
        g = self.g0_attn(v,mask, g=g0)

        # Layer 1
        x = torch.relu(self.conv1(x, ei))
        v,mask = extract_hosts(x, servers, n_servers, users, n_users)
        rtr = x[rtrs]
        v = torch.cat([v, rtr], dim=1)
        mask = torch.cat([mask, rtr_mask], dim=1)
        g = self.g1_attn(v,mask, g=g)

        # Layer 2
        x = torch.relu(self.conv2(x, ei))
        v,mask = extract_hosts(x, servers, n_servers, users, n_users)
        rtr = x[rtrs]
        v = torch.cat([v, rtr], dim=1)
        mask = torch.cat([mask, rtr_mask], dim=1)
        g = self.g2_attn(v,mask, g=g) # B x d_g

        # B x 16 x d
        z,mask = extract_hosts(x, servers, n_servers, users, n_users)

        # Attach global vec to all nodes in each batch
        z = torch.cat(
            [z, g.unsqueeze(1).repeat(1,z.size(1),1)],
            dim=-1
        )
        node_a = self.node_actions(z) * mask

        # B x 16 x a_n
        nbatches = node_a.size(0)

        # Make rows actions, and columns nodes
        node_a = node_a.transpose(1,2)  # B x a_n x 16
        node_a = node_a.reshape(        # B x 16*a_n
            nbatches,
            (MAX_SERVERS+MAX_USERS)*self.node_action_space
        )

        # Calculate edge-level action probs
        src,dst = action_edges
        src = x[src]; dst = x[dst]

        if self.concat_edges:
            edge_a = self.edge_actions(torch.cat([src,dst], dim=-1))
        else:
            edge_a = self.edge_actions(src) * self.edge_actions(dst)

        # Add in global vector
        edge_a = torch.cat([
            edge_a,
            g.repeat_interleave(MAX_EDGES,0)
        ], dim=1)
        edge_a = self.edge_out(edge_a)

        # Assume edge actions are always in groups of 8 (as they are in CAGE4)
        edge_a = edge_a.reshape(        # B x 8 x a_e
            node_a.size(0),
            MAX_EDGES,
            edge_a.size(-1)
        )
        edge_a = edge_a.transpose(1,2)  # B x a_e x 8 (columns are nodes, rows are actions)
        edge_a = edge_a.reshape(        # B x 8*a_e
            nbatches, edge_a.size(1)*MAX_EDGES
        )

        # Finally, compute prob of taking a global action
        # (Not an action upon a node or an edge. E.g. sleep)
        glb_a = self.global_out(g) # B x d

        out = torch.cat([node_a, edge_a, glb_a], dim=-1)

        # blue_agent_4 sends in groups of 3 subnets.
        # Really makes batching tricky
        if multi_subnet:
            out = out.reshape(out.size(0)//3, out.size(1)*3)

        out[out == 0] = -float('inf')   # So softmax prob is 0
        out = self.sm(out)

        return Categorical(out)


class InductiveCriticNetwork(nn.Module):
    def __init__(self, in_dim, global_state_space=3,
                 hidden1=256, hidden2=64, gdim=64, lr=0.001):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.out = nn.Sequential(
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, 1)
        )

        self.gs = nn.Linear(
            global_state_space, gdim
        )
        self.g0_attn = SimpleSelfAttention(in_dim, hidden1, gdim)
        self.g1_attn = SimpleSelfAttention(hidden1, hidden1, gdim)
        self.g2_attn = SimpleSelfAttention(hidden2, hidden2, gdim)

        self.out = nn.Sequential(
            nn.Linear(gdim, gdim//2),
            nn.ReLU(),
            nn.Linear(gdim//2, 1)
        )
        self.opt = Adam(self.parameters(), lr)

    def forward(self, x, ei, global_vec, servers, n_servers, users, n_users, action_edges, multi_subnet):
        g0 = self.gs(global_vec)

        v,mask = extract_hosts(x, servers, n_servers, users, n_users)
        g = self.g0_attn(v, mask, g=g0)

        x = torch.relu(self.conv1(x, ei))
        v,mask = extract_hosts(x, servers, n_servers, users, n_users)
        g = self.g1_attn(v, mask, g=g)

        x = torch.relu(self.conv2(x, ei))
        v,mask = extract_hosts(x, servers, n_servers, users, n_users)
        g = self.g2_attn(v, mask, g=g)

        # I guess just average the three global vectors together?
        if multi_subnet:
            g = g.reshape(g.size(0) // 3, 3, g.size(-1))
            g = g.mean(dim=1)

        return self.out(g)


class InductiveGraphPPOAgent():
    '''
    Class to manage agents' memories and learning (when training)
    When training is complete, uses the InductiveActorNetwork to decide
    which action to take
    '''
    def __init__(self, in_dim, gamma=0.99, lmbda=0.95, clip=0.1, bs=5, epochs=6,
                 a_kwargs=dict(), c_kwargs=dict(), training=True, concat_edges=False, device=None):

        self.device = device or torch.device("cpu")
        self.actor = InductiveActorNetwork(in_dim, concat_edges=concat_edges, **a_kwargs).to(self.device)
        self.critic = InductiveCriticNetwork(in_dim, **c_kwargs).to(self.device)
        self.memory = MultiPPOMemory(bs, agents=5)

        self.args = (in_dim,)
        self.kwargs = dict(
            gamma=gamma, lmbda=lmbda, clip=clip, bs=bs, epochs=epochs,
            a_kwargs=a_kwargs, c_kwargs=c_kwargs, training=training, concat_edges=concat_edges
        )

        # PPO Hyperparams
        self.gamma = gamma
        self.lmbda = lmbda
        self.clip = clip
        self.bs = bs
        self.epochs = epochs

        self.training = training
        self.deterministic = False
        self.mse = nn.MSELoss()

    # Required by CAGE but not utilized
    def end_episode(self):
        pass

    # Required by CAGE but not utilized
    def set_initial_values(self, action_space, observation):
        pass

    def train(self):
        '''
        Set modules to training mode
        '''
        self.actor.train()
        self.critic.train()

    def eval(self):
        '''
        Set modules to eval mode 
        '''
        self.training = False
        self.actor.eval()
        self.critic.eval()

    def _zero_grad(self):
        '''
        Reset opt
        '''
        self.actor.opt.zero_grad()
        self.critic.opt.zero_grad()

    def _step(self):
        """
        Optimizer step compatible CPU / CUDA / TPU-XLA.
        """
        use_xla = XLA_AVAILABLE and str(self.device).startswith("xla")

        if use_xla:
            xm.optimizer_step(self.actor.opt, barrier=False)
            xm.optimizer_step(self.critic.opt, barrier=True)
        else:
            self.actor.opt.step()
            self.critic.opt.step()

    def _to_device(self, value):
        return value.to(self.device) if torch.is_tensor(value) else value

    def _move_state(self, state):
        x, ei, global_vec, servers, n_servers, users, n_users, action_edges, multi_subnet = state
        return (
            self._to_device(x),
            self._to_device(ei),
            self._to_device(global_vec),
            self._to_device(servers),
            self._to_device(n_servers),
            self._to_device(users),
            self._to_device(n_users),
            self._to_device(action_edges),
            self._to_device(multi_subnet),
        )


    def set_deterministic(self, val):
        self.deterministic = val

    def set_mems(self, mems):
        self.memory.mems = mems

    def save(self, outf='saved_models/ppo.pt'):
        me = (self.args, self.kwargs)

        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'agent': me
        }, outf)

    def load_weights(self, path):
        data = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])

    @torch.no_grad()
    def get_action(self, obs, *args):
        if not hasattr(self, "_printed_device"):
            self._printed_device = True
        '''
        Sample an action from the actor's distribution
        given the current state. 

        If eval(), only returns the action 
        If train() returns action, value, and log prob 
        '''
        state,is_blocked = obs
        if is_blocked:
            return None

        state = self._move_state(state)
        distro = self.actor(*state)

        # I don't know why this would ever be called
        # during training, but just in case, putting the
        # logic block outside the training check
        if self.deterministic:
            action = distro.probs.argmax()
        else:
            action = distro.sample()

        if not self.training:
            return action.item()

        value = self.critic(*state)
        prob = distro.log_prob(action)
        return action.item(), value.item(), prob.item()

    def remember(self, idx, s, a, v, p, r, t):
        '''
        Save an observation to the agent's memory buffer
        '''
        self.memory.remember(idx, s,a,v,p,r,t)

    # def learn(self, verbose=False):
    #     '''        
    #     This runs the PPO update algorithm on memories stored in self.memory 
    #     Assumes that an external process is adding memories to the buffer
    #     '''
    #     for e in range(self.epochs):
    #         s,a,v,p,r,t, batches = self.memory.get_batches()

    #         # Calculate discounted reward
    #         rewards = []
    #         discounted_reward = 0
    #         for reward, is_terminal in zip(reversed(r), reversed(t)):
    #             if is_terminal:
    #                 discounted_reward = 0
    #             discounted_reward = reward + self.gamma * discounted_reward
    #             rewards.insert(0, discounted_reward)

    #         # Normalize 
    #         r = torch.tensor(rewards, dtype=torch.float, device=self.device)
    #         r = (r - r.mean()) / (r.std() + 1e-5) # Normalize rewards

    #         # Calculate advantage 
    #         advantages = r - torch.tensor(v, device=self.device)
    #         closs,aloss,eloss = 0,0,0

    #         # Optimize for clipped advantage for each minibatch 
    #         for b_idx,b in enumerate(batches):
    #             b = b.tolist()
    #             new_probs = []

    #             # Combine graphs from minibatches so GNN is called once
    #             s_ = [s[idx] for idx in b]
    #             a_ = [a[idx] for idx in b]
    #             batched_states = self._move_state(combine_marl_states(s_))

    #             self._zero_grad()

    #             # Forward pass 
    #             dist = self.actor(*batched_states)
    #             critic_vals = self.critic(*batched_states)

    #             new_probs = dist.log_prob(torch.tensor(a_, device=self.device))
    #             old_probs = torch.tensor([p[i] for i in b], device=self.device)
    #             entropy = dist.entropy()

    #             a_t = advantages[b]

    #             # Equiv to exp(new) / exp(old) b.c. recall: these are log probs
    #             r_theta = (new_probs - old_probs).exp()
    #             clipped_r_theta = torch.clip(
    #                 r_theta, min=1-self.clip, max=1+self.clip
    #             )

    #             # Use whichever one is minimal
    #             actor_loss = torch.min(r_theta*a_t, clipped_r_theta*a_t)
    #             actor_loss = -actor_loss.mean()

    #             # Critic uses MSE loss between expected value of state and observed
    #             # reward with discount factor
    #             critic_loss = self.mse(r[b].unsqueeze(-1), critic_vals)

    #             # Not totally necessary but maybe will help?
    #             entropy_loss = entropy.mean()

    #             # Calculate gradient and backprop
    #             total_loss = actor_loss + 0.5*critic_loss - 0.01*entropy_loss
    #             total_loss.backward()
    #             self._step()

    #             # Print loss for each minibatch if verbose 
    #             # (aggregate loss is printed regardless)
    #             if verbose:
    #                 print(f'[{e}] C-Loss: {0.5*critic_loss.item():0.4f}  A-Loss: {actor_loss.item():0.4f} E-loss: {-entropy_loss.item()*0.01:0.4f}')

    #             closs += critic_loss.item()
    #             aloss += actor_loss.item()
    #             eloss += entropy_loss.item()

    #         # Print avg loss across minibatches
    #         closs /= len(batches)
    #         aloss /= len(batches)
    #         eloss /= len(batches)
    #         print(f'[{e}] C-Loss: {0.5*closs:0.4f}  A-Loss: {aloss:0.4f} E-loss: {-eloss*0.01:0.4f}')

    #     # After we have sampled our minibatches e times, clear the memory buffer
    #     self.memory.clear()
    #     return total_loss.item()



    def learn(self, verbose=False, debug=True, debug_batches=3):
        """
        PPO update compatible CPU / CUDA / TPU-XLA.

        debug=True:
            affiche les temps des étapes importantes.
        debug_batches:
            nombre de premiers minibatches à profiler en détail par epoch.
            Exemple: debug_batches=3 affiche les 3 premiers batches seulement.
        """
        import time
        import gc
        import torch

        use_xla = XLA_AVAILABLE and str(self.device).startswith("xla")
        use_cuda = str(self.device).startswith("cuda")

        if use_xla:
            try:
                import torch_xla.core.xla_model as xm
            except Exception:
                xm = None
                use_xla = False
        else:
            xm = None

        def sync_device():
            """
            Synchronisation uniquement pour debug/timing.
            Attention: sur TPU, trop de sync ralentit l'entraînement.
            """
            if use_cuda:
                torch.cuda.synchronize()
            elif use_xla and xm is not None:
                xm.mark_step()

        def print_mem(prefix):
            """
            Affichage mémoire compatible CUDA.
            Pour XLA, on essaie seulement si la fonction existe.
            """
            if use_cuda:
                allocated = torch.cuda.memory_allocated(self.device) / 1024**2
                reserved = torch.cuda.memory_reserved(self.device) / 1024**2
                print(
                    f"{prefix} | CUDA memory allocated={allocated:.1f}MB "
                    f"reserved={reserved:.1f}MB",
                    flush=True,
                )
            elif use_xla and xm is not None:
                try:
                    info = xm.get_memory_info(str(self.device))
                    print(f"{prefix} | XLA memory info={info}", flush=True)
                except Exception:
                    print(f"{prefix} | XLA memory info unavailable", flush=True)

        total_loss = None
        last_loss_value = 0.0

        if debug:
            print(
                f"[LEARN] start | device={self.device} "
                f"use_xla={use_xla} use_cuda={use_cuda} epochs={self.epochs}",
                flush=True,
            )
            print_mem("[LEARN] initial")

        for e in range(self.epochs):
            epoch_t0 = time.time()

            if debug:
                print(f"\n[LEARN] epoch {e} start", flush=True)

            # ============================================================
            # 1) Get batches from memory
            # ============================================================
            t0 = time.time()
            s, a, v, p, rewards_raw, terminals, batches = self.memory.get_batches()

            if debug:
                print(
                    f"[LEARN] epoch {e} get_batches done "
                    f"time={time.time() - t0:.2f}s "
                    f"memory_size={len(s)} batches={len(batches)}",
                    flush=True,
                )

            # ============================================================
            # 2) Discounted rewards sur CPU
            # ============================================================
            t0 = time.time()

            rewards = []
            discounted_reward = 0.0

            for reward, is_terminal in zip(reversed(rewards_raw), reversed(terminals)):
                if is_terminal:
                    discounted_reward = 0.0

                discounted_reward = float(reward) + self.gamma * discounted_reward
                rewards.insert(0, discounted_reward)

            if debug:
                print(
                    f"[LEARN] epoch {e} discounted rewards done "
                    f"time={time.time() - t0:.2f}s",
                    flush=True,
                )

            # ============================================================
            # 3) Move scalar arrays once per epoch
            # ============================================================
            t0 = time.time()

            r = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
            v_t = torch.as_tensor(v, dtype=torch.float32, device=self.device)
            a_t_all = torch.as_tensor(a, dtype=torch.long, device=self.device)
            p_t_all = torch.as_tensor(p, dtype=torch.float32, device=self.device)

            r = (r - r.mean()) / (r.std() + 1e-5)
            advantages = r - v_t

            if debug:
                sync_device()
                print(
                    f"[LEARN] epoch {e} tensors moved to device "
                    f"time={time.time() - t0:.2f}s",
                    flush=True,
                )
                print_mem(f"[LEARN] epoch {e} after tensor move")

            # Loss accumulators as tensors.
            # Important: no .item() inside minibatch loop.
            closs = torch.zeros((), dtype=torch.float32, device=self.device)
            aloss = torch.zeros((), dtype=torch.float32, device=self.device)
            eloss = torch.zeros((), dtype=torch.float32, device=self.device)

            # ============================================================
            # 4) Minibatch PPO update
            # ============================================================
            for b_idx, b in enumerate(batches):
                batch_t0 = time.time()
                do_debug_batch = debug and (b_idx < debug_batches)

                if do_debug_batch:
                    print(
                        f"\n[LEARN] epoch {e} batch {b_idx}/{len(batches)} start",
                        flush=True,
                    )

                # --------------------------------------------------------
                # A) Convert batch indices
                # --------------------------------------------------------
                t_indices = time.time()

                if torch.is_tensor(b):
                    b_list = b.tolist()
                else:
                    b_list = list(b)

                b_idx_t = torch.as_tensor(
                    b_list,
                    dtype=torch.long,
                    device=self.device,
                )

                if do_debug_batch:
                    print(
                        f"[LEARN] epoch {e} batch {b_idx} indices done "
                        f"time={time.time() - t_indices:.2f}s "
                        f"batch_size={len(b_list)}",
                        flush=True,
                    )

                # --------------------------------------------------------
                # B) Combine graph states on CPU
                # --------------------------------------------------------
                t_combine = time.time()

                s_ = [s[idx] for idx in b_list]
                combined_state = combine_marl_states(s_)

                if do_debug_batch:
                    print(
                        f"[LEARN] epoch {e} batch {b_idx} combine_marl_states done "
                        f"time={time.time() - t_combine:.2f}s",
                        flush=True,
                    )

                # --------------------------------------------------------
                # C) Move graph batch to CPU/GPU/TPU
                # --------------------------------------------------------
                t_move = time.time()

                batched_states = self._move_state(combined_state)

                if do_debug_batch:
                    sync_device()
                    print(
                        f"[LEARN] epoch {e} batch {b_idx} move_state done "
                        f"time={time.time() - t_move:.2f}s",
                        flush=True,
                    )
                    print_mem(f"[LEARN] epoch {e} batch {b_idx} after move_state")

                # --------------------------------------------------------
                # D) Forward actor + critic
                # --------------------------------------------------------
                t_forward = time.time()

                self._zero_grad()

                dist = self.actor(*batched_states)
                critic_vals = self.critic(*batched_states)

                if do_debug_batch:
                    sync_device()
                    print(
                        f"[LEARN] epoch {e} batch {b_idx} forward done "
                        f"time={time.time() - t_forward:.2f}s",
                        flush=True,
                    )

                # --------------------------------------------------------
                # E) PPO losses
                # --------------------------------------------------------
                t_loss = time.time()

                new_probs = dist.log_prob(a_t_all[b_idx_t])
                old_probs = p_t_all[b_idx_t]
                entropy = dist.entropy()

                adv_batch = advantages[b_idx_t]

                r_theta = (new_probs - old_probs).exp()
                clipped_r_theta = torch.clip(
                    r_theta,
                    min=1 - self.clip,
                    max=1 + self.clip,
                )

                actor_loss = -torch.min(
                    r_theta * adv_batch,
                    clipped_r_theta * adv_batch,
                ).mean()

                critic_loss = self.mse(
                    r[b_idx_t].unsqueeze(-1),
                    critic_vals,
                )

                entropy_loss = entropy.mean()

                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

                if do_debug_batch:
                    sync_device()
                    print(
                        f"[LEARN] epoch {e} batch {b_idx} loss compute done "
                        f"time={time.time() - t_loss:.2f}s",
                        flush=True,
                    )

                # --------------------------------------------------------
                # F) Backward + optimizer step
                # --------------------------------------------------------
                t_backward = time.time()

                total_loss.backward()
                self._step()

                if do_debug_batch:
                    sync_device()
                    print(
                        f"[LEARN] epoch {e} batch {b_idx} backward+step done "
                        f"time={time.time() - t_backward:.2f}s",
                        flush=True,
                    )
                    print_mem(f"[LEARN] epoch {e} batch {b_idx} after step")

                # --------------------------------------------------------
                # G) Accumulate losses without .item()
                # --------------------------------------------------------
                closs = closs + critic_loss.detach()
                aloss = aloss + actor_loss.detach()
                eloss = eloss + entropy_loss.detach()

                # Nettoyage des grosses références Python
                del s_
                del combined_state
                del batched_states
                del dist
                del critic_vals
                del new_probs
                del old_probs
                del entropy
                del adv_batch
                del r_theta
                del clipped_r_theta
                del actor_loss
                del critic_loss
                del entropy_loss

                if do_debug_batch:
                    print(
                        f"[LEARN] epoch {e} batch {b_idx} total batch time "
                        f"{time.time() - batch_t0:.2f}s",
                        flush=True,
                    )

            # ============================================================
            # 5) End epoch logs
            # ============================================================
            n_batches = max(len(batches), 1)

            # Une seule synchronisation CPU à la fin de l'epoch.
            closs_print = (0.5 * closs / n_batches).detach().cpu().item()
            aloss_print = (aloss / n_batches).detach().cpu().item()
            eloss_print = (-0.01 * eloss / n_batches).detach().cpu().item()

            print(
                f"[{e}] C-Loss: {closs_print:0.4f}  "
                f"A-Loss: {aloss_print:0.4f} "
                f"E-loss: {eloss_print:0.4f} "
                f"epoch_time={time.time() - epoch_t0:.2f}s",
                flush=True,
            )

            if total_loss is not None:
                last_loss_value = total_loss.detach().cpu().item()
            else:
                last_loss_value = 0.0

            del r
            del v_t
            del a_t_all
            del p_t_all
            del advantages
            del closs
            del aloss
            del eloss

            gc.collect()

            if use_cuda:
                torch.cuda.empty_cache()

            if use_xla and xm is not None:
                xm.mark_step()

            if debug:
                print_mem(f"[LEARN] epoch {e} end")

        self.memory.clear()

        if debug:
            print(f"[LEARN] finished | last_loss={last_loss_value:.4f}", flush=True)

        return last_loss_value






def load(in_f, device=None):
    '''
    Loads model checkpoint file 
    '''
    map_location = device or "cpu"
    data = torch.load(in_f, map_location=map_location)
    args,kwargs = data['agent']

    agent = InductiveGraphPPOAgent(*args, **kwargs, device=device)
    agent.actor.load_state_dict(data['actor'])
    agent.critic.load_state_dict(data['critic'])

    agent.eval()
    return agent

