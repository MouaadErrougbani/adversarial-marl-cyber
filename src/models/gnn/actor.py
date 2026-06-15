# src/models/gnn/actor.py
import torch

from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical

from src.models.gnn.encoders import GraphEncoder
from src.models.gnn.attention import (
    SimpleSelfAttention,
)
from src.models.gnn.helpers import (
    extract_hosts,
)

from src.environments.cage4.constants import (
    MAX_SERVERS,
    MAX_USERS,
    POSSIBLE_NEIGHBORS,
)

MAX_EDGES = POSSIBLE_NEIGHBORS



class InductiveActorNetwork(nn.Module):
    def __init__(self, in_dim, global_state_space=3,
                 node_action_space=4, edge_action_space=2, global_action_space=1,
                 hidden1=256, hidden2=64, gdim=64, lr=0.0003, concat_edges=False, encoder='gnn_gcn'):
        super().__init__()

        self.encoder = GraphEncoder(in_dim, hidden1, hidden2, encoder=encoder)

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
    
    def _compute_router_groups(self, action_edges, multi_subnet):
        rtrs = action_edges.unique(sorted=True).squeeze(-1)

        bs = rtrs.size(0) // 9

        rtrs = rtrs.reshape(bs, 9)

        if multi_subnet:
            rtrs = rtrs.repeat_interleave(3, 0)

        rtr_mask = torch.ones(
            rtrs.size(0),
            9,
            1,
        )

        return rtrs, rtr_mask

    def _update_global_embedding(self, x, g, attn, rtrs, rtr_mask, servers, n_servers, users, n_users):
        v, mask = extract_hosts(x, servers, n_servers, users, n_users)
        rtr = x[rtrs]

        v = torch.cat([v, rtr], dim=1)
        mask = torch.cat([mask, rtr_mask], dim=1)

        return attn(v, mask, g=g)

    def _encode_graph(self, x, ei, global_vec, rtrs, rtr_mask, servers, n_servers, users, n_users):
        x1, x2 = self.encoder(x, ei)
        g = self.global_net(global_vec)

        g = self._update_global_embedding(
            x, g,
            self.g0_attn,
            rtrs, rtr_mask,
            servers, n_servers,
            users, n_users
        )



        g = self._update_global_embedding(
            x1, g,
            self.g1_attn,
            rtrs, rtr_mask,
            servers, n_servers,
            users, n_users
        )


        g = self._update_global_embedding(
            x2, g,
            self.g2_attn,
            rtrs, rtr_mask,
            servers, n_servers,
            users, n_users
        )

        return x2, g

    def _compute_node_actions(self, x, g, servers, n_servers, users, n_users):
        z, mask = extract_hosts(x, servers, n_servers, users, n_users)

        z = torch.cat(
            [z, g.unsqueeze(1).repeat(1, z.size(1), 1)],
            dim=-1
        )

        node_a = self.node_actions(z) * mask

        nbatches = node_a.size(0)

        node_a = node_a.transpose(1, 2)

        node_a = node_a.reshape(
            nbatches,
            (MAX_SERVERS + MAX_USERS) * self.node_action_space
        )

        return node_a

    def _compute_edge_actions(
        self,
        x,
        g,
        action_edges,
        nbatches
    ):
        src, dst = action_edges

        src = x[src]
        dst = x[dst]

        if self.concat_edges:
            edge_a = self.edge_actions(
                torch.cat([src, dst], dim=-1)
            )
        else:
            edge_a = self.edge_actions(src) * self.edge_actions(dst)

        edge_a = torch.cat([
            edge_a,
            g.repeat_interleave(MAX_EDGES, 0)
        ], dim=1)

        edge_a = self.edge_out(edge_a)

        edge_a = edge_a.reshape(
            nbatches,
            MAX_EDGES,
            edge_a.size(-1)
        )

        edge_a = edge_a.transpose(1, 2)

        edge_a = edge_a.reshape(
            nbatches,
            edge_a.size(1) * MAX_EDGES
        )

        return edge_a

    def _build_action_distribution(self, node_a, edge_a, glb_a, multi_subnet):
        out = torch.cat(
            [node_a, edge_a, glb_a],
            dim=-1
        )

        if multi_subnet:
            out = out.reshape(
                out.size(0) // 3,
                out.size(1) * 3
            )

        out[out == 0] = -float('inf')

        out = self.sm(out)

        return Categorical(out)

    def forward(self, x, ei, global_vec, servers, n_servers, users, n_users, action_edges, multi_subnet):

        rtrs, rtr_mask = self._compute_router_groups(action_edges, multi_subnet)

        x, g = self._encode_graph( x, ei, global_vec, rtrs, rtr_mask, servers, n_servers, users, n_users)

        node_a = self._compute_node_actions(x, g, servers, n_servers, users, n_users)

        edge_a = self._compute_edge_actions(x, g, action_edges, node_a.size(0))

        glb_a = self.global_out(g) # B x d

        out = self._build_action_distribution(node_a, edge_a, glb_a, multi_subnet)

        return out
