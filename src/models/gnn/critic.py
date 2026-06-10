# src/models/gnn/critic.py

import torch
from torch import nn
from torch.optim import Adam

from src.models.gnn.attention import (
    SimpleSelfAttention,
)

from src.models.gnn.encoders import (
    GraphEncoder,
)

from src.models.gnn.helpers import (
    extract_hosts,
)

    
class InductiveCriticNetwork(nn.Module):
    def __init__(self, in_dim, global_state_space=3,
                 hidden1=256, hidden2=64, gdim=64, lr=0.001, encoder='gnn_gcn'):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hidden1, hidden2, encoder=encoder)

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


    def _encode_graph(
        self,
        x,
        ei,
        global_vec,
        servers,
        n_servers,
        users,
        n_users
    ):
        x1, x2 = self.encoder(x, ei)
        g0 = self.gs(global_vec)

        v,mask = extract_hosts(x, servers, n_servers, users, n_users)
        g = self.g0_attn(v, mask, g=g0)

        
        v,mask = extract_hosts(x1, servers, n_servers, users, n_users)
        g = self.g1_attn(v, mask, g=g)

        v,mask = extract_hosts(x2, servers, n_servers, users, n_users)
        g = self.g2_attn(v, mask, g=g)
        return g

    def forward(self, x, ei, global_vec, servers, n_servers, users, n_users, action_edges, multi_subnet):
        
        g = self._encode_graph(x, ei, global_vec, servers, n_servers, users, n_users)
        
        # I guess just average the three global vectors together?
        if multi_subnet:
            g = g.reshape(g.size(0) // 3, 3, g.size(-1))
            g = g.mean(dim=1)

        return self.out(g)
