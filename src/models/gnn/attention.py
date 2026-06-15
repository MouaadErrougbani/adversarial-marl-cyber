# src/models/gnn/attention.py
import torch
import torch.nn as nn


class SimpleSelfAttention(nn.Module):
    '''
    Implementing global-node self-attention from
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
            nn.Linear(h_dim + g_dim, g_dim),
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
            g = torch.zeros(
                v.size(0),
                self.g_dim,
                dtype=v.dtype,
                device=v.device,
            )

        att = self.att(v)                   # B x N x h
        feat = self.feat(v)                 # B x N x h
        out = (att * feat * mask).sum(dim=1) # B x h

        g_ = self.glb(
            torch.cat([out, g], dim=-1)
        )

        return g + g_