#src/models/gnn/encoders.py
import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv, GATConv


class GraphEncoder(nn.Module):
    """
    Shared GCN encoder.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        output_dim=64,
        encoder='gnn_gcn',
    ):
        self.encoder =encoder.strip().lower()
        super().__init__()
        if self.encoder == 'gnn_gcn':
            self.conv1 = GCNConv(
                input_dim,
                hidden_dim,
            )

            self.conv2 = GCNConv(
                hidden_dim,
                output_dim,
            )
        elif self.encoder == 'gnn_gat':
            self.conv1 = GATConv(
                input_dim,
                hidden_dim,
                heads=8,
                concat=False,
                dropout=0.6
            )

            self.conv2 = GATConv(
                hidden_dim,
                output_dim,
                heads=1,
                concat=False,
                dropout=0.6
            )
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder}")

    def forward(
        self,
        x,
        edge_index,
    ):

        h1 = torch.relu(
            self.conv1(
                x,
                edge_index,
            )
        )

        h2 = torch.relu(
            self.conv2(
                h1,
                edge_index,
            )
        )

        return h1, h2