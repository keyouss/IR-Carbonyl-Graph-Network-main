"""NE-GNN model definitions."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINEConv, GINConv
from torch_geometric.utils import scatter


class GCNEConv(nn.Module):
    """GCN-style convolution that uses encoded edge features."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.edge_encoder = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        edge_emb = self.edge_encoder(edge_attr)
        out = scatter(edge_emb * x[row], col, dim=0, reduce="sum")
        return F.leaky_relu(self.linear(x) + out, 0.1)


class GATEConv(nn.Module):
    """GAT-style convolution that includes edge features in attention."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.attn = nn.Linear(3 * in_dim, heads)
        self.linear = nn.Linear(in_dim, out_dim)
        self.edge_encoder = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        edge_emb = self.edge_encoder(edge_attr)
        x_cat = torch.cat([x[row], x[col], edge_emb], dim=-1)
        alpha = F.softmax(self.attn(x_cat), dim=0).unsqueeze(-1)
        h_node = self.linear(x)
        weighted = (x[row] + edge_emb).unsqueeze(1) * alpha
        out = scatter(weighted, col, dim=0, reduce="sum")
        return F.leaky_relu(h_node + out.mean(dim=1), 0.1)


class GNNGraph(nn.Module):
    """Graph encoder that pools selected carbonyl-neighbor nodes."""

    def __init__(
        self,
        node_dim: int = 20,
        edge_dim: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_nodes: int = 2,
        conv_type: str = "GINE",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.conv_type = conv_type.upper()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
        )

        if self.conv_type in {"GINE", "GCNE", "GATE"}:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
            )
        else:
            self.edge_encoder = None

        self.convs = nn.ModuleList([self._build_conv(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(0.2)

    def _build_conv(self, hidden_dim: int):
        if self.conv_type == "GCN":
            return GCNConv(hidden_dim, hidden_dim)
        if self.conv_type == "GAT":
            return GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        if self.conv_type == "GIN":
            return GINConv(self._mlp(hidden_dim), train_eps=True)
        if self.conv_type == "GINE":
            return GINEConv(self._mlp(hidden_dim), edge_dim=hidden_dim, train_eps=True)
        if self.conv_type == "GCNE":
            return GCNEConv(hidden_dim, hidden_dim)
        if self.conv_type == "GATE":
            return GATEConv(hidden_dim, hidden_dim)
        raise ValueError(f"Unsupported conv_type: {self.conv_type}")

    @staticmethod
    def _mlp(hidden_dim: int):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x, edge_index, edge_attr, data):
        h = self.node_encoder(x)
        edge_emb = self.edge_encoder(edge_attr) if self.edge_encoder else None

        for conv in self.convs:
            if self.conv_type in {"GINE", "GCNE", "GATE"}:
                h = conv(h, edge_index, edge_emb)
            else:
                h = conv(h, edge_index)
            h = self.dropout(h)

        carbonyl_mask = data.carbonyl_mask.view(-1, 1).float()
        if self.num_nodes == 1:
            selected_nodes = carbonyl_mask.squeeze() == 0.6
            return h[selected_nodes].view(-1, self.hidden_dim)
        selected_nodes = carbonyl_mask.squeeze() == 1
        return h[selected_nodes].view(-1, self.hidden_dim * self.num_nodes)


class GNNPredictor(nn.Module):
    """Regression head on top of the NE-GNN graph encoder."""

    def __init__(
        self,
        node_dim: int = 20,
        edge_dim: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_nodes: int = 2,
        conv_type: str = "GINE",
    ):
        super().__init__()
        self.gnn = GNNGraph(node_dim, edge_dim, hidden_dim, num_layers, num_nodes, conv_type)
        self.predict = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * num_nodes),
            nn.Linear(hidden_dim * num_nodes, hidden_dim * num_nodes),
            nn.BatchNorm1d(hidden_dim * num_nodes),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * num_nodes, 1),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)

    def forward(self, x, edge_index, edge_attr, data):
        graph_embed = self.gnn(x, edge_index, edge_attr, data)
        return self.predict(graph_embed).squeeze(-1)
