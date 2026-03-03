"""
GraphRiverCast (GRC): A Topology-Informed AI Foundation Model for Global River Forecasting

This module implements the core GraphRiverCast architecture, combining:
- Feature Encoder: Handles static geomorphic attributes
- Graph Encoder: GCN layers for upstream-downstream information propagation
- Temporal Encoder: GRU for modeling time-evolution of states

Reference:
    Ren et al. "Global River Forecasting with a Topology-Informed AI Foundation Model"
"""
import math
import numbers
from typing import List, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Size
from torch.nn.parameter import Parameter
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv

_shape_t = Union[int, List[int], Size]

# ============================================================
# RMSNorm Layer
# ============================================================
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: Optional[float] = 1e-6,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + self.eps) * self.weight


# ============================================================
# Graph Encoder (GCN)
# Propagates information along upstream-downstream pathways
# ============================================================
class GraphEncoder(nn.Module):
    """Graph Convolutional Network encoder for river network topology."""

    def __init__(self, input_dim, hid_dim, gnn_out_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.gnn_out_dim = gnn_out_dim
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hid_dim, add_self_loops=True))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hid_dim, hid_dim, add_self_loops=True))
        self.convs.append(GCNConv(hid_dim, gnn_out_dim, add_self_loops=True))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        batch_size, num_nodes, num_features = x.shape

        # Create batch of graphs
        data_list = [Data(x=x[i], edge_index=edge_index[i]) for i in range(batch_size)]
        batch = Batch.from_data_list(data_list)

        # Graph convolutions with residual connections
        x = batch.x
        for conv in self.convs[:-1]:
            x_new = conv(x, batch.edge_index)
            x_new = F.gelu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new

        # Last layer
        x_new = self.convs[-1](x, batch.edge_index)
        x_new = self.dropout(x_new)
        x = x + x_new

        return x.view(batch_size, num_nodes, self.gnn_out_dim)


# ============================================================
# GraphRiverCast Model
# ============================================================
class GraphRiverCast(nn.Module):
    """
    GraphRiverCast (GRC): Topology-informed foundation model for global river forecasting.

    The model integrates three complementary components:
    - Feature Encoder: Handles static geomorphic attributes (channel geometry, elevation)
    - Temporal Encoder: GRU for modeling time-evolution of states
    - Graph Encoder: GCN for propagating information along upstream-downstream pathways

    Two operational modes are supported:
    - GRC-HotStart: Initialized with river states for maximum short-term forecasting fidelity
    - GRC-ColdStart: Standalone hydrodynamic simulator driven solely by meteorological forcings
    """

    # Hardcoded model configuration
    DEFAULT_CONFIG = {
        'hid_size': 64,
        'fmix_size': 128,
        'use_river_var': False,
        'use_temporal': True,
        'use_spatial': True,
        'use_static_var': True,
        'spatial_num_layer': 2,
        'dropout_rate': 0.1,
        'eps': 1e-6,
    }

    def __init__(self, cfg: dict = None, task: dict = None):
        super().__init__()

        # Merge with defaults
        self.cfg = {**self.DEFAULT_CONFIG, **(cfg or {})}
        self.task = task or {}

        # Model configuration
        self.use_spatial = self.cfg['use_spatial']
        self.use_temporal = self.cfg['use_temporal']
        self.use_static_var = self.cfg['use_static_var']
        self.use_river_var = self.cfg.get('use_river_var', False)

        # Time window settings
        win_cfg = self.task.get('window', {}).get(self.task.get('type', 'predict'), {})
        self.hist_len = win_cfg.get('history', 365)
        self.fut_len = win_cfg.get('future', 2922)

        self.spatial_num_layer = self.cfg.get('spatial_num_layer', 2)
        self.temporal_num_layer = 1

        # Input dimensions
        base_dim = 1  # runoff
        river_dim = 3  # outflw, rivdph, storage
        static_dim = 18 if self.use_static_var else 0
        self.in_dim = base_dim + river_dim + static_dim
        self.out_dim = 3  # outflw, rivdph, storage

        self.hid_size = self.cfg['hid_size']
        self.fmix_size = self.cfg['fmix_size']
        self.eps = self.cfg.get('eps', 1e-6)
        self.dropout_rate = self.cfg.get('dropout_rate', 0.1)

        # Layers
        self.embed = nn.Linear(self.in_dim, self.hid_size)
        self.readout = nn.Linear(self.hid_size, self.out_dim)
        self.out_norm = RMSNorm([self.hid_size], eps=self.eps)

        # Feature mixing
        self.feat_mix = nn.Sequential(
            nn.Linear(self.hid_size, self.fmix_size),
            nn.SiLU(),
            nn.Linear(self.fmix_size, self.hid_size),
            nn.Dropout(self.dropout_rate),
        )
        self.fmix_norm = RMSNorm([self.hid_size], eps=self.eps)

        # Temporal module (GRU)
        if self.use_temporal:
            self.gru_cell = nn.GRU(
                input_size=self.hid_size,
                hidden_size=self.hid_size,
                num_layers=self.temporal_num_layer,
                batch_first=True,
                dropout=self.dropout_rate
            )
            self.gru_norm = RMSNorm([self.hid_size], eps=self.eps)
            self.hid_norm = RMSNorm([self.hid_size], eps=self.eps)

        # Graph Encoder (GCN) - propagates information along river network
        if self.use_spatial:
            self.graph_encoder = GraphEncoder(
                self.hid_size, self.hid_size, self.hid_size,
                num_layers=self.spatial_num_layer,
                dropout=self.dropout_rate
            )
            self.gnn_norm = RMSNorm([self.hid_size], eps=self.eps)

    def forward(self, inputs):
        river_hist = inputs.get("river_hist") if self.use_river_var else None
        runoff_hist = inputs["runoff_hist"]
        runoff_fut = inputs["runoff_fut"]
        static_var = inputs.get("static_var") if self.use_static_var else None
        edge_index = inputs["edge_index"]

        # Concatenate history and future runoff
        x_in = torch.cat([runoff_hist, runoff_fut], dim=1)
        bs, ts, n, _ = x_in.shape

        # Initialize hidden state
        if self.use_temporal:
            h_t = torch.zeros(self.temporal_num_layer, bs * n, self.hid_size,
                            device=x_in.device, dtype=x_in.dtype)

        all_preds = []
        current_river = None

        for t in range(ts):
            # Get current river data
            if self.use_river_var:
                if t < self.hist_len:
                    current_river = river_hist[:, t]
                else:
                    current_river = all_preds[-1]
            else:
                if t == 0:
                    current_river = torch.zeros(bs, n, 3, device=x_in.device, dtype=x_in.dtype)
                else:
                    current_river = all_preds[-1]

            # Build input
            if self.use_static_var:
                x_t = torch.cat((x_in[:, t], current_river, static_var), dim=-1)
            else:
                x_t = torch.cat((x_in[:, t], current_river), dim=-1)

            x_t = self.embed(x_t)

            # Feature mixing
            x_t = x_t + self.feat_mix(self.fmix_norm(x_t))

            # Graph Encoder - propagate along river network topology
            if self.use_spatial:
                graph_out = self.graph_encoder(self.gnn_norm(x_t), edge_index)
                x_t = x_t + graph_out

            # Temporal module
            if self.use_temporal:
                gru_input = self.gru_norm(x_t.view(bs * n, 1, -1))
                _, h_t = self.gru_cell(gru_input, self.hid_norm(h_t))
                x_t = x_t + h_t[-1].view(bs, n, -1)

            # Output
            x_delta = self.readout(self.out_norm(x_t))
            river_t = current_river + x_delta.view(bs, n, -1) if current_river is not None else x_delta.view(bs, n, -1)
            all_preds.append(river_t)

        river_fut_hat = torch.stack(all_preds, dim=1)[:, self.hist_len:]
        return {"river_fut_hat": river_fut_hat}


# Backward-compatible alias
GCN_GRU = GraphRiverCast
