"""
=================================================================
model.py — PHYSICS-INFORMED GNN WITH CONTINUOUS EDGE FIELDS
=================================================================
Architecture:
    1. GNN encoder → node embeddings h_i
    2. EdgeFieldMLP → continuous u(ξ), w(ξ), θ(ξ) per element
    3. Autograd → du/dξ, d²w/dξ², d³w/dξ³ (in losses.py)
    4. Hard BCs: zero displacement at supports via ramp masking

Output per element:
    fields(ξ) = [u_axial, w_transverse, θ_rotation]
    for ξ ∈ [0, 1] along each element

I22_grad returned with requires_grad=True for dM/dI22 sensitivity.
=================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MLP, GraphNetworkLayer


class EdgeFieldMLP(nn.Module):
    """
    Continuous displacement field for ONE element.

    Input:  ξ ∈ [0,1] (local coordinate) + h_i, h_j (node embeddings)
    Output: [u_axial(ξ), w_transverse(ξ), θ_rotation(ξ)]

    ξ=0 → node i end
    ξ=1 → node j end
    """

    def __init__(self, hidden_dim, n_dofs=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + 2 * hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, n_dofs),
        )
        # Initialize last layer with small weights
        # to avoid starting at exactly zero
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, xi, h_i, h_j):
        """
        Args:
            xi:  (n_points, 1) local coordinates, requires_grad=True
            h_i: (n_points, H) source node embedding
            h_j: (n_points, H) target node embedding

        Returns:
            fields: (n_points, 3) = [u, w, θ] at each ξ point
        """
        inp = torch.cat([xi, h_i, h_j], dim=-1)
        return self.net(inp)


class FramePIGNN(nn.Module):
    """
    Physics-Informed GNN with continuous edge fields.

    Architecture:
        Encoder → N × MessagePassing → EdgeFieldMLP → Hard BCs

    Forward returns:
        h:        (N, H)        node embeddings
        xi:       (E, n_pts, 1) local coordinates (requires_grad)
        fields:   (E, n_pts, 3) [u, w, θ] per element per point
        I22_grad: (E,)          I22 with requires_grad for sensitivity
    """

    def __init__(self,
                 node_in_dim=9,
                 edge_attr_dim=11,
                 hidden_dim=128,
                 n_layers=6):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # GNN: node features → node embeddings
        self.encoder = MLP(node_in_dim, [hidden_dim], hidden_dim,
                           act=True)

        self.conv_layers = nn.ModuleList([
            GraphNetworkLayer(hidden_dim, edge_attr_dim, aggr='add')
            for _ in range(n_layers)
        ])

        # Edge field: ξ + embeddings → displacement field
        self.edge_field = EdgeFieldMLP(hidden_dim, n_dofs=3)

    def encode(self, data):
        """
        Run GNN to get node embeddings.

        Returns:
            h: (N, hidden_dim) node embeddings
        """
        h = self.encoder(data.x)

        for conv in self.conv_layers:
            h = conv(h, data.edge_index, data.edge_attr)
            h = F.silu(h)

        return h

    def evaluate_field(self, h, connectivity, xi, bc_mask=None):
        """
        Evaluate displacement field at local coordinates ξ.

        Args:
            h:            (N, H) node embeddings from GNN
            connectivity: (E, 2) element node pairs
            xi:           (E, n_pts, 1) local coords, requires_grad=True
            bc_mask:      (N,) 1 at supports, 0 at free nodes

        Returns:
            fields: (E, n_pts, 3) = [u, w, θ] per element per point
        """
        n1 = connectivity[:, 0]
        n2 = connectivity[:, 1]
        E_num = connectivity.shape[0]
        n_pts = xi.shape[1]

        h_i = h[n1]  # (E, H)
        h_j = h[n2]  # (E, H)

        # Expand for n_pts evaluation points per element
        h_i_exp = h_i.unsqueeze(1).expand(-1, n_pts, -1)
        h_j_exp = h_j.unsqueeze(1).expand(-1, n_pts, -1)

        # Flatten for MLP
        xi_flat = xi.reshape(-1, 1)
        h_i_flat = h_i_exp.reshape(-1, self.hidden_dim)
        h_j_flat = h_j_exp.reshape(-1, self.hidden_dim)

        # Evaluate field
        fields_flat = self.edge_field(xi_flat, h_i_flat, h_j_flat)
        fields = fields_flat.reshape(E_num, n_pts, 3)

        # Hard BCs via ramp masking
        if bc_mask is not None:
            bc_i = bc_mask[n1]  # (E,) 1 if node i is support
            bc_j = bc_mask[n2]  # (E,) 1 if node j is support

            xi_vals = xi.squeeze(-1)  # (E, n_pts)

            # bc_i=1: mask = ξ        (zero at ξ=0, node i)
            # bc_i=0: mask = 1        (no effect)
            mask_i = 1.0 - bc_i.unsqueeze(1) * (1.0 - xi_vals)

            # bc_j=1: mask = (1-ξ)    (zero at ξ=1, node j)
            # bc_j=0: mask = 1        (no effect)
            mask_j = 1.0 - bc_j.unsqueeze(1) * xi_vals

            bc_multiplier = (mask_i * mask_j).unsqueeze(-1)  # (E, n_pts, 1)
            fields = fields * bc_multiplier

        return fields

    def forward(self, data, n_quad_pts=5):
        """
        Full forward pass.

        Args:
            data:       PyG Data object
            n_quad_pts: number of evaluation points per element

        Returns:
            h:         (N, H)          node embeddings
            xi:        (E, n_pts, 1)   local coordinates (with grad)
            fields:    (E, n_pts, 3)   [u, w, θ]
            I22_grad:  (E,)            I22 with requires_grad for sensitivity
        """
        # 1. GNN encoding
        h = self.encode(data)

        # 2. Create ξ points with gradient tracking
        E_num = data.connectivity.shape[0]
        xi_1d = torch.linspace(0, 1, n_quad_pts, device=data.x.device)
        xi = xi_1d.unsqueeze(0).unsqueeze(-1).expand(
            E_num, -1, -1).clone()
        xi.requires_grad_(True)

        # 3. Evaluate displacement field with hard BCs
        bc_mask = data.bc_disp.squeeze(-1)
        fields = self.evaluate_field(h, data.connectivity, xi, bc_mask)

        # 4. Gradient-tracked I22 for sensitivity
        I22_grad = data.prop_I22.detach().clone().requires_grad_(True)

        return h, xi, fields, I22_grad