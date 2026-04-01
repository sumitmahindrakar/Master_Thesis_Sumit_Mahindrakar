"""
=================================================================
model.py — PIGNN v2: Fixed Initialization + Batched MP
=================================================================
Changes from v1:
  1. Small Xavier init on decoder last layer (NOT zero)
     → Gradients flow to ALL 829K params from step 1
  2. Default 10 MP layers (was 6)
     → Info propagates across full frame for varying loads  
  3. forward_mask for correct batched momentum conservation
     → Fixes edge ordering bug when DataLoader batches graphs
=================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


# ════════════════════════════════════════════════
# MLP Building Block
# ════════════════════════════════════════════════

class MLP(nn.Module):
    """MLP with CELU activation and optional LayerNorm."""

    def __init__(self, dims, layer_norm=False, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
        self.layer_norm = (
            nn.LayerNorm(dims[-1]) if layer_norm else None
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.celu(x)
                if self.dropout is not None:
                    x = self.dropout(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x


# ════════════════════════════════════════════════
# Momentum-Conserving Message Passing
# ════════════════════════════════════════════════

class MomentumMessagePassing(MessagePassing):
    """
    Momentum-conserving message passing: m_ij = -m_ji
    
    FIXED: Uses forward_mask (bool tensor) to correctly
    identify forward vs backward edges in batched graphs.
    
    Old code used edge_index[:, :E] which BREAKS when PyG
    concatenates [fwd1|bwd1|fwd2|bwd2|...] across graphs.
    """

    def __init__(self, hidden_dim, edge_dim):
        super().__init__(aggr='add')

        self.edge_mlp = MLP(
            [edge_dim + 2 * hidden_dim, hidden_dim, hidden_dim],
            layer_norm=True
        )
        self.node_mlp = MLP(
            [2 * hidden_dim, hidden_dim, hidden_dim],
            layer_norm=True
        )

    def forward(self, h, edge_index, edge_attr, forward_mask=None):
        """
        Args:
            h:            (N, H)  node embeddings
            edge_index:   (2, 2E) all edges (fwd + bwd)
            edge_attr:    (2E, D) edge features
            forward_mask: (2E,)   bool — True for forward edges
                          If None, falls back to first-half split
                          (only correct for single unbatched graph)
        """
        N = h.shape[0]

        # ── Select forward edges ──
        if forward_mask is not None:
            ei_fwd = edge_index[:, forward_mask]
            ea_fwd = edge_attr[forward_mask]
        else:
            E = edge_index.shape[1] // 2
            ei_fwd = edge_index[:, :E]
            ea_fwd = edge_attr[:E]

        # ── Compute messages for forward edges ──
        src_fwd = h[ei_fwd[0]]   # sender
        dst_fwd = h[ei_fwd[1]]   # receiver
        msg_fwd = self.edge_mlp(
            torch.cat([ea_fwd, dst_fwd, src_fwd], dim=-1)
        )

        # ── Aggregate: +msg to receivers, -msg to senders ──
        agg_pos = torch.zeros(N, msg_fwd.shape[1],
                              device=h.device, dtype=h.dtype)
        agg_pos.scatter_add_(
            0,
            ei_fwd[1].unsqueeze(1).expand_as(msg_fwd),
            msg_fwd
        )

        agg_neg = torch.zeros(N, msg_fwd.shape[1],
                              device=h.device, dtype=h.dtype)
        agg_neg.scatter_add_(
            0,
            ei_fwd[0].unsqueeze(1).expand_as(msg_fwd),
            -msg_fwd
        )

        agg_total = agg_pos + agg_neg

        # ── Node update with residual ──
        h_new = self.node_mlp(
            torch.cat([h, agg_total], dim=-1)
        )
        h_out = h + h_new

        # ── Edge feature update (residual, anti-symmetric) ──
        edge_attr_new = edge_attr.clone()
        if forward_mask is not None:
            bwd_mask = ~forward_mask
            edge_attr_new[forward_mask] = (
                edge_attr[forward_mask] + msg_fwd
            )
            edge_attr_new[bwd_mask] = (
                edge_attr[bwd_mask] - msg_fwd
            )
        else:
            E = edge_index.shape[1] // 2
            edge_attr_new[:E] = edge_attr[:E] + msg_fwd
            edge_attr_new[E:] = edge_attr[E:] - msg_fwd

        return h_out, edge_attr_new


# ════════════════════════════════════════════════
# PIGNN Model
# ════════════════════════════════════════════════

class PIGNN(nn.Module):
    """
    Physics-Informed GNN for 2D frame structures.
    
    v2 changes:
      - Small Xavier decoder init (gain=0.01) instead of zero
      - 10 MP layers by default (was 6)
      - forward_mask support for batched graphs
    """

    OUT_DIM = 3  # [ux/u_c, uz/u_c, θ/θ_c]

    def __init__(self,
                 node_in_dim=10,
                 edge_in_dim=7,
                 hidden_dim=128,
                 n_layers=10,
                 decoder_init_gain=0.01):
        super().__init__()
        H = hidden_dim
        self._hidden_dim = H
        self._n_layers = n_layers
        self._init_gain = decoder_init_gain

        # ── Encoders ──
        self.node_encoder = MLP(
            [node_in_dim, H, H], layer_norm=True
        )
        self.edge_encoder = MLP(
            [edge_in_dim, H, H], layer_norm=True
        )

        # ── Processor ──
        self.mp_layers = nn.ModuleList([
            MomentumMessagePassing(H, H)
            for _ in range(n_layers)
        ])

        # ── Final aggregation norm ──
        self.final_norm = nn.LayerNorm(2 * H)

        # ── Separate decoder per DOF ──
        self.decoder_ux = MLP([2 * H, H, 1], dropout=0.0)
        self.decoder_uz = MLP([2 * H, H, 1], dropout=0.0)
        self.decoder_th = MLP([2 * H, H, 1], dropout=0.0)

        # For dropout
        # self.decoder_ux = nn.Sequential(
        #     nn.Linear(2 * H, H),
        #     nn.CELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(H, 1)
        # )
        # self.decoder_uz = nn.Sequential(
        #     nn.Linear(2 * H, H),
        #     nn.CELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(H, 1)
        # )
        # self.decoder_th = nn.Sequential(
        #     nn.Linear(2 * H, H),
        #     nn.CELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(H, 1)
        # )

        # ══ FIXED: Small Xavier init (not zero) ══
        self._init_decoders(gain=decoder_init_gain)

    def _init_decoders(self, gain=0.01):
        """
        Initialize decoder last layers with small Xavier weights.
        
        WHY NOT ZERO:
          With zero weights, ∂loss/∂W_hidden = 0 for ALL hidden
          layers (chain rule through zero last-layer weights).
          Only 387 of 829,827 params get gradient → training stalls.
        
        WHY SMALL (gain=0.01):
          Initial predictions ≈ O(0.01) non-dimensional
          → Physical displacements ≈ O(0.01 × u_c)
          → Small enough that energy doesn't explode
          → Large enough for meaningful gradients everywhere
        """
        # with torch.no_grad():
        #     for decoder in [self.decoder_ux,
        #                     self.decoder_uz,
        #                     self.decoder_th]:
        #         last = decoder.layers[-1]
        #         nn.init.xavier_uniform_(
        #             last.weight, gain=gain
        #         )
        #         nn.init.zeros_(last.bias)

        # For dropout
        with torch.no_grad():
            for decoder in [self.decoder_ux, self.decoder_uz, self.decoder_th]:
                # Find last Linear layer
                for module in reversed(list(decoder.modules())):
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight, gain=gain)
                        nn.init.zeros_(module.bias)
                        break

    def forward(self, data):
        """
        Forward: node/edge features → displacement prediction.
        Returns: (N, 3) non-dimensional [ux/u_c, uz/u_c, θ/θ_c]
        """
        N = data.x.shape[0]

        # Get forward_mask for batched MP
        forward_mask = getattr(
            data, 'edge_forward_mask', None
        )

        # ── Encode ──
        h = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)

        # ── Process ──
        for mp in self.mp_layers:
            h, e = mp(h, data.edge_index, e, forward_mask)

        # ── Final aggregation ──
        incoming = torch.zeros(
            N, e.shape[1], device=h.device, dtype=h.dtype
        )
        incoming.scatter_add_(
            0,
            data.edge_index[1].unsqueeze(1).expand_as(e),
            e
        )

        z_local = self.final_norm(
            torch.cat([h, incoming], dim=-1)
        )

        # ── Decode ──
        ux = self.decoder_ux(z_local)
        uz = self.decoder_uz(z_local)
        th = self.decoder_th(z_local)
        pred = torch.cat([ux, uz, th], dim=-1)

        # ── Hard BC ──
        pred = self._apply_hard_bc(pred, data)
        # pred.retain_grad()# ADDED FOR RESIDUAL CALCULATION

        return pred

    def _apply_hard_bc(self, pred, data):
        """Zero displacement at support nodes."""
        pred = pred.clone()
        disp_mask = (1.0 - data.bc_disp)
        rot_mask  = (1.0 - data.bc_rot)
        pred[:, 0:2] *= disp_mask
        pred[:, 2:3] *= rot_mask
        return pred

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def summary(self):
        n_enc = sum(
            p.numel() for p in
            list(self.node_encoder.parameters()) +
            list(self.edge_encoder.parameters())
        )
        n_proc = sum(
            p.numel() for p in
            self.mp_layers.parameters()
        )
        n_dec = sum(
            p.numel() for p in
            list(self.decoder_ux.parameters()) +
            list(self.decoder_uz.parameters()) +
            list(self.decoder_th.parameters())
        )

        init_type = (
            "SMALL XAVIER (gain={:.0e})".format(self._init_gain)
        )

        print(f"\n{'═' * 55}")
        print(f"  PIGNN Model Summary (Energy-Based v2)")
        print(f"{'═' * 55}")
        print(f"  Node input:    "
              f"{self.node_encoder.layers[0].in_features}")
        print(f"  Edge input:    "
              f"{self.edge_encoder.layers[0].in_features}")
        print(f"  Hidden dim:    {self._hidden_dim}")
        print(f"  MP layers:     {self._n_layers}")
        print(f"  Output:        {self.OUT_DIM} "
              f"(separate decoder per DOF)")
        print(f"  Decoder init:  {init_type}")
        print(f"  Activation:    CELU")
        print(f"  Message type:  Momentum-conserving")
        print(f"  ─────────────────────────────────")
        print(f"  Encoder params:  {n_enc:>10,}")
        print(f"  Processor params:{n_proc:>10,}")
        print(f"  Decoder params:  {n_dec:>10,}")
        print(f"  Total params:    "
              f"{self.count_params():>10,}")
        print(f"{'═' * 55}\n")