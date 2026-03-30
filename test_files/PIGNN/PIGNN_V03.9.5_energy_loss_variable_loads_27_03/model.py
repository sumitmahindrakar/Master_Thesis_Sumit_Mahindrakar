"""
=================================================================
model.py — PIGNN with Zero-Init Decoder (following Dalton et al.)
=================================================================

Key design choices from the paper:
  1. Zero init on decoder last layer (start at u=0)
  2. Separate decoder MLP per output DOF
  3. CELU activation (smooth, doesn't saturate positive)
  4. Momentum-conserving message passing (m_ij = -m_ji)
  5. Residual connections in message passing
  6. LayerNorm after encoder and processor
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

    def __init__(self, dims, layer_norm=False):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
        self.layer_norm = (
            nn.LayerNorm(dims[-1]) if layer_norm else None
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.celu(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x


# ════════════════════════════════════════════════
# Momentum-Conserving Message Passing
# ════════════════════════════════════════════════

class MomentumMessagePassing(MessagePassing):
    """
    Message passing with momentum conservation:
      m_ij = edge_mlp([e_ij || h_i || h_j])
      
    Messages are anti-symmetric: m_ij = -m_ji
    This is enforced by aggregating:
      received_ij = Σ m_ij  (j→i, j>i)
      received_ji = Σ -m_ji (i→j, i<j)
      total = received_ij + received_ji
    
    For undirected graphs with edges stored as (i→j, j→i),
    we use the first half as forward, second as backward.
    """

    def __init__(self, hidden_dim, edge_dim):
        super().__init__(aggr='add')

        self.edge_mlp = MLP(
            [edge_dim + 2*hidden_dim, hidden_dim, hidden_dim],
            layer_norm=True
        )
        self.node_mlp = MLP(
            [2*hidden_dim, hidden_dim, hidden_dim],
            layer_norm=True
        )

    def forward(self, h, edge_index, edge_attr, n_elements):
        """
        Args:
            h: (N, H) node embeddings
            edge_index: (2, 2E) [fwd edges | bwd edges]
            edge_attr: (2E, D) edge features
            n_elements: E (number of undirected edges)
        """
        # E = n_elements
        # Ensure E is an integer
        if isinstance(n_elements, torch.Tensor):
            E = n_elements.item()
        else:
            E = n_elements
        
        # For batched graphs, use half of edge count
        if E * 2 != edge_index.shape[1]:
            E = edge_index.shape[1] // 2

        # Forward edges: first E
        ei_fwd = edge_index[:, :E]
        ea_fwd = edge_attr[:E]

        # Compute messages for forward edges
        src_fwd = h[ei_fwd[0]]  # sender features
        dst_fwd = h[ei_fwd[1]]  # receiver features
        msg_fwd = self.edge_mlp(
            torch.cat([ea_fwd, dst_fwd, src_fwd], dim=-1)
        )

        # Aggregate: forward messages to receivers
        N = h.shape[0]
        agg_pos = torch.zeros(N, msg_fwd.shape[1],
                              device=h.device)
        agg_pos.scatter_add_(
            0,
            ei_fwd[1].unsqueeze(1).expand_as(msg_fwd),
            msg_fwd
        )

        # Aggregate: NEGATIVE forward messages to senders
        # (momentum conservation: m_ji = -m_ij)
        agg_neg = torch.zeros(N, msg_fwd.shape[1],
                              device=h.device)
        agg_neg.scatter_add_(
            0,
            ei_fwd[0].unsqueeze(1).expand_as(msg_fwd),
            -msg_fwd
        )

        # Total aggregated messages
        agg_total = agg_pos + agg_neg

        # Node update
        h_new = self.node_mlp(
            torch.cat([h, agg_total], dim=-1)
        )

        # Residual connection
        h_out = h + h_new

        # Update edge features (residual)
        # For backward edges, negate the message
        edge_attr_new = edge_attr.clone()
        edge_attr_new[:E] = edge_attr[:E] + msg_fwd
        edge_attr_new[E:] = edge_attr[E:] - msg_fwd

        return h_out, edge_attr_new


# ════════════════════════════════════════════════
# PIGNN Model
# ════════════════════════════════════════════════

class PIGNN(nn.Module):
    """
    Physics-Informed GNN for 2D frame structures.
    
    Architecture (following Dalton et al.):
      Encoder:   node_enc(V) → h, edge_enc(E) → e
      Processor: K rounds of momentum-conserving MP
      Decoder:   separate MLP per output DOF (ux, uz, θ)
    
    Key features:
      - Zero-initialized decoder (start at u=0)
      - Momentum-conserving messages (m_ij = -m_ji)
      - Separate decoder per DOF
      - CELU activation throughout
      - Hard BC enforcement
    """

    OUT_DIM = 3  # [ux/u_c, uz/u_c, θ/θ_c]

    def __init__(self,
                 node_in_dim=10,
                 edge_in_dim=7,
                 hidden_dim=128,
                 n_layers=6):
        super().__init__()
        H = hidden_dim

        # ── Encoders ──
        self.node_encoder = MLP(
            [node_in_dim, H, H], layer_norm=True
        )
        self.edge_encoder = MLP(
            [edge_in_dim, H, H], layer_norm=True
        )

        # ── Processor (K message passing steps) ──
        self.mp_layers = nn.ModuleList([
            MomentumMessagePassing(H, H)
            for _ in range(n_layers)
        ])

        # ── Final aggregation + LayerNorm ──
        # After MP: concat node embedding with 
        # aggregated incoming messages
        self.final_norm = nn.LayerNorm(2 * H)

        # ── Separate decoder per DOF ──
        # Each outputs a single scalar
        self.decoder_ux = MLP([2*H, H, 1])
        self.decoder_uz = MLP([2*H, H, 1])
        self.decoder_th = MLP([2*H, H, 1])

        # ── Zero-initialize decoder last layers ──
        self._zero_init_decoders()

    def _zero_init_decoders(self):
        """
        Initialize decoder last-layer weights to zero.
        This ensures network starts at u=0.
        
        From the paper (utils.py, gen_zero_params_gnn):
          "zero out the weights in the last layer 
           of the decoder FCNNs"
        """
        with torch.no_grad():
            for decoder in [self.decoder_ux,
                           self.decoder_uz,
                           self.decoder_th]:
                last = decoder.layers[-1]
                last.weight.zero_()
                last.bias.zero_()

    def forward(self, data):
        """
        Forward pass: V, E → displacement prediction
        
        Returns: (N, 3) non-dimensional displacements
                 [ux/u_c, uz/u_c, θ/θ_c]
        """
        # N = data.x.shape[0]
        # E = data.n_elements

        N = data.x.shape[0]
    
        # Fix for batched data
        if hasattr(data, 'num_graphs') and data.num_graphs > 1:
            # Batched: n_elements is sum across all graphs
            E = data.edge_index.shape[1] // 2
        else:
            # Single graph
            E = data.n_elements if isinstance(data.n_elements, int) else data.n_elements.item()

        # ── Encode ──
        h = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)

        # ── Process (K rounds of message passing) ──
        for mp in self.mp_layers:
            h, e = mp(h, data.edge_index, e, E)

        # ── Aggregate final messages ──
        # Sum incoming edge embeddings to each node
        edge_index = data.edge_index
        incoming = torch.zeros(
            N, e.shape[1], device=h.device
        )
        incoming.scatter_add_(
            0,
            edge_index[1].unsqueeze(1).expand_as(e),
            e
        )

        # Concatenate node embedding with incoming
        z_local = self.final_norm(
            torch.cat([h, incoming], dim=-1)
        )

        # ── Decode (separate MLP per DOF) ──
        ux = self.decoder_ux(z_local)   # (N, 1)
        uz = self.decoder_uz(z_local)   # (N, 1)
        th = self.decoder_th(z_local)   # (N, 1)

        pred = torch.cat([ux, uz, th], dim=-1)  # (N, 3)

        # ── Hard boundary conditions ──
        pred = self._apply_hard_bc(pred, data)

        return pred

    def _apply_hard_bc(self, pred, data):
        """
        Enforce zero displacement at support nodes.
        Same as paper: U_final = U_pred * interior_mask
        """
        pred = pred.clone()
        disp_mask = (1.0 - data.bc_disp)   # (N, 1)
        rot_mask = (1.0 - data.bc_rot)      # (N, 1)
        pred[:, 0:2] *= disp_mask
        pred[:, 2:3] *= rot_mask
        return pred

    def count_params(self):
        return sum(
            p.numel() for p in self.parameters()
        )

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

        print(f"\n{'═'*55}")
        print(f"  PIGNN Model Summary (Energy-Based)")
        print(f"{'═'*55}")
        print(f"  Node input:    "
              f"{self.node_encoder.layers[0].in_features}")
        print(f"  Edge input:    "
              f"{self.edge_encoder.layers[0].in_features}")
        print(f"  Hidden dim:    "
              f"{self.node_encoder.layers[-1].out_features}")
        print(f"  MP layers:     {len(self.mp_layers)}")
        print(f"  Output:        {self.OUT_DIM} "
              f"(separate decoder per DOF)")
        print(f"  Decoder init:  ZERO "
              f"(start at u=0)")
        print(f"  Activation:    CELU")
        print(f"  Message type:  Momentum-conserving")
        print(f"  ─────────────────────────────────")
        print(f"  Encoder params:  {n_enc:>10,}")
        print(f"  Processor params:{n_proc:>10,}")
        print(f"  Decoder params:  {n_dec:>10,}")
        print(f"  Total params:    "
              f"{self.count_params():>10,}")
        print(f"{'═'*55}\n")


