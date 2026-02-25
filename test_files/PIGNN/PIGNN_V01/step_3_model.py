"""
=================================================================
step_3_model.py — PIGNN MODEL ARCHITECTURE
=================================================================
This file contains THE NEURAL NETWORK:
  - MLP (building block)
  - ProcessorBlock (message passing layer)
  - FramePIGNN (complete model)

Architecture flow:
  Node features (N,9)  → Node Encoder MLP → h_node (N,128)
  Edge features (2E,11)→ Edge Encoder MLP → h_edge (2E,128)
       ↓
  6 rounds of Message Passing (ProcessorBlock):
       │
       ├─ Edge Update:  h_edge = MLP(h_src ‖ h_dst ‖ h_edge)
       │
       ├─ Aggregation:  agg_j = Σ h_edge for all edges → node j
       │
       └─ Node Update:  h_node = MLP(h_node ‖ agg)
       ↓
  Node Decoder MLP → node_pred (N,6)  [ux,uy,uz,rx,ry,rz]
  Elem Decoder MLP → elem_pred (E,7)  [Mx,My,Mz,Fx,Fy,Fz,dBM/dI]
=================================================================
"""
import os

os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\PIGNN\PIGNN_V01")
print(f"Working directory: {os.getcwd()}")

import torch
import torch.nn as nn
from typing import Tuple


# ================================================================
# BUILDING BLOCK: MLP
# ================================================================

class MLP(nn.Module):
    """
    Multi-Layer Perceptron.
    
    Linear → LayerNorm → SiLU → Linear → LayerNorm → SiLU
    
    If input_dim == output_dim: adds residual connection (skip)
    SiLU activation: smooth, works well for physics problems
    """

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dim: int = None, n_layers: int = 2):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = output_dim

        self.residual = (input_dim == output_dim)

        layers = []
        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.SiLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.residual:
            out = out + x  # Skip connection
        return out


# ================================================================
# MESSAGE PASSING: ProcessorBlock
# ================================================================

def scatter_sum(src, index, dim_size):
    """Sum src rows by index (pure PyTorch)."""
    H = src.shape[1]
    out = torch.zeros(dim_size, H, device=src.device, dtype=src.dtype)
    index_expanded = index.unsqueeze(1).expand_as(src)
    out.scatter_add_(0, index_expanded, src)
    return out


class ProcessorBlock(nn.Module):
    """
    ONE ROUND of message passing on the structural graph.
    
    This is THE CORE of the GNN. Each round:
    
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  STEP 1: EDGE UPDATE                                │
    │  ─────────────────                                  │
    │  For each directed edge (node_i → node_j):          │
    │                                                     │
    │    input = concat[ h_node_i,  ← source node state   │
    │                    h_node_j,  ← target node state   │
    │                    h_edge  ]  ← current edge state   │
    │                                                     │
    │    h_edge_new = MLP(input) + h_edge  ← residual     │
    │                                                     │
    │  This lets each beam/column "see" what's happening  │
    │  at both its endpoints.                             │
    │                                                     │
    ├─────────────────────────────────────────────────────┤
    │                                                     │
    │  STEP 2: AGGREGATION                                │
    │  ───────────────────                                │
    │  For each node j:                                   │
    │                                                     │
    │    agg_j = Σ h_edge_new  (all edges pointing TO j)  │
    │                                                     │
    │  This collects "messages" from all connected beams  │
    │  and columns. Like summing forces at a joint.       │
    │                                                     │
    ├─────────────────────────────────────────────────────┤
    │                                                     │
    │  STEP 3: NODE UPDATE                                │
    │  ───────────────────                                │
    │  For each node j:                                   │
    │                                                     │
    │    input = concat[ h_node_j,  ← current state       │
    │                    agg_j   ]  ← aggregated messages  │
    │                                                     │
    │    h_node_new = MLP(input) + h_node  ← residual     │
    │                                                     │
    │  Each joint updates based on what all its members   │
    │  are "telling" it.                                  │
    │                                                     │
    └─────────────────────────────────────────────────────┘
    
    After 6 rounds:
      - Node at floor 6 has information from floor 0 (6 hops)
      - Every node "knows" about loads, BCs, and properties
        everywhere in the structure
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        # Edge MLP: takes concatenated [h_src, h_dst, h_edge] → new h_edge
        #           input size = 3 × hidden_dim
        self.edge_mlp = MLP(
            input_dim=3 * hidden_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
        )

        # Node MLP: takes concatenated [h_node, aggregated] → new h_node
        #           input size = 2 × hidden_dim
        self.node_mlp = MLP(
            input_dim=2 * hidden_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
        )

        # Layer normalization for training stability
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_node, h_edge, edge_index):
        """
        Args:
            h_node:     (N, H) node hidden states
            h_edge:     (2E, H) edge hidden states
            edge_index: (2, 2E) [source_nodes; target_nodes]
        
        Returns:
            h_node_new: (N, H) updated node states
            h_edge_new: (2E, H) updated edge states
        """
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        N = h_node.shape[0]

        # ── Step 1: Edge Update ──
        h_src = h_node[src_idx]    # (2E, H) source node features
        h_dst = h_node[dst_idx]    # (2E, H) target node features

        edge_input = torch.cat([h_src, h_dst, h_edge], dim=1)  # (2E, 3H)
        h_edge_new = self.edge_norm(
            h_edge + self.edge_mlp(edge_input))  # (2E, H) + residual

        # ── Step 2: Aggregate edges → nodes ──
        agg = scatter_sum(h_edge_new, dst_idx, dim_size=N)  # (N, H)

        # ── Step 3: Node Update ──
        node_input = torch.cat([h_node, agg], dim=1)  # (N, 2H)
        h_node_new = self.node_norm(
            h_node + self.node_mlp(node_input))  # (N, H) + residual

        return h_node_new, h_edge_new


# ================================================================
# COMPLETE MODEL: FramePIGNN
# ================================================================

class FramePIGNN(nn.Module):
    """
    Physics-Informed Graph Neural Network for structural frames.
    
    COMPLETE FORWARD PASS:
    
    ┌──────────────────────────────────────────────────────┐
    │                                                      │
    │  INPUT:                                              │
    │    node features x     (N, 9)   per node             │
    │    edge features ea    (2E, 11) per directed edge    │
    │    edge_index          (2, 2E)  connectivity         │
    │                                                      │
    │  ENCODE:                                             │
    │    h_node = NodeEncoder(x)      (N, 9)  → (N, 128)  │
    │    h_edge = EdgeEncoder(ea)     (2E,11) → (2E,128)  │
    │                                                      │
    │  PROCESS (×6 rounds):                                │
    │    h_node, h_edge = ProcessorBlock(h_node, h_edge)   │
    │    ↑ each round:                                     │
    │      edge update → aggregate → node update           │
    │                                                      │
    │  DECODE:                                             │
    │    node_pred = NodeDecoder(h_node)  (N, 128) → (N,6) │
    │    h_elem = avg(h_fwd, h_bwd)      (2E,128) → (E,128)│
    │    elem_pred = ElemDecoder(h_elem)  (E, 128) → (E,7) │
    │                                                      │
    │  OUTPUT:                                             │
    │    node_pred (N, 6): [ux, uy, uz, rx, ry, rz]       │
    │    elem_pred (E, 7): [Mx,My,Mz, Fx,Fy,Fz, dBM/dI]  │
    │                                                      │
    └──────────────────────────────────────────────────────┘
    """

    def __init__(self,
                 node_in_dim: int = 9,
                 edge_in_dim: int = 11,
                 hidden_dim: int = 128,
                 n_processors: int = 6,
                 node_out_dim: int = 6,
                 elem_out_dim: int = 7):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_processors = n_processors
        self.node_out_dim = node_out_dim
        self.elem_out_dim = elem_out_dim

        # ── ENCODERS ──
        # Transform raw features to hidden dimension
        self.node_encoder = MLP(
            input_dim=node_in_dim,    # 9 node features
            output_dim=hidden_dim,     # 128
            hidden_dim=hidden_dim,
        )

        self.edge_encoder = MLP(
            input_dim=edge_in_dim,    # 11 edge features
            output_dim=hidden_dim,     # 128
            hidden_dim=hidden_dim,
        )

        # ── PROCESSOR BLOCKS (message passing layers) ──
        # 6 rounds = information travels 6 hops
        # Your frame has max 12 nodes between base and top
        # With bidirectional edges: 6 rounds covers full structure
        self.processors = nn.ModuleList([
            ProcessorBlock(hidden_dim)
            for _ in range(n_processors)
        ])

        # ── DECODERS ──
        # Node decoder: hidden state → physical predictions
        self.node_decoder = nn.Sequential(
            MLP(hidden_dim, hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, node_out_dim),
        )

        # Element decoder: hidden state → forces/moments/sensitivity
        self.elem_decoder = nn.Sequential(
            MLP(hidden_dim, hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, elem_out_dim),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _edges_to_elements(self, h_edge, n_elements):
        """
        Convert directed edges → structural elements.
        
        Each element has 2 directed edges (forward + backward).
        Average them to get element-level features.
        
        Edge ordering (from Step 2):
          [graph1_fwd(E), graph1_bwd(E), graph2_fwd(E), ...]
        """
        total_directed = h_edge.shape[0]
        E = n_elements
        n_graphs = total_directed // (2 * E)

        h = h_edge.view(n_graphs, 2 * E, -1)
        h_fwd = h[:, :E, :]
        h_bwd = h[:, E:, :]
        h_elem = (h_fwd + h_bwd) / 2.0

        return h_elem.reshape(n_graphs * E, -1)

    def forward(self, data):
        """
        Full forward pass.
        
        Args:
            data: PyG Data with .x, .edge_attr, .edge_index, .n_elements
        
        Returns:
            node_pred: (N, 6)  displacements + rotations
            elem_pred: (E, 7)  moments + forces + sensitivity
        """
        # ── 1. ENCODE ──
        h_node = self.node_encoder(data.x)          # (N, 128)
        h_edge = self.edge_encoder(data.edge_attr)   # (2E, 128)

        # ── 2. MESSAGE PASSING (6 rounds) ──
        for processor in self.processors:
            h_node, h_edge = processor(h_node, h_edge, data.edge_index)

        # ── 3. DECODE NODES ──
        node_pred = self.node_decoder(h_node)  # (N, 6)

        # ── 4. DECODE ELEMENTS ──
        n_elem = data.n_elements
        if isinstance(n_elem, torch.Tensor):
            n_elem = n_elem.item()
        h_elem = self._edges_to_elements(h_edge, n_elem)  # (E, 128)
        elem_pred = self.elem_decoder(h_elem)  # (E, 7)

        return node_pred, elem_pred

    def predict_components(self, data):
        """Forward pass with named outputs."""
        node_pred, elem_pred = self.forward(data)
        return {
            'displacement': node_pred[:, 0:3],
            'rotation':     node_pred[:, 3:6],
            'moment':       elem_pred[:, 0:3],
            'force':        elem_pred[:, 3:6],
            'sensitivity':  elem_pred[:, 6:7],
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


# ================================================================
# MODEL FACTORY
# ================================================================

def create_model(size: str = 'medium') -> FramePIGNN:
    """
    Create model with preset sizes.
    
    'small':  64 hidden, 4 processors  (~  50K params)
    'medium': 128 hidden, 6 processors (~ 350K params)
    'large':  256 hidden, 8 processors (~1.5M params)
    """
    configs = {
        'small':  {'hidden_dim': 64,  'n_processors': 4},
        'medium': {'hidden_dim': 128, 'n_processors': 6},
        'large':  {'hidden_dim': 256, 'n_processors': 8},
    }

    cfg = configs[size]
    model = FramePIGNN(
        node_in_dim=9,
        edge_in_dim=11,
        hidden_dim=cfg['hidden_dim'],
        n_processors=cfg['n_processors'],
    )

    print(f"  Created '{size}' model: "
          f"hidden={cfg['hidden_dim']}, "
          f"processors={cfg['n_processors']}, "
          f"params={model.count_parameters():,}")

    return model


# ================================================================
# MAIN — Test the model
# ================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  FramePIGNN Model Test")
    print("=" * 60)

    # Load test graph
    data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
    data = data_list[0]

    # Create model
    model = create_model('medium')

    # Print architecture
    print(f"\n  Architecture:")
    print(f"  {'─'*50}")
    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        print(f"  {name:<20} {n_params:>8,} params")
    print(f"  {'─'*50}")
    print(f"  {'TOTAL':<20} {model.count_parameters():>8,} params")

    # Test forward pass
    print(f"\n  Forward pass:")
    model.eval()
    with torch.no_grad():
        node_pred, elem_pred = model(data)
    print(f"    node_pred: {node_pred.shape}  "
          f"[ux,uy,uz,rx,ry,rz]")
    print(f"    elem_pred: {elem_pred.shape}  "
          f"[Mx,My,Mz,Fx,Fy,Fz,dBM/dI]")

    print(f"\n  Message passing detail:")
    print(f"    6 rounds × ProcessorBlock:")
    print(f"      Edge update: MLP(384→128) + residual")
    print(f"      Aggregation: Σ edges → nodes")
    print(f"      Node update: MLP(256→128) + residual")
    print(f"    = information travels 6 hops through frame")

    print(f"\n  ✓ Model ready for physics-informed training")