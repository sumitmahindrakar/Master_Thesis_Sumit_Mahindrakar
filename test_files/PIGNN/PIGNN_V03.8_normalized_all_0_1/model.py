# """
# =================================================================
# model.py — PIGNN for 2D Frame
# =================================================================

# Network outputs ~O(1) non-dimensional values.
# Scaling to physical units happens in corotational module.
# =================================================================
# """

# import torch
# import torch.nn as nn
# from layers import MLP, GraphNetworkLayer


# class PIGNN(nn.Module):

#     OUT_DIM = 3

#     def __init__(self,
#                  node_in_dim=10,
#                  edge_in_dim=7,
#                  hidden_dim=128,
#                  n_layers=6):
#         super().__init__()
#         H = hidden_dim

#         self.node_encoder = MLP(node_in_dim, [H], H, act=True)
#         self.edge_encoder = MLP(edge_in_dim, [H], H, act=True)

#         self.mp_layers = nn.ModuleList([
#             GraphNetworkLayer(H, H, aggr='add')
#             for _ in range(n_layers)
#         ])

#         self.decoder = MLP(H, [H, 64], self.OUT_DIM, act=True)

#     def forward(self, data):
#         x = data.x.clone()

#         h = self.node_encoder(x)
#         e = self.edge_encoder(data.edge_attr)

#         for mp in self.mp_layers:
#             h = h + mp(h, data.edge_index, e)

#         pred = self.decoder(h)

#         pred = self._apply_hard_bc(pred, data)

#         return pred

#     def _apply_hard_bc(self, pred, data):
#         pred = pred.clone()
#         disp_mask = (1.0 - data.bc_disp)
#         rot_mask  = (1.0 - data.bc_rot)
#         pred[:, 0:2] *= disp_mask
#         pred[:, 2:3] *= rot_mask
#         return pred

#     def count_params(self):
#         return sum(p.numel() for p in self.parameters())

#     def summary(self):
#         print(f"\n{'═'*50}")
#         print(f"  PIGNN Model Summary")
#         print(f"{'═'*50}")
#         print(f"  Node input:   "
#               f"{self.node_encoder.layers[0].in_features}")
#         print(f"  Edge input:   "
#               f"{self.edge_encoder.layers[0].in_features}")
#         print(f"  Output:       {self.OUT_DIM}  "
#               f"[u/u_c, w/u_c, θ/θ_c]")
#         print(f"  Output scale: raw ~O(1), "
#               f"scaled in corotational")
#         print(f"  Parameters:   {self.count_params():,}")
#         print(f"{'═'*50}\n")

"""
=================================================================
model.py — PIGNN with non-zero initial bias
=================================================================
"""

import torch
import torch.nn as nn
from layers import MLP, GraphNetworkLayer


class PIGNN(nn.Module):

    OUT_DIM = 3

    def __init__(self,
                 node_in_dim=10,
                 edge_in_dim=7,
                 hidden_dim=128,
                 n_layers=6):
        super().__init__()
        H = hidden_dim

        self.node_encoder = MLP(
            node_in_dim, [H], H, act=True
        )
        self.edge_encoder = MLP(
            edge_in_dim, [H], H, act=True
        )

        self.mp_layers = nn.ModuleList([
            GraphNetworkLayer(H, H, aggr='add')
            for _ in range(n_layers)
        ])

        self.decoder = MLP(
            H, [H, 64], self.OUT_DIM, act=True
        )

        # Initialize decoder last layer bias
        # to push initial output away from zero
        self._init_decoder_bias()

    def _init_decoder_bias(self):
        """
        Set initial bias so network outputs ~0.1-0.5
        instead of ~0. This prevents the zero attractor.
        """
        with torch.no_grad():
            last_layer = self.decoder.layers[-1]
            # Small positive bias
            last_layer.bias.fill_(0.5)
            # last_layer.bias.uniform_(0.3, 0.7)
            # Also scale weights slightly larger
            last_layer.weight.mul_(5.0)

    def forward(self, data):
        x = data.x.clone()

        h = self.node_encoder(x)
        e = self.edge_encoder(data.edge_attr)

        for mp in self.mp_layers:
            h = h + mp(h, data.edge_index, e)

        pred = self.decoder(h)

        pred = self._apply_hard_bc(pred, data)

        return pred

    def _apply_hard_bc(self, pred, data):
        pred = pred.clone()
        disp_mask = (1.0 - data.bc_disp)
        rot_mask  = (1.0 - data.bc_rot)
        pred[:, 0:2] *= disp_mask
        pred[:, 2:3] *= rot_mask
        return pred

    def count_params(self):
        return sum(
            p.numel() for p in self.parameters()
        )

    def summary(self):
        print(f"\n{'═'*50}")
        print(f"  PIGNN Model Summary")
        print(f"{'═'*50}")
        print(f"  Node input:   "
              f"{self.node_encoder.layers[0].in_features}")
        print(f"  Edge input:   "
              f"{self.edge_encoder.layers[0].in_features}")
        print(f"  Output:       {self.OUT_DIM}")
        print(f"  Init bias:    0.1 (non-zero)")
        print(f"  Parameters:   {self.count_params():,}")
        print(f"{'═'*50}\n")