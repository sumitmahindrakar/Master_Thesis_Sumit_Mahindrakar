import torch
import torch.nn as nn
from layers import MLP, GraphNetworkLayer

class PIGNN(nn.Module):

    OUT_DIM = 3       # ◀◀◀ WAS 15

    def __init__(self,
                 node_in_dim=9,
                 edge_in_dim=10,
                 hidden_dim=128,
                 n_layers=6):
        super().__init__()
        H = hidden_dim

        self.node_encoder = MLP(node_in_dim, [H], H, act=True)
        self.edge_encoder = MLP(edge_in_dim, [H], H, act=True)

        self.mp_layers = nn.ModuleList([
            GraphNetworkLayer(H, H, aggr='add')
            for _ in range(n_layers)
        ])

        self.decoder = MLP(H, [H, 64], self.OUT_DIM, act=True)   # ◀◀◀ 3 outputs

    def forward(self, data):
        x = data.x.clone()
        h = self.node_encoder(x)
        e = self.edge_encoder(data.edge_attr)

        for mp in self.mp_layers:
            h = h + mp(h, data.edge_index, e)

        raw = self.decoder(h)              # (N, 3) all ~O(1)

        # ── Output scaling ──
        pred = raw.clone()
        pred[:, 0] = raw[:, 0] * data.u_c           # ux in meters
        pred[:, 1] = raw[:, 1] * data.u_c           # uz in meters
        pred[:, 2] = raw[:, 2] * data.theta_c       # θy in radians

        # ── Hard BC ──
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
        return sum(p.numel() for p in self.parameters())

    def summary(self):
        print(f"\n{'='*50}")
        print(f"  PIGNN (3-output, shape function)")
        print(f"{'='*50}")
        print(f"  Output: {self.OUT_DIM} per node [ux, uz, θy]")
        print(f"  Forces: derived via shape functions")
        print(f"  Parameters: {self.count_params():,}")
        print(f"{'='*50}\n")