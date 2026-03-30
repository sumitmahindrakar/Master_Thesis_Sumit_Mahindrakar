"""
=================================================================
plot_results.py — PIGNN Frame Visualization v4
=================================================================
Frame is in XZ plane:
  - Plot horizontal = coords[:, 0] (X)
  - Plot vertical   = coords[:, 2] (Z)
  - Deformation: ux → horizontal, uy → vertical
=================================================================
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)

from model import PIGNN
from energy_loss import FrameEnergyLoss
from step_2_grapg_constr import FrameData


# ════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════

class PlotConfig:
    checkpoint_path = "RESULTS/checkpoint.pt"
    norm_data_path  = "DATA/graph_dataset_norm.pt"
    raw_data_path   = "DATA/graph_dataset.pt"
    save_dir        = "PLOTS"

    hidden_dim      = 64
    n_layers        = 10
    node_in_dim     = 10
    edge_in_dim     = 7

    cases_to_plot   = [2, 7, 12]
    deform_scale    = None  # None = auto-scale
    dpi             = 150
    train_ratio     = 0.875

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ════════════════════════════════════════════════
# DATA LOADER
# ════════════════════════════════════════════════

class ResultsLoader:

    def __init__(self, cfg: PlotConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        os.makedirs(cfg.save_dir, exist_ok=True)

    def load_all(self):
        print("── Loading data and model ──")

        self.norm_data = torch.load(
            self.cfg.norm_data_path, weights_only=False
        )
        self.norm_data = [d.cpu() for d in self.norm_data]

        self.raw_data = torch.load(
            self.cfg.raw_data_path, weights_only=False
        )
        self.raw_data = [d.cpu() for d in self.raw_data]

        from normalizer import PhysicsScaler
        if not hasattr(self.norm_data[0], 'F_c'):
            self.norm_data = (
                PhysicsScaler.compute_and_store_list(
                    self.norm_data)
            )
        if not hasattr(self.raw_data[0], 'u_c'):
            self.raw_data = (
                PhysicsScaler.compute_and_store_list(
                    self.raw_data)
            )

        self.model = PIGNN(
            node_in_dim=self.cfg.node_in_dim,
            edge_in_dim=self.cfg.edge_in_dim,
            hidden_dim=self.cfg.hidden_dim,
            n_layers=self.cfg.n_layers,
        ).to(self.device)

        ckpt = torch.load(
            self.cfg.checkpoint_path,
            weights_only=False,
            map_location=self.device
        )
        # self.model.load_state_dict(ckpt['model_state'])
        # self.model.eval()
        # print(f"  Model: {self.cfg.checkpoint_path} "
        #       f"(epoch {ckpt['epoch']})")
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            self.model.load_state_dict(ckpt['model_state'])
            epoch = ckpt.get('epoch', 'N/A')
        else:
            self.model.load_state_dict(ckpt)
            epoch = 'N/A'

        self.model.eval()

        print(f"  Model: {self.cfg.checkpoint_path} (epoch {epoch})")

        self.predictions = []
        with torch.no_grad():
            for ndata in self.norm_data:
                nd = ndata.clone().to(self.device)
                pr = self.model(nd)
                pp = torch.zeros_like(pr)
                pp[:, 0] = pr[:, 0] * nd.u_c
                pp[:, 1] = pr[:, 1] * nd.u_c
                pp[:, 2] = pr[:, 2] * nd.theta_c
                self.predictions.append(pp.cpu())
        print(f"  Predictions: {len(self.predictions)}")

        # ── Verify coordinate plane ──
        c = self.raw_data[0].coords.numpy()
        x_range = c[:, 0].max() - c[:, 0].min()
        y_range = c[:, 1].max() - c[:, 1].min()
        z_range = c[:, 2].max() - c[:, 2].min()
        print(f"\n  Coordinate ranges:")
        print(f"    X: {x_range:.4f}")
        print(f"    Y: {y_range:.4f}")
        print(f"    Z: {z_range:.4f}")

        # Determine which 2 axes to use
        ranges = [x_range, y_range, z_range]
        sorted_axes = np.argsort(ranges)[::-1]
        # Two largest ranges are the plotting axes
        self.h_axis = sorted_axes[0]  # horizontal
        self.v_axis = sorted_axes[1]  # vertical

        # If two axes have same range, prefer X,Z
        if y_range < 1e-6:
            self.h_axis = 0  # X
            self.v_axis = 2  # Z
        elif z_range < 1e-6:
            self.h_axis = 0  # X
            self.v_axis = 1  # Y
        elif x_range < 1e-6:
            self.h_axis = 1  # Y
            self.v_axis = 2  # Z

        axis_names = ['X', 'Y', 'Z']
        print(f"    → Plotting: {axis_names[self.h_axis]}"
              f" (horizontal) vs "
              f"{axis_names[self.v_axis]} (vertical)")

        return self

    def get_node_xy(self, case_idx):
        """
        Return [N, 2] plotting coordinates:
          col 0 = horizontal axis
          col 1 = vertical axis
        """
        c = self.raw_data[case_idx].coords.numpy()
        return np.column_stack([
            c[:, self.h_axis],
            c[:, self.v_axis]
        ])

    def get_elements(self, case_idx):
        conn = self.raw_data[case_idx].connectivity.numpy()
        return [(conn[e, 0], conn[e, 1])
                for e in range(conn.shape[0])]

    def get_supports(self, case_idx):
        bc = self.raw_data[case_idx].bc_disp.squeeze()
        return torch.where(bc.abs() > 0)[0].tolist()

    def get_loaded_nodes(self, case_idx):
        F = self.raw_data[case_idx].F_ext
        mag = F.pow(2).sum(dim=1).sqrt()
        return torch.where(mag > 1e-10)[0].tolist()


# ════════════════════════════════════════════════
# FRAME PLOTTER
# ════════════════════════════════════════════════

class FramePlotter:

    def __init__(self, loader, cfg):
        self.loader = loader
        self.cfg = cfg

    def _auto_scale(self, case_idx):
        """Compute deformation scale so deformed shape
        is visible (~10% of structure span)."""
        xy = self.loader.get_node_xy(case_idx)
        span = max(
            xy[:, 0].max() - xy[:, 0].min(),
            xy[:, 1].max() - xy[:, 1].min(),
            1e-6
        )
        rd = self.loader.raw_data[case_idx]
        max_d = max(
            rd.y_node[:, :2].abs().max().item(), 1e-12
        )
        return 0.10 * span / max_d

    def plot_frame_structure(self, case_idx, ax=None,
                              show_labels=True,
                              show_loads=True):
        """Plot undeformed frame with supports and loads."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        xy = self.loader.get_node_xy(case_idx)
        elems = self.loader.get_elements(case_idx)
        supports = self.loader.get_supports(case_idx)
        loaded = self.loader.get_loaded_nodes(case_idx)
        rd = self.loader.raw_data[case_idx]

        # ── Elements ──
        for (n1, n2) in elems:
            ax.plot(
                [xy[n1, 0], xy[n2, 0]],
                [xy[n1, 1], xy[n2, 1]],
                'k-', linewidth=2.5, zorder=1
            )

        # ── All nodes ──
        ax.scatter(xy[:, 0], xy[:, 1],
                   c='steelblue', s=50, zorder=3,
                   edgecolors='black', linewidth=1)

        # ── Supports ──
        if supports:
            sc = xy[supports]
            ax.scatter(sc[:, 0], sc[:, 1],
                       c='red', s=250, marker='^',
                       zorder=5, edgecolors='darkred',
                       linewidth=1.5, label='Fixed')

        # ── External loads ──
        if show_loads and loaded:
            F = rd.F_ext.numpy()
            fmax = np.abs(F[:, :2]).max()
            if fmax > 1e-10:
                span = max(
                    xy[:, 0].max() - xy[:, 0].min(),
                    xy[:, 1].max() - xy[:, 1].min()
                )
                arrow_s = 0.10 * span / fmax

                for n in loaded:
                    fx = F[n, 0]
                    fy = F[n, 1]
                    if abs(fx) > 1e-10 or abs(fy) > 1e-10:
                        ax.annotate(
                            '',
                            xy=(xy[n, 0], xy[n, 1]),
                            xytext=(
                                xy[n, 0] - fx * arrow_s,
                                xy[n, 1] - fy * arrow_s
                            ),
                            arrowprops=dict(
                                arrowstyle='->'
                                ',head_width=0.3'
                                ',head_length=0.15',
                                color='green', lw=2
                            ),
                            zorder=6
                        )

        # ── Node labels ──
        if show_labels:
            for n in range(xy.shape[0]):
                ax.annotate(
                    f'{n}', (xy[n, 0], xy[n, 1]),
                    textcoords="offset points",
                    xytext=(5, 5), fontsize=6,
                    fontweight='bold', color='navy'
                )

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X [m]', fontsize=11)
        ax.set_ylabel('Z [m]', fontsize=11)
        ax.set_title(
            f'Case {case_idx}: Frame Structure',
            fontsize=12, fontweight='bold'
        )
        if supports:
            ax.legend(fontsize=9, loc='best')
        return ax

    def plot_deformed_comparison(self, case_idx,
                                  scale=None, ax=None):
        """Undeformed + Kratos + PIGNN deformed shapes."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        if scale is None:
            scale = (self.cfg.deform_scale
                     or self._auto_scale(case_idx))

        xy = self.loader.get_node_xy(case_idx)
        elems = self.loader.get_elements(case_idx)
        supports = self.loader.get_supports(case_idx)
        rd = self.loader.raw_data[case_idx]

        kratos = rd.y_node.numpy()       # [N, 3]
        pignn = self.loader.predictions[case_idx].numpy()

        # ── Deformed coordinates ──
        # ux → horizontal shift,  uy → vertical shift
        kratos_def = xy.copy()
        kratos_def[:, 0] += kratos[:, 0] * scale
        kratos_def[:, 1] += kratos[:, 1] * scale

        pignn_def = xy.copy()
        pignn_def[:, 0] += pignn[:, 0] * scale
        pignn_def[:, 1] += pignn[:, 1] * scale

        # ── Undeformed ──
        for (n1, n2) in elems:
            ax.plot(
                [xy[n1, 0], xy[n2, 0]],
                [xy[n1, 1], xy[n2, 1]],
                '-', color='#AAAAAA', linewidth=2.5,
                zorder=1
            )
        ax.scatter(xy[:, 0], xy[:, 1],
                   c='gray', s=25, zorder=2, alpha=0.5)

        # ── Kratos deformed ──
        for (n1, n2) in elems:
            ax.plot(
                [kratos_def[n1, 0], kratos_def[n2, 0]],
                [kratos_def[n1, 1], kratos_def[n2, 1]],
                'b-', linewidth=2.5, zorder=3
            )
        ax.scatter(kratos_def[:, 0], kratos_def[:, 1],
                   c='blue', s=40, zorder=4,
                   edgecolors='darkblue', linewidth=0.8)

        # ── PIGNN deformed ──
        for (n1, n2) in elems:
            ax.plot(
                [pignn_def[n1, 0], pignn_def[n2, 0]],
                [pignn_def[n1, 1], pignn_def[n2, 1]],
                'r--', linewidth=2.5, zorder=5
            )
        ax.scatter(pignn_def[:, 0], pignn_def[:, 1],
                   c='red', s=40, zorder=6,
                   edgecolors='darkred', linewidth=0.8)

        # ── Supports ──
        if supports:
            sc = xy[supports]
            ax.scatter(sc[:, 0], sc[:, 1],
                       c='green', s=250, marker='^',
                       zorder=7, edgecolors='darkgreen',
                       linewidth=1.5)

        # ── Error annotation ──
        err = np.sqrt(np.mean(
            (pignn[:, :2] - kratos[:, :2]) ** 2
        ))
        ref = np.sqrt(np.mean(kratos[:, :2] ** 2))
        rel = err / max(ref, 1e-12) * 100

        ax.text(
            0.02, 0.98,
            f'Rel. error: {rel:.2f}%\n'
            f'Scale: ×{scale:.0f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round',
                      facecolor='wheat', alpha=0.9)
        )

        # ── Legend ──
        handles = [
            mpatches.Patch(color='#AAAAAA',
                           label='Undeformed'),
            mpatches.Patch(color='blue',
                           label=f'Kratos (×{scale:.0f})'),
            mpatches.Patch(color='red',
                           label=f'PIGNN (×{scale:.0f})'),
            mpatches.Patch(color='green',
                           label='Support'),
        ]
        ax.legend(handles=handles, fontsize=9,
                  loc='lower right', framealpha=0.9)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X [m]', fontsize=11)
        ax.set_ylabel('Z [m]', fontsize=11)
        ax.set_title(
            f'Case {case_idx}: Deformed Shape',
            fontsize=12, fontweight='bold'
        )
        return ax

    def plot_single_case_detail(self, case_idx):
        """4-panel detailed view for one case."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle(
            f'Case {case_idx} — Detailed Analysis',
            fontsize=15, fontweight='bold'
        )

        # (0,0) Undeformed structure
        self.plot_frame_structure(case_idx, ax=axes[0, 0])

        # (0,1) Deformed comparison
        self.plot_deformed_comparison(
            case_idx, ax=axes[0, 1]
        )

        # (1,0) Displacement bar chart
        rd = self.loader.raw_data[case_idx]
        k = rd.y_node.numpy()
        p = self.loader.predictions[case_idx].numpy()
        n = k.shape[0]
        ids = np.arange(n)
        w = 0.35

        ax = axes[1, 0]
        ax.bar(ids - w/2, k[:, 0] * 1e3, w,
               label='Kratos', color='#1565C0', alpha=0.85)
        ax.bar(ids + w/2, p[:, 0] * 1e3, w,
               label='PIGNN', color='#E53935', alpha=0.85)
        ax.set_xlabel('Node ID')
        ax.set_ylabel('u_x [mm]')
        ax.set_title('Horizontal Displacement u_x')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(ids[::2])

        # (1,1) Per-node absolute error
        ax = axes[1, 1]
        err_ux = np.abs(p[:, 0] - k[:, 0]) * 1e3
        err_uy = np.abs(p[:, 1] - k[:, 1]) * 1e3
        err_th = np.abs(p[:, 2] - k[:, 2]) * 1e3

        ax.bar(ids - 0.25, err_ux, 0.25,
               label='|Δu_x| [mm]', color='#2196F3',
               alpha=0.8)
        ax.bar(ids, err_uy, 0.25,
               label='|Δu_z| [mm]', color='#4CAF50',
               alpha=0.8)
        ax.bar(ids + 0.25, err_th, 0.25,
               label='|Δθ| [mrad]', color='#FF9800',
               alpha=0.8)
        ax.set_xlabel('Node ID')
        ax.set_ylabel('Abs Error')
        ax.set_title('Per-Node Absolute Error')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(ids[::2])

        plt.tight_layout()
        path = os.path.join(
            self.cfg.save_dir,
            f'detail_case{case_idx}.png'
        )
        plt.savefig(path, dpi=self.cfg.dpi,
                    bbox_inches='tight')
        plt.show()
        print(f"  Saved: {path}")

    def plot_single_deformed(self, case_idx, scale=None,
                              save=True):
        """
        Large, clean single-case deformed shape plot.
        No cluttered text overlays.
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))

        if scale is None:
            scale = (self.cfg.deform_scale
                     or self._auto_scale(case_idx))

        xy = self.loader.get_node_xy(case_idx)
        elems = self.loader.get_elements(case_idx)
        supports = self.loader.get_supports(case_idx)
        rd = self.loader.raw_data[case_idx]
        n_train = int(
            len(self.loader.raw_data) * self.cfg.train_ratio
        )
        tag = 'TRAIN' if case_idx < n_train else 'TEST'

        kratos = rd.y_node.numpy()
        pignn = self.loader.predictions[case_idx].numpy()

        # Deformed coordinates
        kratos_def = xy.copy()
        kratos_def[:, 0] += kratos[:, 0] * scale
        kratos_def[:, 1] += kratos[:, 1] * scale

        pignn_def = xy.copy()
        pignn_def[:, 0] += pignn[:, 0] * scale
        pignn_def[:, 1] += pignn[:, 1] * scale

        # ── Undeformed ──
        for (n1, n2) in elems:
            ax.plot(
                [xy[n1, 0], xy[n2, 0]],
                [xy[n1, 1], xy[n2, 1]],
                '-', color='#CCCCCC', linewidth=3,
                zorder=1, solid_capstyle='round'
            )

        # ── Kratos deformed ──
        for i, (n1, n2) in enumerate(elems):
            label = 'Kratos' if i == 0 else None
            ax.plot(
                [kratos_def[n1, 0], kratos_def[n2, 0]],
                [kratos_def[n1, 1], kratos_def[n2, 1]],
                'b-', linewidth=3, zorder=3,
                solid_capstyle='round', label=label
            )

        # ── PIGNN deformed ──
        for i, (n1, n2) in enumerate(elems):
            label = 'PIGNN' if i == 0 else None
            ax.plot(
                [pignn_def[n1, 0], pignn_def[n2, 0]],
                [pignn_def[n1, 1], pignn_def[n2, 1]],
                'r--', linewidth=3, zorder=5,
                solid_capstyle='round', label=label
            )

        # ── Nodes (small, unobtrusive) ──
        ax.scatter(
            kratos_def[:, 0], kratos_def[:, 1],
            c='blue', s=25, zorder=4,
            edgecolors='darkblue', linewidth=0.5
        )
        ax.scatter(
            pignn_def[:, 0], pignn_def[:, 1],
            c='red', s=25, zorder=6,
            edgecolors='darkred', linewidth=0.5
        )

        # ── Supports ──
        if supports:
            sc = xy[supports]
            ax.scatter(
                sc[:, 0], sc[:, 1],
                c='green', s=200, marker='^',
                zorder=8, edgecolors='darkgreen',
                linewidth=1.5, label='Support'
            )

        # ── Error metrics (small, bottom-left) ──
        err_ux = np.sqrt(np.mean(
            (pignn[:, 0] - kratos[:, 0]) ** 2))
        err_uz = np.sqrt(np.mean(
            (pignn[:, 1] - kratos[:, 1]) ** 2))
        err_th = np.sqrt(np.mean(
            (pignn[:, 2] - kratos[:, 2]) ** 2))

        ref_ux = np.abs(kratos[:, 0]).max()
        ref_uz = np.abs(kratos[:, 1]).max()
        ref_th = np.abs(kratos[:, 2]).max()

        rel_ux = err_ux / max(ref_ux, 1e-12) * 100
        rel_uz = err_uz / max(ref_uz, 1e-12) * 100
        rel_th = err_th / max(ref_th, 1e-12) * 100

        # ── Clean legend (outside plot) ──
        ax.legend(
            fontsize=12, loc='best',
            framealpha=0.95, edgecolor='gray',
            fancybox=True
        )

        # ── Title with key info ──
        ax.set_title(
            f'Case {case_idx} [{tag}] — '
            f'Deformed Shape (×{scale:.0f})\n'
            f'Error:  u_x = {rel_ux:.2f}%,  '
            f'u_z = {rel_uz:.2f}%,  '
            f'θ = {rel_th:.2f}%',
            fontsize=14, fontweight='bold', pad=15
        )

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2, linestyle=':')
        ax.set_xlabel('X [m]', fontsize=13)
        ax.set_ylabel('Z [m]', fontsize=13)
        ax.tick_params(labelsize=11)

        # ── Add some padding around the structure ──
        all_x = np.concatenate([
            xy[:, 0], kratos_def[:, 0], pignn_def[:, 0]
        ])
        all_y = np.concatenate([
            xy[:, 1], kratos_def[:, 1], pignn_def[:, 1]
        ])
        x_margin = (all_x.max() - all_x.min()) * 0.08
        y_margin = (all_y.max() - all_y.min()) * 0.08
        ax.set_xlim(all_x.min() - x_margin,
                    all_x.max() + x_margin)
        ax.set_ylim(all_y.min() - y_margin,
                    all_y.max() + y_margin)

        plt.tight_layout()

        if save:
            path = os.path.join(
                self.cfg.save_dir,
                f'deformed_case{case_idx}.png'
            )
            plt.savefig(path, dpi=self.cfg.dpi,
                        bbox_inches='tight')
            print(f"  Saved: {path}")

        plt.show()
        return ax


# ════════════════════════════════════════════════
# DISPLACEMENT SCATTER
# ════════════════════════════════════════════════

class DisplacementPlotter:

    def __init__(self, loader, cfg):
        self.loader = loader
        self.cfg = cfg

    def plot_displacement_scatter(self, case_indices=None):
        if case_indices is None:
            case_indices = list(range(
                len(self.loader.raw_data)))

        n_train = int(
            len(self.loader.raw_data) * self.cfg.train_ratio
        )

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            'PIGNN vs Kratos — Train (blue) / Test (red)',
            fontsize=14, fontweight='bold'
        )
        dof_names = ['u_x [m]', 'u_z [m]', 'θ [rad]']

        for dof in range(3):
            ax = axes[dof]
            k_tr, p_tr = [], []
            k_te, p_te = [], []

            for ci in case_indices:
                kv = self.loader.raw_data[ci] \
                    .y_node[:, dof].numpy()
                pv = self.loader.predictions[ci] \
                    [:, dof].numpy()
                if ci < n_train:
                    k_tr.extend(kv); p_tr.extend(pv)
                else:
                    k_te.extend(kv); p_te.extend(pv)

            k_tr = np.array(k_tr); p_tr = np.array(p_tr)
            k_te = np.array(k_te); p_te = np.array(p_te)

            ax.scatter(k_tr, p_tr, c='#2196F3', s=10,
                       alpha=0.3, label='Train')
            if len(k_te) > 0:
                ax.scatter(k_te, p_te, c='#E53935', s=25,
                           alpha=0.7, label='Test',
                           edgecolors='darkred',
                           linewidth=0.3)

            all_k = np.concatenate([k_tr, k_te]) \
                if len(k_te) > 0 else k_tr
            all_p = np.concatenate([p_tr, p_te]) \
                if len(p_te) > 0 else p_tr
            vmin = min(all_k.min(), all_p.min())
            vmax = max(all_k.max(), all_p.max())
            m = 0.1 * (vmax - vmin) \
                if vmax > vmin else 0.1
            lims = [vmin - m, vmax + m]
            ax.plot(lims, lims, 'k--', lw=1.5, alpha=0.7)

            # Metrics on test set
            if len(k_te) > 1:
                ss_r = np.sum((p_te - k_te) ** 2)
                ss_t = np.sum(
                    (k_te - k_te.mean()) ** 2)
                r2 = 1 - ss_r / max(ss_t, 1e-30)
                rmse = np.sqrt(np.mean(
                    (p_te - k_te) ** 2))
                title = (f'{dof_names[dof]}\n'
                         f'Test R²={r2:.4f}, '
                         f'RMSE={rmse:.2e}')
            else:
                ss_r = np.sum((p_tr - k_tr) ** 2)
                ss_t = np.sum(
                    (k_tr - k_tr.mean()) ** 2)
                r2 = 1 - ss_r / max(ss_t, 1e-30)
                title = (f'{dof_names[dof]}\n'
                         f'R²={r2:.4f}')

            ax.set_xlabel(f'Kratos')
            ax.set_ylabel(f'PIGNN')
            ax.set_title(title, fontsize=11)
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_aspect('equal')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.cfg.save_dir,
                            'displacement_scatter.png')
        plt.savefig(path, dpi=self.cfg.dpi,
                    bbox_inches='tight')
        plt.show()
        print(f"  Saved: {path}")

    def plot_error_heatmap(self, case_indices=None):
        if case_indices is None:
            case_indices = list(range(
                min(20, len(self.loader.raw_data))))

        n_train = int(
            len(self.loader.raw_data) * self.cfg.train_ratio
        )
        errors = []
        for ci in case_indices:
            k = self.loader.raw_data[ci].y_node.numpy()
            p = self.loader.predictions[ci].numpy()
            row = []
            for dof in range(3):
                ref = np.abs(k[:, dof]).max()
                if ref > 1e-12:
                    rmse = np.sqrt(np.mean(
                        (p[:, dof] - k[:, dof]) ** 2))
                    row.append(rmse / ref * 100)
                else:
                    row.append(0.0)
            errors.append(row)
        errors = np.array(errors)

        fig, ax = plt.subplots(
            figsize=(8, max(6, len(case_indices)*0.4)))
        im = ax.imshow(errors, cmap='RdYlGn_r',
                       aspect='auto', vmin=0,
                       vmax=min(errors.max()*1.1, 50))
        ax.set_xticks(range(3))
        ax.set_xticklabels(['u_x', 'u_z', 'θ'])
        ax.set_yticks(range(len(case_indices)))
        labels = [
            f'{ci} [{"TR" if ci < n_train else "TE"}]'
            for ci in case_indices
        ]
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title('Relative Error [%]',
                      fontweight='bold')

        for i in range(len(case_indices)):
            for j in range(3):
                v = errors[i, j]
                clr = 'white' if v > 25 else 'black'
                ax.text(j, i, f'{v:.1f}%',
                        ha='center', va='center',
                        fontsize=8, color=clr,
                        fontweight='bold')

        plt.colorbar(im, ax=ax, label='Error [%]')
        plt.tight_layout()
        path = os.path.join(self.cfg.save_dir,
                            'error_heatmap.png')
        plt.savefig(path, dpi=self.cfg.dpi,
                    bbox_inches='tight')
        plt.show()
        print(f"  Saved: {path}")


# ════════════════════════════════════════════════
# ENERGY PLOTS
# ════════════════════════════════════════════════

class EnergyPlotter:

    def __init__(self, loader, cfg):
        self.loader = loader
        self.cfg = cfg
        self.loss_fn = FrameEnergyLoss()

    def plot_energy_comparison(self, case_indices):
        kU, kW, kPi = [], [], []
        pU, pW, pPi = [], [], []
        for ci in case_indices:
            rd = self.loader.raw_data[ci]
            Uk = self.loss_fn._strain_energy(
                rd.y_node, rd).item()
            Wk = self.loss_fn._external_work(
                rd.y_node, rd).item()
            kU.append(Uk); kW.append(Wk)
            kPi.append(Uk - Wk)

            Up = self.loss_fn._strain_energy(
                self.loader.predictions[ci], rd).item()
            Wp = self.loss_fn._external_work(
                self.loader.predictions[ci], rd).item()
            pU.append(Up); pW.append(Wp)
            pPi.append(Up - Wp)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Energy: PIGNN vs Kratos',
                     fontsize=14, fontweight='bold')
        x = np.arange(len(case_indices))
        w = 0.35

        for idx, (title, kv, pv) in enumerate([
            ('Strain Energy U', kU, pU),
            ('External Work W', kW, pW),
            ('Potential Π', kPi, pPi),
        ]):
            ax = axes[idx]
            ax.bar(x - w/2, kv, w, label='Kratos',
                   color='#1565C0', alpha=0.85)
            ax.bar(x + w/2, pv, w, label='PIGNN',
                   color='#E53935', alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(case_indices)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        path = os.path.join(self.cfg.save_dir,
                            'energy_comparison.png')
        plt.savefig(path, dpi=self.cfg.dpi,
                    bbox_inches='tight')
        plt.show()
        print(f"  Saved: {path}")

    def plot_energy_scatter(self):
        cases = list(range(len(self.loader.raw_data)))
        kPi, pPi = [], []
        for ci in cases:
            rd = self.loader.raw_data[ci]
            Uk = self.loss_fn._strain_energy(
                rd.y_node, rd).item()
            Wk = self.loss_fn._external_work(
                rd.y_node, rd).item()
            kPi.append(Uk - Wk)
            Up = self.loss_fn._strain_energy(
                self.loader.predictions[ci], rd).item()
            Wp = self.loss_fn._external_work(
                self.loader.predictions[ci], rd).item()
            pPi.append(Up - Wp)

        kPi = np.array(kPi); pPi = np.array(pPi)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(kPi, pPi, c='#E53935', s=40,
                   alpha=0.6, edgecolors='darkred')
        vmin = min(kPi.min(), pPi.min())
        vmax = max(kPi.max(), pPi.max())
        m = 0.1 * (vmax - vmin)
        lims = [vmin - m, vmax + m]
        ax.plot(lims, lims, 'k--', lw=1.5, label='Perfect')

        ss_r = np.sum((pPi - kPi) ** 2)
        ss_t = np.sum((kPi - kPi.mean()) ** 2)
        r2 = 1 - ss_r / max(ss_t, 1e-30)

        ax.set_xlabel('Kratos Π', fontsize=12)
        ax.set_ylabel('PIGNN Π', fontsize=12)
        ax.set_title(f'Potential Energy\nR²={r2:.4f}',
                     fontweight='bold')
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_aspect('equal')
        ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.cfg.save_dir,
                            'energy_scatter.png')
        plt.savefig(path, dpi=self.cfg.dpi,
                    bbox_inches='tight')
        plt.show()
        print(f"  Saved: {path}")


# ════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════

def print_summary(loader, cfg):
    n_train = int(
        len(loader.raw_data) * cfg.train_ratio)
    n = len(loader.raw_data)

    print(f"\n{'='*80}")
    print(f"  {'Case':>5} {'Set':>4} | {'Rel_ux':>9} | "
          f"{'Rel_uy':>9} | {'Rel_θ':>8}")
    print(f"  {'-'*50}")

    tr = {0: [], 1: [], 2: []}
    te = {0: [], 1: [], 2: []}

    for ci in range(n):
        k = loader.raw_data[ci].y_node.numpy()
        p = loader.predictions[ci].numpy()
        tag = 'TR' if ci < n_train else 'TE'
        rmse = np.sqrt(np.mean((p - k)**2, axis=0))
        ref = np.abs(k).max(axis=0)
        rel = np.where(ref > 1e-12,
                       rmse / ref * 100, 0)
        d = tr if ci < n_train else te
        for dof in range(3):
            d[dof].append(rel[dof])
        print(f"  {ci:5d} [{tag}] | "
              f"{rel[0]:8.2f}% | "
              f"{rel[1]:8.2f}% | "
              f"{rel[2]:7.2f}%")

    print(f"  {'-'*50}")
    if tr[0]:
        print(f"  TRAIN AVG   | "
              f"{np.mean(tr[0]):8.2f}% | "
              f"{np.mean(tr[1]):8.2f}% | "
              f"{np.mean(tr[2]):7.2f}%")
    if te[0]:
        print(f"  TEST  AVG   | "
              f"{np.mean(te[0]):8.2f}% | "
              f"{np.mean(te[1]):8.2f}% | "
              f"{np.mean(te[2]):7.2f}%")
    print(f"{'='*80}")


# ════════════════════════════════════════════════
# MULTI-CASE PLOT
# ════════════════════════════════════════════════

def plot_multi(loader, cfg, cases):
    n = len(cases)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(7*cols, 6*rows))
    fig.suptitle(
        'Deformed Shape: PIGNN (red) vs Kratos (blue)',
        fontsize=16, fontweight='bold', y=1.02)

    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    fp = FramePlotter(loader, cfg)
    for idx, ci in enumerate(cases):
        r, c = divmod(idx, cols)
        fp.plot_deformed_comparison(ci, ax=axes[r, c])

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    plt.tight_layout()
    path = os.path.join(cfg.save_dir,
                        'multi_case_frames.png')
    plt.savefig(path, dpi=cfg.dpi, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {path}")


def plot_structures(loader, cfg, cases):
    n = len(cases)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(7*cols, 6*rows))
    fig.suptitle('Frame Structures (Undeformed)',
                 fontsize=16, fontweight='bold', y=1.02)

    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    fp = FramePlotter(loader, cfg)
    for idx, ci in enumerate(cases):
        r, c = divmod(idx, cols)
        fp.plot_frame_structure(
            ci, ax=axes[r, c], show_labels=False)

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    plt.tight_layout()
    path = os.path.join(cfg.save_dir,
                        'multi_structures.png')
    plt.savefig(path, dpi=cfg.dpi, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print("  PIGNN — Plot Results v4 (XZ plane)")
    print("=" * 60)

    cfg = PlotConfig()
    loader = ResultsLoader(cfg)
    loader.load_all()

    n_total = len(loader.raw_data)
    n_train = int(n_total * cfg.train_ratio)

    # Mix of train + test cases
    cases = cfg.cases_to_plot or [0, 1, 2]
    test_cases = list(range(
        n_train, min(n_train + 2, n_total)))
    all_cases = sorted(set(cases + test_cases))

    print(f"\n  Cases: {all_cases}")
    print(f"  Train: {[c for c in all_cases if c < n_train]}")
    print(f"  Test:  {[c for c in all_cases if c >= n_train]}")

    # ═══════════════════════════════════════
    # 1. Undeformed structures
    # ═══════════════════════════════════════
    print(f"\n── 1. Undeformed Structures ──")
    plot_structures(loader, cfg, all_cases[:6])

    # ═══════════════════════════════════════
    # 2. Deformed comparison
    # ═══════════════════════════════════════
    print(f"\n── 2. Deformed Shape Comparison ──")
    plot_multi(loader, cfg, all_cases[:6])

    # ═══════════════════════════════════════
    # INDIVIDUAL DEFORMED SHAPE PLOTS
    # ═══════════════════════════════════════
    print(f"\n── Individual Deformed Plots ──")
    fp = FramePlotter(loader, cfg)
    for ci in all_cases:
        fp.plot_single_deformed(ci)

    # ═══════════════════════════════════════
    # 3. Detailed single-case analysis
    # ═══════════════════════════════════════
    print(f"\n── 3. Detailed Analysis ──")
    fp = FramePlotter(loader, cfg)
    for ci in all_cases[:3]:
        fp.plot_single_case_detail(ci)

    # ═══════════════════════════════════════
    # 4. Displacement scatter
    # ═══════════════════════════════════════
    print(f"\n── 4. Displacement Scatter ──")
    dp = DisplacementPlotter(loader, cfg)
    dp.plot_displacement_scatter()

    # ═══════════════════════════════════════
    # 5. Error heatmap (all cases)
    # ═══════════════════════════════════════
    print(f"\n── 5. Error Heatmap ──")
    dp.plot_error_heatmap(list(range(n_total)))

    # ═══════════════════════════════════════
    # 6. Energy comparison
    # ═══════════════════════════════════════
    print(f"\n── 6. Energy ──")
    ep = EnergyPlotter(loader, cfg)
    ep.plot_energy_comparison(all_cases)
    ep.plot_energy_scatter()

    # ═══════════════════════════════════════
    # 7. Summary table
    # ═══════════════════════════════════════
    print_summary(loader, cfg)

    print(f"\n{'='*60}")
    print(f"  DONE — all plots in {cfg.save_dir}/")
    print(f"{'='*60}")