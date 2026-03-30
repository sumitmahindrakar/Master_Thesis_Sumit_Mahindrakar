"""
working_pignn_test.py
=====================
Tests 3 approaches that CAN work for stiff frames:

A) Jacobi-preconditioned energy (fixes conditioning)
B) Normalized equilibrium residual (avoids energy altogether)  
C) Hybrid: supervised warm-start + energy fine-tune

All operate on the actual PIGNN model.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path

CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)

from model import PIGNN
from step_2_grapg_constr import FrameData

# ════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════

print(f"{'='*70}")
print(f"  WORKING PIGNN APPROACHES")
print(f"{'='*70}")

data_list = torch.load("DATA/graph_dataset_norm.pt",
                        weights_only=False)
raw_data = torch.load("DATA/graph_dataset.pt",
                       weights_only=False)

from normalizer import PhysicsScaler
if not hasattr(data_list[0], 'F_c'):
    data_list = PhysicsScaler.compute_and_store_list(data_list)
if not hasattr(raw_data[0], 'F_c'):
    raw_data = PhysicsScaler.compute_and_store_list(raw_data)

device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')
print(f"  Device: {device}")
print(f"  {len(data_list)} graphs")

N = data_list[0].num_nodes
n_elem = data_list[0].connectivity.shape[0]

# ═══ Precompute K diagonal for Jacobi preconditioning ═══

def compute_K_diagonal(data):
    """
    Assemble ONLY the diagonal of K for each node/DOF.
    Used for Jacobi preconditioning.
    Returns: (N, 3) tensor of K_diag values.
    """
    conn = data.connectivity.numpy()
    n_nodes = data.num_nodes
    dirs = data.elem_directions.numpy()
    L = data.elem_lengths.numpy()
    EA = (data.prop_E * data.prop_A).numpy()
    EI = (data.prop_E * data.prop_I22).numpy()

    K_diag = np.zeros((n_nodes, 3))

    for e in range(len(conn)):
        nA, nB = conn[e]
        c = dirs[e, 0]
        s = dirs[e, 2]
        Le = L[e]

        ea_l = EA[e] / Le
        ei_l = EI[e] / Le
        ei_l2 = EI[e] / Le**2
        ei_l3 = EI[e] / Le**3

        # Local K diagonal: [ea_l, 12*ei_l3, 4*ei_l,
        #                     ea_l, 12*ei_l3, 4*ei_l]
        # Transform: K_global_diag from T^T K_loc T diagonal
        # For node A (DOFs 0,1,2 in local → ux,uz,θ in global):
        # ux contribution: c²*ea_l + s²*12*ei_l3
        # uz contribution: s²*ea_l + c²*12*ei_l3
        # θ contribution: 4*ei_l (θ_loc = -θ, so (-1)²*4*ei_l)

        K_diag[nA, 0] += c**2 * ea_l + s**2 * 12*ei_l3
        K_diag[nA, 1] += s**2 * ea_l + c**2 * 12*ei_l3
        K_diag[nA, 2] += 4 * ei_l

        K_diag[nB, 0] += c**2 * ea_l + s**2 * 12*ei_l3
        K_diag[nB, 1] += s**2 * ea_l + c**2 * 12*ei_l3
        K_diag[nB, 2] += 4 * ei_l

    return torch.tensor(K_diag, dtype=torch.float32)


# Precompute for first case (same mesh → same K_diag structure)
K_diag = compute_K_diagonal(raw_data[0])
print(f"\n  K diagonal (Jacobi preconditioner):")
print(f"    ux: [{K_diag[:,0].min():.4e}, {K_diag[:,0].max():.4e}]")
print(f"    uz: [{K_diag[:,1].min():.4e}, {K_diag[:,1].max():.4e}]")
print(f"    θ:  [{K_diag[:,2].min():.4e}, {K_diag[:,2].max():.4e}]")

# Jacobi scale: 1/sqrt(K_diag)
J_scale = 1.0 / torch.sqrt(K_diag.clamp(min=1.0))
print(f"  Jacobi scale:")
print(f"    ux: [{J_scale[:,0].min():.6e}, {J_scale[:,0].max():.6e}]")
print(f"    uz: [{J_scale[:,1].min():.6e}, {J_scale[:,1].max():.6e}]")
print(f"    θ:  [{J_scale[:,2].min():.6e}, {J_scale[:,2].max():.6e}]")

# Store in all data objects
for d in data_list:
    d.J_scale = J_scale.clone()
for d in raw_data:
    d.J_scale = J_scale.clone()


# ════════════════════════════════════════════════════════
# HELPER: Compute per-DOF errors
# ════════════════════════════════════════════════════════

def compute_errors(model, norm_data, raw_data, device,
                   output_mode='nondim'):
    model.eval()
    errs = {'ux': [], 'uz': [], 'th': [], 'total': []}
    with torch.no_grad():
        for nd, rd in zip(norm_data, raw_data):
            nd = nd.clone().to(device)
            pred_raw = model(nd)

            if output_mode == 'nondim':
                u_pred = torch.zeros_like(pred_raw)
                u_pred[:, 0] = pred_raw[:, 0] * nd.ux_c
                u_pred[:, 1] = pred_raw[:, 1] * nd.uz_c
                u_pred[:, 2] = pred_raw[:, 2] * nd.theta_c
            elif output_mode == 'jacobi':
                J = nd.J_scale.to(device)
                u_pred = pred_raw * J
            else:
                u_pred = pred_raw

            u_true = rd.y_node.to(device)
            for d, name in enumerate(['ux', 'uz', 'th']):
                e = (u_pred[:,d] - u_true[:,d]).pow(2).sum().sqrt()
                r = u_true[:,d].pow(2).sum().sqrt().clamp(min=1e-15)
                errs[name].append((e/r).item())

            e_t = (u_pred - u_true).pow(2).sum().sqrt()
            r_t = u_true.pow(2).sum().sqrt().clamp(min=1e-15)
            errs['total'].append((e_t/r_t).item())

    return {k: np.mean(v) for k, v in errs.items()}


# ════════════════════════════════════════════════════════
# APPROACH A: Jacobi-Preconditioned Energy
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  APPROACH A: Jacobi-Preconditioned Energy")
print(f"{'='*70}")
print(f"""
  Key idea: network outputs u_hat ~ O(1)
  Physical displacements: u = u_hat × J_scale
  Where J_scale = 1/√K_diag
  
  This transforms the energy Hessian from K to D^(-1/2) K D^(-1/2)
  which has MUCH better conditioning.
""")


class JacobiEnergyLoss(nn.Module):
    """
    Energy loss with Jacobi preconditioning.
    
    Network outputs u_hat (O(1) for all DOFs).
    Physical: u = u_hat * J_scale where J_scale = 1/√K_diag.
    Energy computed in float64 for precision.
    """

    def forward(self, model, data, axial_weight=1.0):
        pred_raw = model(data)  # (N, 3) from GNN

        # ── Jacobi scaling to physical ──
        J = data.J_scale.to(pred_raw.device)
        u_phys = pred_raw * J  # (N, 3) physical

        # ── Upcast to float64 ──
        u64 = u_phys.double()

        conn = data.connectivity
        nA, nB = conn[:, 0].long(), conn[:, 1].long()
        n_elem = conn.shape[0]

        L = data.elem_lengths.double()
        EA = (data.prop_E * data.prop_A).double()
        EI = (data.prop_E * data.prop_I22).double()
        c = data.elem_directions[:, 0].double()
        s = data.elem_directions[:, 2].double()
        F_ext = data.F_ext.double()

        # ── Local displacements ──
        ux_A = u64[nA, 0]; uz_A = u64[nA, 1]; th_A = u64[nA, 2]
        ux_B = u64[nB, 0]; uz_B = u64[nB, 1]; th_B = u64[nB, 2]

        u_A_loc =  c*ux_A + s*uz_A
        w_A_loc = -s*ux_A + c*uz_A
        u_B_loc =  c*ux_B + s*uz_B
        w_B_loc = -s*ux_B + c*uz_B
        th_A_loc = -th_A
        th_B_loc = -th_B

        d_local = torch.stack([
            u_A_loc, w_A_loc, th_A_loc,
            u_B_loc, w_B_loc, th_B_loc
        ], dim=1)

        # ── Element stiffness ──
        ea_l = EA/L; ei_l = EI/L
        ei_l2 = EI/L**2; ei_l3 = EI/L**3

        K_loc = torch.zeros(n_elem, 6, 6,
                            dtype=torch.float64,
                            device=u64.device)

        K_loc[:,0,0] =  ea_l*axial_weight
        K_loc[:,0,3] = -ea_l*axial_weight
        K_loc[:,3,0] = -ea_l*axial_weight
        K_loc[:,3,3] =  ea_l*axial_weight

        K_loc[:,1,1] =  12*ei_l3; K_loc[:,1,2] =  6*ei_l2
        K_loc[:,1,4] = -12*ei_l3; K_loc[:,1,5] =  6*ei_l2
        K_loc[:,2,1] =  6*ei_l2;  K_loc[:,2,2] =  4*ei_l
        K_loc[:,2,4] = -6*ei_l2;  K_loc[:,2,5] =  2*ei_l
        K_loc[:,4,1] = -12*ei_l3; K_loc[:,4,2] = -6*ei_l2
        K_loc[:,4,4] =  12*ei_l3; K_loc[:,4,5] = -6*ei_l2
        K_loc[:,5,1] =  6*ei_l2;  K_loc[:,5,2] =  2*ei_l
        K_loc[:,5,4] = -6*ei_l2;  K_loc[:,5,5] =  4*ei_l

        # ── U = ½ d^T K d ──
        Kd = torch.bmm(K_loc, d_local.unsqueeze(2))
        U = 0.5 * torch.bmm(
            d_local.unsqueeze(1), Kd
        ).squeeze().sum()

        # ── W = F · u ──
        W = (F_ext[:,0]*u64[:,0] + F_ext[:,1]*u64[:,1]
             + F_ext[:,2]*u64[:,2]).sum()

        Pi = U - W

        # Normalize
        n_graphs = data.num_graphs if (
            hasattr(data, 'batch') and data.batch is not None
        ) else 1
        E_c = (data.F_c * data.ux_c).double().clamp(min=1e-30)
        if hasattr(data, 'batch') and data.batch is not None:
            E_c = E_c[0]

        Pi_norm = Pi / (E_c * n_graphs)

        loss_dict = {
            'Pi': Pi_norm.item(),
            'U': (U/n_graphs).item(),
            'W': (W/n_graphs).item(),
            'U_over_W': (U/W.abs().clamp(min=1e-30)).item(),
            'pred_range': [pred_raw.min().item(),
                          pred_raw.max().item()],
            'u_range': [u_phys.min().item(),
                       u_phys.max().item()],
        }

        return Pi_norm.float(), loss_dict, pred_raw, u_phys


# ════════════════════════════════════════════════════════
# APPROACH B: Normalized Equilibrium Residual
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  APPROACH B: Normalized Equilibrium Residual")
print(f"{'='*70}")
print(f"""
  Key idea: instead of energy Π = U - W, enforce Ku = F directly.
  
  Residual at node i: R_i = Σ(element forces at i) - F_ext_i
  Loss = (1/N_free) Σ ||R_i / K_diag_i||²
  
  Normalizing by K_diag makes all residual components O(1).
  This is mathematically a preconditioned least-squares: ||D⁻¹(Ku-F)||²
""")


class EquilibriumResidualLoss(nn.Module):
    """
    Normalized equilibrium residual loss.
    
    For each free node, computes the force imbalance
    and normalizes by the diagonal stiffness.
    
    This avoids the energy conditioning problem entirely.
    The network can output either:
      - Non-dimensional displacements (mode='nondim')
      - Jacobi-scaled displacements (mode='jacobi')
    """

    def __init__(self, output_mode='jacobi'):
        super().__init__()
        self.output_mode = output_mode

    def forward(self, model, data, axial_weight=1.0):
        pred_raw = model(data)

        # ── Convert to physical displacements ──
        if self.output_mode == 'jacobi':
            J = data.J_scale.to(pred_raw.device)
            u_phys = pred_raw * J
        else:
            ux_c = data.ux_c.to(pred_raw.device)
            uz_c = data.uz_c.to(pred_raw.device)
            theta_c = data.theta_c.to(pred_raw.device)
            if hasattr(data, 'batch') and data.batch is not None:
                ux_c = ux_c[data.batch]
                uz_c = uz_c[data.batch]
                theta_c = theta_c[data.batch]
            u_phys = torch.zeros_like(pred_raw)
            u_phys[:, 0] = pred_raw[:, 0] * ux_c
            u_phys[:, 1] = pred_raw[:, 1] * uz_c
            u_phys[:, 2] = pred_raw[:, 2] * theta_c

        # ── Compute element internal forces ──
        u64 = u_phys.double()

        conn = data.connectivity
        nA, nB = conn[:, 0].long(), conn[:, 1].long()
        n_elem = conn.shape[0]
        n_nodes = pred_raw.shape[0]

        L = data.elem_lengths.double()
        EA = (data.prop_E * data.prop_A).double()
        EI = (data.prop_E * data.prop_I22).double()
        c = data.elem_directions[:, 0].double()
        s = data.elem_directions[:, 2].double()
        F_ext = data.F_ext.double()

        # Local displacements
        ux_A = u64[nA, 0]; uz_A = u64[nA, 1]; th_A = u64[nA, 2]
        ux_B = u64[nB, 0]; uz_B = u64[nB, 1]; th_B = u64[nB, 2]

        u_A_loc =  c*ux_A + s*uz_A
        w_A_loc = -s*ux_A + c*uz_A
        u_B_loc =  c*ux_B + s*uz_B
        w_B_loc = -s*ux_B + c*uz_B
        th_A_loc = -th_A
        th_B_loc = -th_B

        d_local = torch.stack([
            u_A_loc, w_A_loc, th_A_loc,
            u_B_loc, w_B_loc, th_B_loc
        ], dim=1)

        # Element stiffness
        ea_l = EA/L; ei_l = EI/L
        ei_l2 = EI/L**2; ei_l3 = EI/L**3

        K_loc = torch.zeros(n_elem, 6, 6,
                            dtype=torch.float64,
                            device=u64.device)

        K_loc[:,0,0] =  ea_l*axial_weight
        K_loc[:,0,3] = -ea_l*axial_weight
        K_loc[:,3,0] = -ea_l*axial_weight
        K_loc[:,3,3] =  ea_l*axial_weight

        K_loc[:,1,1] =  12*ei_l3; K_loc[:,1,2] =  6*ei_l2
        K_loc[:,1,4] = -12*ei_l3; K_loc[:,1,5] =  6*ei_l2
        K_loc[:,2,1] =  6*ei_l2;  K_loc[:,2,2] =  4*ei_l
        K_loc[:,2,4] = -6*ei_l2;  K_loc[:,2,5] =  2*ei_l
        K_loc[:,4,1] = -12*ei_l3; K_loc[:,4,2] = -6*ei_l2
        K_loc[:,4,4] =  12*ei_l3; K_loc[:,4,5] = -6*ei_l2
        K_loc[:,5,1] =  6*ei_l2;  K_loc[:,5,2] =  2*ei_l
        K_loc[:,5,4] = -6*ei_l2;  K_loc[:,5,5] =  4*ei_l

        # f_local = K_local @ d_local per element
        f_local = torch.bmm(K_loc, d_local.unsqueeze(2)).squeeze(2)

        # ── Transform back to global ──
        # f_global = T^T @ f_local
        f_global_A = torch.zeros(n_elem, 3,
                                 dtype=torch.float64,
                                 device=u64.device)
        f_global_B = torch.zeros_like(f_global_A)

        # Node A: [f_u_loc, f_w_loc, f_th_loc] → [Fx, Fz, M]
        f_global_A[:, 0] = c*f_local[:,0] - s*f_local[:,1]
        f_global_A[:, 1] = s*f_local[:,0] + c*f_local[:,1]
        f_global_A[:, 2] = -f_local[:, 2]  # θ_loc = -θ → M = -M_loc

        f_global_B[:, 0] = c*f_local[:,3] - s*f_local[:,4]
        f_global_B[:, 1] = s*f_local[:,3] + c*f_local[:,4]
        f_global_B[:, 2] = -f_local[:, 5]

        # ── Assemble nodal force sums ──
        F_internal = torch.zeros(n_nodes, 3,
                                 dtype=torch.float64,
                                 device=u64.device)
        F_internal.scatter_add_(0, nA.unsqueeze(1).expand_as(f_global_A),
                                f_global_A)
        F_internal.scatter_add_(0, nB.unsqueeze(1).expand_as(f_global_B),
                                f_global_B)

        # ── Residual = internal - external ──
        R = F_internal - F_ext  # (N, 3)

        # ── Mask: only free DOFs ──
        bc_disp = data.bc_disp.double().to(u64.device)
        bc_rot = data.bc_rot.double().to(u64.device)
        free_disp = (1.0 - bc_disp)  # (N, 1)
        free_rot = (1.0 - bc_rot)    # (N, 1)

        free_mask = torch.cat([free_disp, free_disp, free_rot],
                              dim=1)  # (N, 3)

        R_free = R * free_mask

        # ── Normalize by K_diag (Jacobi preconditioning) ──
        K_d = data.J_scale.double().to(u64.device)
        # J_scale = 1/√K_diag, so K_diag = 1/J_scale²
        K_diag_inv = K_d**2  # 1/K_diag
        R_normalized = R_free * K_diag_inv  # = R / K_diag

        # ── Loss: mean squared normalized residual ──
        n_free = free_mask.sum().clamp(min=1)
        loss = (R_normalized**2).sum() / n_free

        # Also compute raw residual norm for logging
        R_raw_norm = (R_free**2).sum().sqrt()

        n_graphs = data.num_graphs if (
            hasattr(data, 'batch') and data.batch is not None
        ) else 1

        loss_dict = {
            'residual_loss': loss.item(),
            'R_raw': R_raw_norm.item() / n_graphs,
            'R_ux': (R_free[:, 0]**2).sum().sqrt().item() / n_graphs,
            'R_uz': (R_free[:, 1]**2).sum().sqrt().item() / n_graphs,
            'R_My': (R_free[:, 2]**2).sum().sqrt().item() / n_graphs,
            'pred_range': [pred_raw.min().item(),
                          pred_raw.max().item()],
        }

        return loss.float(), loss_dict, pred_raw, u_phys


# ════════════════════════════════════════════════════════
# APPROACH C: Supervised + Physics
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  APPROACH C: Supervised warm-start + Physics fine-tune")
print(f"{'='*70}")


class SupervisedLoss(nn.Module):
    """Per-DOF normalized supervised loss."""

    def __init__(self, output_mode='jacobi'):
        super().__init__()
        self.output_mode = output_mode

    def forward(self, model, norm_data, raw_data):
        pred_raw = model(norm_data)

        if self.output_mode == 'jacobi':
            J = norm_data.J_scale.to(pred_raw.device)
            u_pred = pred_raw * J
        else:
            u_pred = torch.zeros_like(pred_raw)
            u_pred[:, 0] = pred_raw[:, 0] * norm_data.ux_c
            u_pred[:, 1] = pred_raw[:, 1] * norm_data.uz_c
            u_pred[:, 2] = pred_raw[:, 2] * norm_data.theta_c

        u_true = raw_data.y_node.to(pred_raw.device)

        # Per-DOF normalized MSE
        losses = []
        for d in range(3):
            ref = u_true[:, d].pow(2).mean().clamp(min=1e-30)
            mse = (u_pred[:, d] - u_true[:, d]).pow(2).mean()
            losses.append(mse / ref)

        loss = sum(losses) / 3

        loss_dict = {
            'ux_loss': losses[0].item(),
            'uz_loss': losses[1].item(),
            'th_loss': losses[2].item(),
        }

        return loss, loss_dict, pred_raw, u_pred


# ════════════════════════════════════════════════════════
# CREATE MODELS (fresh for each test)
# ════════════════════════════════════════════════════════

def make_model(hidden=64, layers=3):
    model = PIGNN(
        node_in_dim=10, edge_in_dim=7,
        hidden_dim=hidden, n_layers=layers,
    ).to(device)

    # Small Xavier init (NOT zero!)
    with torch.no_grad():
        for dec in [model.decoder_ux, model.decoder_uz,
                    model.decoder_th]:
            last = dec.layers[-1]
            nn.init.xavier_uniform_(last.weight, gain=0.01)
            last.bias.zero_()

    return model


# ════════════════════════════════════════════════════════
# RUN TEST A: Jacobi Energy + Adam
# ════════════════════════════════════════════════════════

print(f"\n{'─'*70}")
print(f"  TEST A: Jacobi-preconditioned energy + Adam")
print(f"{'─'*70}")

model_a = make_model()
loss_fn_a = JacobiEnergyLoss()
opt_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)

hist_a = {'step': [], 'Pi': [], 'ux': [], 'uz': [], 'th': []}

for step in range(3000):
    model_a.train()

    # Process each graph individually (avoids batch device issues)
    total_loss = 0.0
    total_dict = None
    for nd in data_list:
        nd = nd.clone().to(device)
        opt_a.zero_grad()
        loss, ld, _, _ = loss_fn_a(model_a, nd)
        loss.backward()
        total_loss += loss.item()
        total_dict = ld

    torch.nn.utils.clip_grad_norm_(model_a.parameters(), 50.0)
    opt_a.step()

    if step % 300 == 0 or step == 2999:
        errs = compute_errors(model_a, data_list, raw_data,
                              device, output_mode='jacobi')
        hist_a['step'].append(step)
        hist_a['Pi'].append(total_loss / len(data_list))
        hist_a['ux'].append(errs['ux'])
        hist_a['uz'].append(errs['uz'])
        hist_a['th'].append(errs['th'])

        print(f"  {step:5d}: Π={total_loss/len(data_list):11.4e}  "
              f"err=[{errs['ux']:.4f}, {errs['uz']:.4f}, "
              f"{errs['th']:.4f}]  "
              f"pred=[{total_dict['pred_range'][0]:.4f}, "
              f"{total_dict['pred_range'][1]:.4f}]")


# ════════════════════════════════════════════════════════
# RUN TEST B: Equilibrium Residual + Adam
# ════════════════════════════════════════════════════════

print(f"\n{'─'*70}")
print(f"  TEST B: Equilibrium residual + Adam")
print(f"{'─'*70}")

model_b = make_model()
loss_fn_b = EquilibriumResidualLoss(output_mode='jacobi')
opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)

hist_b = {'step': [], 'R': [], 'ux': [], 'uz': [], 'th': []}

for step in range(3000):
    model_b.train()

    total_loss = 0.0
    total_dict = None
    for nd in data_list:
        nd = nd.clone().to(device)
        opt_b.zero_grad()
        loss, ld, _, _ = loss_fn_b(model_b, nd)
        loss.backward()
        total_loss += loss.item()
        total_dict = ld

    torch.nn.utils.clip_grad_norm_(model_b.parameters(), 50.0)
    opt_b.step()

    if step % 300 == 0 or step == 2999:
        errs = compute_errors(model_b, data_list, raw_data,
                              device, output_mode='jacobi')
        hist_b['step'].append(step)
        hist_b['R'].append(total_loss / len(data_list))
        hist_b['ux'].append(errs['ux'])
        hist_b['uz'].append(errs['uz'])
        hist_b['th'].append(errs['th'])

        print(f"  {step:5d}: R={total_loss/len(data_list):11.4e}  "
              f"err=[{errs['ux']:.4f}, {errs['uz']:.4f}, "
              f"{errs['th']:.4f}]  "
              f"R_dof=[{total_dict['R_ux']:.2e}, "
              f"{total_dict['R_uz']:.2e}, "
              f"{total_dict['R_My']:.2e}]")


# ════════════════════════════════════════════════════════
# RUN TEST C: Supervised warm-start → physics fine-tune
# ════════════════════════════════════════════════════════

print(f"\n{'─'*70}")
print(f"  TEST C: Supervised warm-start + equilibrium fine-tune")
print(f"{'─'*70}")

model_c = make_model()
loss_fn_sup = SupervisedLoss(output_mode='jacobi')
loss_fn_phys = EquilibriumResidualLoss(output_mode='jacobi')
opt_c = torch.optim.Adam(model_c.parameters(), lr=1e-3)

hist_c = {'step': [], 'loss': [], 'ux': [], 'uz': [], 'th': []}

# Phase 1: Supervised (1000 steps)
print(f"\n  Phase 1: SUPERVISED (1000 steps)")
for step in range(1000):
    model_c.train()

    total_loss = 0.0
    for nd, rd in zip(data_list, raw_data):
        nd = nd.clone().to(device)
        rd_dev = rd.clone().to(device)
        opt_c.zero_grad()
        loss, ld, _, _ = loss_fn_sup(model_c, nd, rd_dev)
        loss.backward()
        total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model_c.parameters(), 50.0)
    opt_c.step()

    if step % 200 == 0 or step == 999:
        errs = compute_errors(model_c, data_list, raw_data,
                              device, output_mode='jacobi')
        hist_c['step'].append(step)
        hist_c['loss'].append(total_loss / len(data_list))
        hist_c['ux'].append(errs['ux'])
        hist_c['uz'].append(errs['uz'])
        hist_c['th'].append(errs['th'])

        print(f"  {step:5d}: L={total_loss/len(data_list):11.4e}  "
              f"err=[{errs['ux']:.4f}, {errs['uz']:.4f}, "
              f"{errs['th']:.4f}]")

# Phase 2: Physics fine-tune (2000 steps)
print(f"\n  Phase 2: PHYSICS (equilibrium) fine-tune (2000 steps)")
opt_c2 = torch.optim.Adam(model_c.parameters(), lr=1e-4)

for step in range(2000):
    model_c.train()

    total_loss = 0.0
    for nd in data_list:
        nd = nd.clone().to(device)
        opt_c2.zero_grad()
        loss, ld, _, _ = loss_fn_phys(model_c, nd)
        loss.backward()
        total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model_c.parameters(), 50.0)
    opt_c2.step()

    if step % 400 == 0 or step == 1999:
        errs = compute_errors(model_c, data_list, raw_data,
                              device, output_mode='jacobi')
        hist_c['step'].append(1000 + step)
        hist_c['loss'].append(total_loss / len(data_list))
        hist_c['ux'].append(errs['ux'])
        hist_c['uz'].append(errs['uz'])
        hist_c['th'].append(errs['th'])

        print(f"  {1000+step:5d}: R={total_loss/len(data_list):11.4e}  "
              f"err=[{errs['ux']:.4f}, {errs['uz']:.4f}, "
              f"{errs['th']:.4f}]")


# ════════════════════════════════════════════════════════
# COMPARISON
# ════════════════════════════════════════════════════════

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('PIGNN Training Approaches Comparison',
             fontsize=14, fontweight='bold')

colors = {'A': 'blue', 'B': 'red', 'C': 'green'}

for d, name in enumerate(['ux', 'uz', 'th']):
    ax = axes[d]
    if hist_a[name]:
        ax.semilogy(hist_a['step'], hist_a[name],
                    'b-o', ms=4, label='A: Jacobi Energy')
    if hist_b[name]:
        ax.semilogy(hist_b['step'], hist_b[name],
                    'r-s', ms=4, label='B: Eq. Residual')
    if hist_c[name]:
        ax.semilogy(hist_c['step'], hist_c[name],
                    'g-^', ms=4, label='C: Supervised+Physics')
    ax.axhline(y=0.05, color='gray', linestyle=':',
               label='5% target')
    ax.set_title(f'{name} relative error')
    ax.set_xlabel('Step')
    ax.set_ylim(1e-4, 2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("RESULTS", exist_ok=True)
plt.savefig('RESULTS/working_approaches.png',
            dpi=150, bbox_inches='tight')
plt.show()


# ════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  FINAL COMPARISON")
print(f"{'='*70}")

for label, h, key_list in [
    ('A: Jacobi Energy', hist_a, ['ux', 'uz', 'th']),
    ('B: Eq. Residual', hist_b, ['ux', 'uz', 'th']),
    ('C: Supervised+Physics', hist_c, ['ux', 'uz', 'th']),
]:
    if h['ux']:
        best = {k: min(h[k]) for k in key_list}
        worst = max(best.values())
        print(f"\n  {label}:")
        print(f"    Best err: ux={best['ux']:.4f}, "
              f"uz={best['uz']:.4f}, θ={best['th']:.4f}")
        print(f"    Worst DOF: {worst:.4f}")
        if worst < 0.05:
            print(f"    ✓ EXCELLENT (<5%)")
        elif worst < 0.10:
            print(f"    ✓ GOOD (<10%)")
        elif worst < 0.50:
            print(f"    ~ PARTIAL (some DOFs converging)")
        else:
            print(f"    ✗ NOT converged")

print(f"\n{'='*70}")
print(f"  RECOMMENDATIONS")
print(f"{'='*70}")
print(f"""
  For UNSUPERVISED only:
    → Use Approach B (equilibrium residual) if it converges
    → Key: Jacobi preconditioning + float64 energy
    
  For BEST results:
    → Use Approach C (supervised warm-start + physics fine-tune)
    → 1000 supervised steps → 2000 physics steps
    → Demonstrates physics regularization value

  For thesis:
    → Show comparison of all three approaches
    → Discuss conditioning challenges
    → Highlight that direct Ku=F solve converges perfectly
      but GNN training requires special treatment
""")