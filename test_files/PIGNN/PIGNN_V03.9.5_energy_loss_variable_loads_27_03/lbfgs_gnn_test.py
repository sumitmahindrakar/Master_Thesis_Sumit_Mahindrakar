"""
lbfgs_gnn_test.py — Test L-BFGS training on actual PIGNN
=========================================================
Proves that L-BFGS can train the GNN via energy minimization.
Uses float64 for energy, float32 for model.
"""

import torch
import numpy as np
import os
from pathlib import Path

CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)

from model import PIGNN
from energy_loss import FrameEnergyLoss
from step_2_grapg_constr import FrameData

# ════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════

print(f"{'='*70}")
print(f"  L-BFGS GNN Training Test")
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

print(f"  {len(data_list)} graphs loaded")
print(f"  F_c = {data_list[0].F_c.item():.4e}")
print(f"  u_c = {data_list[0].u_c.item():.4e}")

# ════════════════════════════════════════════════════════
# Float64 MATRIX ENERGY (verified correct in Test 2)
# ════════════════════════════════════════════════════════

class Float64EnergyLoss(torch.nn.Module):
    """
    Energy loss computed entirely in float64.
    Uses matrix form (d^T K d) for numerical stability.
    
    The model outputs float32 pred_raw, which we upcast.
    Gradients flow back through float64 → float32 automatically.
    """

    def __init__(self):
        super().__init__()

    def forward(self, model, data, axial_weight=1.0):
        # ── Model prediction (float32) ──
        pred_raw_f32 = model(data)  # (N, 3) non-dim

        # ── Upcast to float64 for energy computation ──
        pred_raw = pred_raw_f32.double()

        # ── Convert to physical displacements ──
        ux_c = data.ux_c.double()
        uz_c = data.uz_c.double()
        theta_c = data.theta_c.double()

        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
            ux_c = ux_c[batch]
            uz_c = uz_c[batch]
            theta_c = theta_c[batch]

        u_phys = torch.zeros_like(pred_raw)
        u_phys[:, 0] = pred_raw[:, 0] * ux_c
        u_phys[:, 1] = pred_raw[:, 1] * uz_c
        u_phys[:, 2] = pred_raw[:, 2] * theta_c

        # ── Element data in float64 ──
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
        ux_A = u_phys[nA, 0]; uz_A = u_phys[nA, 1]
        th_A = u_phys[nA, 2]
        ux_B = u_phys[nB, 0]; uz_B = u_phys[nB, 1]
        th_B = u_phys[nB, 2]

        u_A_loc =  c * ux_A + s * uz_A
        w_A_loc = -s * ux_A + c * uz_A
        u_B_loc =  c * ux_B + s * uz_B
        w_B_loc = -s * ux_B + c * uz_B
        th_A_loc = -th_A
        th_B_loc = -th_B

        d_local = torch.stack([
            u_A_loc, w_A_loc, th_A_loc,
            u_B_loc, w_B_loc, th_B_loc
        ], dim=1)

        # ── Element stiffness matrices ──
        ea_l = EA / L
        ei_l = EI / L
        ei_l2 = EI / L**2
        ei_l3 = EI / L**3

        K_loc = torch.zeros(n_elem, 6, 6,
                            dtype=torch.float64,
                            device=pred_raw.device)

        # Axial (with curriculum weight)
        K_loc[:, 0, 0] =  ea_l * axial_weight
        K_loc[:, 0, 3] = -ea_l * axial_weight
        K_loc[:, 3, 0] = -ea_l * axial_weight
        K_loc[:, 3, 3] =  ea_l * axial_weight

        # Bending
        K_loc[:, 1, 1] =  12*ei_l3
        K_loc[:, 1, 2] =  6*ei_l2
        K_loc[:, 1, 4] = -12*ei_l3
        K_loc[:, 1, 5] =  6*ei_l2
        K_loc[:, 2, 1] =  6*ei_l2
        K_loc[:, 2, 2] =  4*ei_l
        K_loc[:, 2, 4] = -6*ei_l2
        K_loc[:, 2, 5] =  2*ei_l
        K_loc[:, 4, 1] = -12*ei_l3
        K_loc[:, 4, 2] = -6*ei_l2
        K_loc[:, 4, 4] =  12*ei_l3
        K_loc[:, 4, 5] = -6*ei_l2
        K_loc[:, 5, 1] =  6*ei_l2
        K_loc[:, 5, 2] =  2*ei_l
        K_loc[:, 5, 4] = -6*ei_l2
        K_loc[:, 5, 5] =  4*ei_l

        # ── U = ½ d^T K d ──
        Kd = torch.bmm(K_loc, d_local.unsqueeze(2))
        U_per_elem = 0.5 * torch.bmm(
            d_local.unsqueeze(1), Kd
        ).squeeze()
        U = U_per_elem.sum()

        # ── W = F · u ──
        W = (F_ext[:, 0] * u_phys[:, 0]
           + F_ext[:, 1] * u_phys[:, 1]
           + F_ext[:, 2] * u_phys[:, 2]).sum()

        Pi = U - W

        # ── Normalize ──
        if hasattr(data, 'batch') and data.batch is not None:
            n_graphs = data.num_graphs
        else:
            n_graphs = 1

        E_c = (data.F_c[0] * data.ux_c[0]).double().clamp(min=1e-30) \
            if hasattr(data, 'batch') and data.batch is not None \
            else (data.F_c * data.ux_c).double().clamp(min=1e-30)

        Pi_norm = Pi / (E_c * n_graphs)

        # ── Loss dict ──
        loss_dict = {
            'Pi':       Pi_norm.item(),
            'Pi_norm':  Pi_norm.item(),
            'U':        (U / n_graphs).item(),
            'W':        (W / n_graphs).item(),
            'U_over_W': (U / W.abs().clamp(min=1e-30)).item(),
            'raw_range': [pred_raw_f32.min().item(),
                         pred_raw_f32.max().item()],
            'ux_range':  [u_phys[:, 0].min().item(),
                         u_phys[:, 0].max().item()],
            'uz_range':  [u_phys[:, 1].min().item(),
                         u_phys[:, 1].max().item()],
            'th_range':  [u_phys[:, 2].min().item(),
                         u_phys[:, 2].max().item()],
        }

        # Return float32 loss for backward
        return Pi_norm.float(), loss_dict, pred_raw_f32, u_phys.float()


# ════════════════════════════════════════════════════════
# DISPLACEMENT ERROR (per-DOF)
# ════════════════════════════════════════════════════════

# def compute_errors(model, norm_data, raw_data, device):
#     """Compute per-DOF relative errors."""
#     model.eval()
#     errors = {'ux': [], 'uz': [], 'th': [], 'total': []}

#     with torch.no_grad():
#         for nd, rd in zip(norm_data, raw_data):
#             nd = nd.to(device)
#             pred_raw = model(nd)

#             u_pred = torch.zeros_like(pred_raw)
#             u_pred[:, 0] = pred_raw[:, 0] * nd.ux_c
#             u_pred[:, 1] = pred_raw[:, 1] * nd.uz_c
#             u_pred[:, 2] = pred_raw[:, 2] * nd.theta_c

#             u_true = rd.y_node.to(device)

#             for d, name in enumerate(['ux', 'uz', 'th']):
#                 e = (u_pred[:, d] - u_true[:, d]).pow(2).sum().sqrt()
#                 r = u_true[:, d].pow(2).sum().sqrt().clamp(min=1e-15)
#                 errors[name].append((e / r).item())

#             e_tot = (u_pred - u_true).pow(2).sum().sqrt()
#             r_tot = u_true.pow(2).sum().sqrt().clamp(min=1e-15)
#             errors['total'].append((e_tot / r_tot).item())

#     return {k: np.mean(v) for k, v in errors.items()}


def compute_errors(model, norm_data, raw_data, device):
    """Compute per-DOF relative errors."""
    model.eval()
    errors = {'ux': [], 'uz': [], 'th': [], 'total': []}

    with torch.no_grad():
        for nd, rd in zip(norm_data, raw_data):
            nd = nd.clone().to(device)  # ← clone!
            pred_raw = model(nd)

            u_pred = torch.zeros_like(pred_raw)
            u_pred[:, 0] = pred_raw[:, 0] * nd.ux_c
            u_pred[:, 1] = pred_raw[:, 1] * nd.uz_c
            u_pred[:, 2] = pred_raw[:, 2] * nd.theta_c

            u_true = rd.y_node.to(device)

            for d, name in enumerate(['ux', 'uz', 'th']):
                e = (u_pred[:, d] - u_true[:, d]).pow(2).sum().sqrt()
                r = u_true[:, d].pow(2).sum().sqrt().clamp(min=1e-15)
                errors[name].append((e / r).item())

            e_tot = (u_pred - u_true).pow(2).sum().sqrt()
            r_tot = u_true.pow(2).sum().sqrt().clamp(min=1e-15)
            errors['total'].append((e_tot / r_tot).item())

    return {k: np.mean(v) for k, v in errors.items()}

# ════════════════════════════════════════════════════════
# TEST A: L-BFGS with SMALLER model
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  TEST A: L-BFGS + small model (hidden=64, layers=3)")
print(f"{'='*70}")

device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')

model_small = PIGNN(
    node_in_dim=10,
    edge_in_dim=7,
    hidden_dim=64,
    n_layers=3,
).to(device)

# ── Small Xavier init (NOT zero) ──
with torch.no_grad():
    for dec in [model_small.decoder_ux,
                model_small.decoder_uz,
                model_small.decoder_th]:
        last = dec.layers[-1]
        torch.nn.init.xavier_uniform_(last.weight, gain=0.01)
        last.bias.zero_()

n_params = model_small.count_params()
print(f"  Parameters: {n_params:,}")
print(f"  Device: {device}")

# ── Verify non-zero initial output ──
# with torch.no_grad():
#     test_d = data_list[0].to(device)
#     test_pred = model_small(test_d)
#     print(f"  Initial pred: [{test_pred.min().item():.6f}, "
#           f"{test_pred.max().item():.6f}]")
#     assert test_pred.abs().max().item() > 1e-8, \
#         "Initial predictions are zero — init failed!"
with torch.no_grad():
    test_d = data_list[0].clone().to(device)  # ← clone first!
    test_pred = model_small(test_d)
    print(f"  Initial pred: [{test_pred.min().item():.6f}, "
          f"{test_pred.max().item():.6f}]")
    assert test_pred.abs().max().item() > 1e-8, \
        "Initial predictions are zero — init failed!"

# ── Full-batch: all training data as one forward pass ──
from torch_geometric.loader import DataLoader
from step_2_grapg_constr import FrameData

# Use all data for training (full batch)
# full_loader = DataLoader(data_list, batch_size=len(data_list),
#                          shuffle=False)
# full_batch = next(iter(full_loader)).to(device)

full_loader = DataLoader(data_list, batch_size=len(data_list),
                         shuffle=False)
full_batch = next(iter(full_loader)).to(device)

print(f"  Full batch: {full_batch.num_graphs} graphs, "
      f"{full_batch.num_nodes} total nodes")

loss_fn = Float64EnergyLoss()

# ── Target energy ──
with torch.no_grad():
    Pi_targets = []
    for rd in raw_data:
        u_true = rd.y_node
        U_t = FrameEnergyLoss()._strain_energy(u_true, rd)
        W_t = FrameEnergyLoss()._external_work(u_true, rd)
        E_c = (rd.F_c * rd.ux_c).clamp(min=1e-30)
        Pi_targets.append(((U_t - W_t) / E_c).item())
    avg_target = np.mean(Pi_targets)
    print(f"  Target Π/E_c: {avg_target:.6e}")


# ════════════════════════════════════
# Phase 1: Bending only (axial_w=0)
# ════════════════════════════════════

print(f"\n  Phase 1: BENDING ONLY (L-BFGS)")
print(f"  {'-'*60}")

optimizer = torch.optim.LBFGS(
    model_small.parameters(),
    lr=1.0,
    max_iter=20,
    history_size=50,
    line_search_fn='strong_wolfe',
    tolerance_grad=1e-9,
    tolerance_change=1e-11,
)

history = {'step': [], 'Pi': [], 'err_ux': [],
           'err_uz': [], 'err_th': []}

for step in range(200):
    model_small.train()

    def closure():
        optimizer.zero_grad()
        loss, _, _, _ = loss_fn(model_small, full_batch,
                                axial_weight=0.0)
        loss.backward()
        return loss

    loss_val = optimizer.step(closure)

    if step % 10 == 0 or step < 5:
        errs = compute_errors(model_small, data_list,
                              raw_data, device)
        history['step'].append(step)
        history['Pi'].append(loss_val.item())
        history['err_ux'].append(errs['ux'])
        history['err_uz'].append(errs['uz'])
        history['err_th'].append(errs['th'])

        with torch.no_grad():
            pred = model_small(full_batch)

        # Gradient norm
        gn = 0.0
        for p in model_small.parameters():
            if p.grad is not None:
                gn += p.grad.float().norm().item()**2
        gn = gn**0.5

        print(f"  {step:4d}: Π={loss_val.item():11.4e}  "
              f"err=[{errs['ux']:.4f}, {errs['uz']:.4f}, "
              f"{errs['th']:.4f}]  "
              f"|∇|={gn:.2e}  "
              f"raw=[{pred.min().item():.4f}, "
              f"{pred.max().item():.4f}]")

        if gn < 1e-6 and step > 10:
            print(f"  Phase 1 converged!")
            break

print(f"\n  Phase 1 final errors: "
      f"ux={history['err_ux'][-1]:.4f}, "
      f"uz={history['err_uz'][-1]:.4f}, "
      f"θ={history['err_th'][-1]:.4f}")


# ════════════════════════════════════
# Phase 2: Full energy (axial ramp)
# ════════════════════════════════════

print(f"\n  Phase 2: FULL ENERGY with axial ramp (L-BFGS)")
print(f"  {'-'*60}")

optimizer2 = torch.optim.LBFGS(
    model_small.parameters(),
    lr=0.5,         # slightly lower for stability
    max_iter=20,
    history_size=50,
    line_search_fn='strong_wolfe',
    tolerance_grad=1e-9,
    tolerance_change=1e-11,
)

ramp_steps = 100
total_steps = 300

for step in range(total_steps):
    model_small.train()

    if step < ramp_steps:
        aw = step / ramp_steps
    else:
        aw = 1.0

    def closure():
        optimizer2.zero_grad()
        loss, _, _, _ = loss_fn(model_small, full_batch,
                                axial_weight=aw)
        loss.backward()
        return loss

    loss_val = optimizer2.step(closure)

    if step % 20 == 0 or step < 5 or step == total_steps - 1:
        errs = compute_errors(model_small, data_list,
                              raw_data, device)
        history['step'].append(200 + step)
        history['Pi'].append(loss_val.item())
        history['err_ux'].append(errs['ux'])
        history['err_uz'].append(errs['uz'])
        history['err_th'].append(errs['th'])

        print(f"  {step:4d}: Π={loss_val.item():11.4e}  "
              f"aw={aw:.2f}  "
              f"err=[{errs['ux']:.4f}, {errs['uz']:.4f}, "
              f"{errs['th']:.4f}]")


# ════════════════════════════════════════════════════════
# TEST B: L-BFGS with ORIGINAL model (hidden=128, layers=6)
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  TEST B: L-BFGS + original model (hidden=128, layers=6)")
print(f"{'='*70}")

model_orig = PIGNN(
    node_in_dim=10,
    edge_in_dim=7,
    hidden_dim=128,
    n_layers=6,
).to(device)

# Small Xavier init
with torch.no_grad():
    for dec in [model_orig.decoder_ux,
                model_orig.decoder_uz,
                model_orig.decoder_th]:
        last = dec.layers[-1]
        torch.nn.init.xavier_uniform_(last.weight, gain=0.01)
        last.bias.zero_()

print(f"  Parameters: {model_orig.count_params():,}")

optimizer3 = torch.optim.LBFGS(
    model_orig.parameters(),
    lr=1.0,
    max_iter=20,
    history_size=100,
    line_search_fn='strong_wolfe',
    tolerance_grad=1e-9,
)

history_b = {'step': [], 'Pi': [], 'err_ux': [],
             'err_uz': [], 'err_th': []}

print(f"\n  Full energy from start:")
for step in range(200):
    model_orig.train()

    def closure():
        optimizer3.zero_grad()
        loss, _, _, _ = loss_fn(model_orig, full_batch,
                                axial_weight=1.0)
        loss.backward()
        return loss

    loss_val = optimizer3.step(closure)

    if step % 10 == 0 or step < 5:
        errs = compute_errors(model_orig, data_list,
                              raw_data, device)
        history_b['step'].append(step)
        history_b['Pi'].append(loss_val.item())
        history_b['err_ux'].append(errs['ux'])
        history_b['err_uz'].append(errs['uz'])
        history_b['err_th'].append(errs['th'])

        print(f"  {step:4d}: Π={loss_val.item():11.4e}  "
              f"err=[{errs['ux']:.4f}, {errs['uz']:.4f}, "
              f"{errs['th']:.4f}]")


# ════════════════════════════════════════════════════════
# TEST C: Adam + Float64 energy (baseline comparison)
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  TEST C: Adam + Float64 energy (baseline)")
print(f"{'='*70}")

model_adam = PIGNN(
    node_in_dim=10,
    edge_in_dim=7,
    hidden_dim=64,
    n_layers=3,
).to(device)

with torch.no_grad():
    for dec in [model_adam.decoder_ux,
                model_adam.decoder_uz,
                model_adam.decoder_th]:
        last = dec.layers[-1]
        torch.nn.init.xavier_uniform_(last.weight, gain=0.01)
        last.bias.zero_()

print(f"  Parameters: {model_adam.count_params():,}")

opt_adam = torch.optim.Adam(model_adam.parameters(), lr=1e-3)

history_c = {'step': [], 'Pi': [], 'err_ux': [],
             'err_uz': [], 'err_th': []}

print(f"\n  Adam lr=1e-3, float64 energy:")
for step in range(2000):
    model_adam.train()
    opt_adam.zero_grad()

    loss, ld, _, _ = loss_fn(model_adam, full_batch,
                              axial_weight=1.0)
    if torch.isnan(loss):
        print(f"  NaN at step {step}")
        break

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model_adam.parameters(), 100.0
    )
    opt_adam.step()

    if step % 200 == 0 or step == 1999:
        errs = compute_errors(model_adam, data_list,
                              raw_data, device)
        history_c['step'].append(step)
        history_c['Pi'].append(ld['Pi'])
        history_c['err_ux'].append(errs['ux'])
        history_c['err_uz'].append(errs['uz'])
        history_c['err_th'].append(errs['th'])

        print(f"  {step:5d}: Π={ld['Pi']:11.4e}  "
              f"err=[{errs['ux']:.4f}, {errs['uz']:.4f}, "
              f"{errs['th']:.4f}]")


# ════════════════════════════════════════════════════════
# COMPARISON PLOT
# ════════════════════════════════════════════════════════

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('L-BFGS vs Adam for PIGNN Energy Training',
             fontsize=14, fontweight='bold')

# 1. Energy convergence
ax = axes[0, 0]
if history['Pi']:
    ax.plot(history['step'], history['Pi'],
            'b-o', ms=3, label='A: Small+L-BFGS')
if history_b['Pi']:
    ax.plot(history_b['step'], history_b['Pi'],
            'r-s', ms=3, label='B: Large+L-BFGS')
if history_c['Pi']:
    ax.plot(history_c['step'], history_c['Pi'],
            'g-^', ms=3, label='C: Small+Adam')
ax.axhline(y=avg_target, color='black', linestyle='--',
           label=f'Target: {avg_target:.4f}')
ax.set_title('Π/E_c convergence')
ax.set_xlabel('Step')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2-4. Per-DOF errors
for d, (name, ax_idx) in enumerate([
    ('ux error', (0, 1)),
    ('uz error', (1, 0)),
    ('θ error', (1, 1))
]):
    ax = axes[ax_idx]
    key = ['err_ux', 'err_uz', 'err_th'][d]

    if history[key]:
        ax.semilogy(history['step'], history[key],
                    'b-o', ms=3, label='A: Small+L-BFGS')
    if history_b[key]:
        ax.semilogy(history_b['step'], history_b[key],
                    'r-s', ms=3, label='B: Large+L-BFGS')
    if history_c[key]:
        ax.semilogy(history_c['step'], history_c[key],
                    'g-^', ms=3, label='C: Small+Adam')
    ax.axhline(y=0.05, color='gray', linestyle=':',
               label='5% error')
    ax.set_title(name)
    ax.set_ylim(1e-4, 2)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("RESULTS", exist_ok=True)
plt.savefig('RESULTS/lbfgs_vs_adam_gnn.png',
            dpi=150, bbox_inches='tight')
plt.show()


# ════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  FINAL COMPARISON")
print(f"{'='*70}")

for name, h in [('A (Small+L-BFGS)', history),
                ('B (Large+L-BFGS)', history_b),
                ('C (Small+Adam)', history_c)]:
    if h['err_ux']:
        best_ux = min(h['err_ux'])
        best_uz = min(h['err_uz'])
        best_th = min(h['err_th'])
        best_Pi = min(h['Pi'])
        print(f"\n  {name}:")
        print(f"    Best Π:  {best_Pi:.6e} "
              f"(target: {avg_target:.6e})")
        print(f"    Best err: ux={best_ux:.4f}, "
              f"uz={best_uz:.4f}, θ={best_th:.4f}")
        converged = max(best_ux, best_uz, best_th) < 0.1
        print(f"    {'✓ CONVERGED' if converged else '✗ NOT converged'}")

print(f"\n{'='*70}")
print(f"  DECISION")
print(f"{'='*70}")
print(f"""
  If L-BFGS tests show errors < 10%:
    → ADOPT L-BFGS for production training
    → Rewrite train.py with L-BFGS optimizer
    → Use two-phase: bending-only → full energy

  If L-BFGS also stalls for GNN (but worked for direct opt):
    → The GNN architecture itself is the bottleneck
    → Consider: deeper per-DOF decoders
    → Consider: skip connections from input loads to output
    → Consider: equilibrium residual loss (avoids K conditioning)

  Key insight: condition number 22M makes Adam IMPOSSIBLE.
  L-BFGS approximates K⁻¹, which is exactly what's needed.
""")