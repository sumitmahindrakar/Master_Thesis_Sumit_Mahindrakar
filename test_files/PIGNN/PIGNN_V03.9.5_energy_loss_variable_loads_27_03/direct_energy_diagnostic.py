"""
direct_energy_diagnostic.py — Comprehensive energy verification
Tests 4 approaches to identify exactly what's broken.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

import os
import time
import torch
import numpy as np
from pathlib import Path

CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)

from step_2_grapg_constr import FrameData

# ── Setup ──
data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
from normalizer import PhysicsScaler
data_list = PhysicsScaler.compute_and_store_list(data_list)
data = data_list[0]

N = data.num_nodes
from energy_loss import FrameEnergyLoss
loss_fn = FrameEnergyLoss()

# ── True solution reference ──
u_true = data.y_node
U_true = loss_fn._strain_energy(u_true, data)
W_true = loss_fn._external_work(u_true, data)
Pi_true = U_true - W_true
E_c = (data.F_c * data.ux_c).item()

print(f"{'='*70}")
print(f"  TRUE SOLUTION REFERENCE")
print(f"{'='*70}")
print(f"  U = {U_true.item():.6e} (strain energy)")
print(f"  W = {W_true.item():.6e} (external work)")
print(f"  Π = {Pi_true.item():.6e} (total potential)")
print(f"  U/W = {(U_true/W_true).item():.6f} (should be 0.5)")
print(f"  E_c = {E_c:.6e}")
print(f"  Π/E_c = {(Pi_true/E_c).item():.6e} (non-dim target)")
print(f"")
print(f"  ux_c = {data.ux_c.item():.6e}")
print(f"  uz_c = {data.uz_c.item():.6e}")
print(f"  θ_c  = {data.theta_c.item():.6e}")
print(f"  ux_c/uz_c = {(data.ux_c/data.uz_c).item():.1f}×")
print(f"")

# True non-dim predictions
pred_true = torch.zeros_like(u_true)
pred_true[:, 0] = u_true[:, 0] / data.ux_c
pred_true[:, 1] = u_true[:, 1] / data.uz_c
pred_true[:, 2] = u_true[:, 2] / data.theta_c
print(f"  True pred_raw ranges:")
print(f"    DOF 0 (ux/ux_c): [{pred_true[:,0].min():.4f}, {pred_true[:,0].max():.4f}]")
print(f"    DOF 1 (uz/uz_c): [{pred_true[:,1].min():.4f}, {pred_true[:,1].max():.4f}]")
print(f"    DOF 2 (θ/θ_c):   [{pred_true[:,2].min():.4f}, {pred_true[:,2].max():.4f}]")

# ── Stiffness conditioning analysis ──
print(f"\n{'='*70}")
print(f"  CONDITIONING ANALYSIS")
print(f"{'='*70}")

EA = data.prop_E * data.prop_A
EI = data.prop_E * data.prop_I22
L = data.elem_lengths

ea_over_L = EA / L
ei_over_L3 = EI / L**3

print(f"  EA/L  range: [{ea_over_L.min():.4e}, {ea_over_L.max():.4e}]")
print(f"  EI/L³ range: [{ei_over_L3.min():.4e}, {ei_over_L3.max():.4e}]")
print(f"  Max ratio:   {(ea_over_L.max()/ei_over_L3.min()).item():.0f}×")
print(f"  ⚠ This is the condition number of the energy Hessian!")


# ════════════════════════════════════════════════════════
# Helper: Direct predictor
# ════════════════════════════════════════════════════════

class DirectPredictor(torch.nn.Module):
    def __init__(self, N, bc_disp, bc_rot):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(N, 3))
        self.bc_disp = bc_disp
        self.bc_rot = bc_rot

    def forward(self, data):
        pred = self.p.clone()
        pred[:, 0:2] = pred[:, 0:2] * (1.0 - self.bc_disp)
        pred[:, 2:3] = pred[:, 2:3] * (1.0 - self.bc_rot)
        return pred


def run_optimization(name, optimizer_type, lr, steps,
                     axial_weight=1.0, use_lbfgs=False):
    """Run direct energy optimization and return history."""
    print(f"\n{'─'*70}")
    print(f"  TEST: {name}")
    print(f"  optimizer={optimizer_type}, lr={lr}, "
          f"steps={steps}, axial_w={axial_weight}")
    print(f"{'─'*70}")

    model = DirectPredictor(N, data.bc_disp, data.bc_rot)

    if use_lbfgs:
        optimizer = torch.optim.LBFGS(
            model.parameters(), lr=lr,
            max_iter=20, line_search_fn='strong_wolfe'
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr
        )

    history = {'step': [], 'Pi': [], 'U_W': [],
               'err_ux': [], 'err_uz': [], 'err_th': []}

    for step in range(steps):

        if use_lbfgs:
            def closure():
                optimizer.zero_grad()
                loss, _, _, _ = loss_fn(
                    model, data, axial_weight=axial_weight
                )
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            Pi_val = loss.item()
            with torch.no_grad():
                _, ld, pr, up = loss_fn(model, data, axial_weight=axial_weight)
        else:
            optimizer.zero_grad()
            loss, ld, pr, up = loss_fn(
                model, data, axial_weight=axial_weight
            )
            if torch.isnan(loss):
                print(f"  ⚠ NaN at step {step}")
                break
            loss.backward()

            # Print gradient norms per DOF
            if step == 0 or (step + 1) == steps:
                g = model.p.grad
                if g is not None:
                    print(f"  Step {step} gradients:")
                    for d in range(3):
                        gd = g[:, d]
                        print(f"    DOF {d}: "
                              f"mean={gd.abs().mean():.4e}, "
                              f"max={gd.abs().max():.4e}")

            optimizer.step()
            Pi_val = ld['Pi']

        # ── Track per-DOF error ──
        with torch.no_grad():
            pred = model(data)
            u_pred = torch.zeros_like(pred)
            u_pred[:, 0] = pred[:, 0] * data.ux_c
            u_pred[:, 1] = pred[:, 1] * data.uz_c
            u_pred[:, 2] = pred[:, 2] * data.theta_c

            err = {}
            for d, name_d in enumerate(['ux', 'uz', 'th']):
                e = (u_pred[:, d] - u_true[:, d]).pow(2).sum().sqrt()
                r = u_true[:, d].pow(2).sum().sqrt().clamp(min=1e-15)
                err[name_d] = (e / r).item()

        history['step'].append(step)
        history['Pi'].append(Pi_val)
        history['U_W'].append(ld['U_over_W'])
        history['err_ux'].append(err['ux'])
        history['err_uz'].append(err['uz'])
        history['err_th'].append(err['th'])

        if step % max(1, steps // 10) == 0 or step == steps - 1:
            print(
                f"  {step:5d}: Π={Pi_val:11.4e}  "
                f"U/W={ld['U_over_W']:.4f}  "
                f"err=[{err['ux']:.3f}, {err['uz']:.3f}, "
                f"{err['th']:.3f}]  "
                f"raw=[{pred.min().item():.4f}, "
                f"{pred.max().item():.4f}]"
            )

    return history


# ════════════════════════════════════════════════════════
# RUN ALL TESTS
# ════════════════════════════════════════════════════════

results = {}

# TEST 1: Original (Adam lr=0.01) — reproduce failure
results['1_adam_0.01'] = run_optimization(
    "Adam lr=0.01 (original)",
    'adam', lr=0.01, steps=2000
)

# TEST 2: Adam with much lower lr
results['2_adam_0.001'] = run_optimization(
    "Adam lr=0.001 (lower)",
    'adam', lr=0.001, steps=5000
)

# TEST 3: Adam lr=0.001 with NO axial (bending only)
results['3_bending_only'] = run_optimization(
    "Adam lr=0.001, bending only (axial_w=0)",
    'adam', lr=0.001, steps=3000, axial_weight=0.0
)

# TEST 4: L-BFGS (proper second-order optimizer)
results['4_lbfgs'] = run_optimization(
    "L-BFGS (second-order)",
    'lbfgs', lr=1.0, steps=200, use_lbfgs=True
)

# TEST 5: Adam with axial curriculum
print(f"\n{'─'*70}")
print(f"  TEST 5: Adam lr=0.001 with axial curriculum")
print(f"{'─'*70}")

model5 = DirectPredictor(N, data.bc_disp, data.bc_rot)
opt5 = torch.optim.Adam(model5.parameters(), lr=0.001)
h5 = {'step': [], 'Pi': [], 'U_W': [],
       'err_ux': [], 'err_uz': [], 'err_th': []}

total_steps = 5000
ramp_end = 2000

for step in range(total_steps):
    # Axial curriculum
    if step < ramp_end:
        aw = step / ramp_end
    else:
        aw = 1.0

    opt5.zero_grad()
    loss, ld, pr, up = loss_fn(model5, data, axial_weight=aw)
    if torch.isnan(loss):
        break
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model5.parameters(), 10.0)
    opt5.step()

    with torch.no_grad():
        pred = model5(data)
        u_pred = torch.zeros_like(pred)
        u_pred[:, 0] = pred[:, 0] * data.ux_c
        u_pred[:, 1] = pred[:, 1] * data.uz_c
        u_pred[:, 2] = pred[:, 2] * data.theta_c

        err = {}
        for d, nm in enumerate(['ux', 'uz', 'th']):
            e = (u_pred[:, d] - u_true[:, d]).pow(2).sum().sqrt()
            r = u_true[:, d].pow(2).sum().sqrt().clamp(min=1e-15)
            err[nm] = (e / r).item()

    h5['step'].append(step)
    h5['Pi'].append(ld['Pi'])
    h5['U_W'].append(ld['U_over_W'])
    h5['err_ux'].append(err['ux'])
    h5['err_uz'].append(err['uz'])
    h5['err_th'].append(err['th'])

    if step % 500 == 0 or step == total_steps - 1:
        print(
            f"  {step:5d}: Π={ld['Pi']:11.4e}  "
            f"U/W={ld['U_over_W']:.4f}  "
            f"axial_w={aw:.2f}  "
            f"err=[{err['ux']:.3f}, {err['uz']:.3f}, "
            f"{err['th']:.3f}]"
        )

results['5_curriculum'] = h5


# ════════════════════════════════════════════════════════
# PLOT COMPARISON
# ════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Direct Energy Optimization: Diagnostic Comparison',
             fontsize=14, fontweight='bold')

target_Pi = (Pi_true / E_c).item()

colors = {
    '1_adam_0.01': 'red',
    '2_adam_0.001': 'blue',
    '3_bending_only': 'green',
    '4_lbfgs': 'purple',
    '5_curriculum': 'orange',
}
labels = {
    '1_adam_0.01': 'Adam lr=0.01',
    '2_adam_0.001': 'Adam lr=0.001',
    '3_bending_only': 'Bending only',
    '4_lbfgs': 'L-BFGS',
    '5_curriculum': 'Curriculum',
}

# 1. Energy convergence
ax = axes[0, 0]
for key, h in results.items():
    ax.plot(h['step'], h['Pi'], color=colors[key],
            label=labels[key], linewidth=1.5)
ax.axhline(y=target_Pi, color='black', linestyle='--',
           linewidth=2, label=f'Target: {target_Pi:.4f}')
ax.set_title('Π/E_c convergence')
ax.set_xlabel('Step')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2. U/W ratio
ax = axes[0, 1]
for key, h in results.items():
    uw = np.clip(h['U_W'], 0, 2)
    ax.plot(h['step'], uw, color=colors[key],
            label=labels[key], linewidth=1.5)
ax.axhline(y=0.5, color='black', linestyle='--',
           label='Target: 0.5')
ax.set_title('U/|W| → 0.5')
ax.set_ylim(0, 2)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 3-5. Per-DOF errors
for d, (dof_name, ax_idx) in enumerate([
    ('ux', (0, 2)), ('uz', (1, 0)), ('θ', (1, 1))
]):
    ax = axes[ax_idx]
    key_d = ['err_ux', 'err_uz', 'err_th'][d]
    for key, h in results.items():
        ax.semilogy(h['step'], h[key_d], color=colors[key],
                    label=labels[key], linewidth=1.5)
    ax.axhline(y=0.05, color='gray', linestyle=':',
               label='5% error')
    ax.set_title(f'{dof_name} relative error')
    ax.set_ylim(1e-4, 2)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# 6. Summary
ax = axes[1, 2]
ax.axis('off')
summary_text = "SUMMARY\n" + "─" * 35 + "\n"
for key, h in results.items():
    final_pi = h['Pi'][-1] if h['Pi'] else float('nan')
    final_ux = h['err_ux'][-1] if h['err_ux'] else float('nan')
    final_uz = h['err_uz'][-1] if h['err_uz'] else float('nan')
    final_th = h['err_th'][-1] if h['err_th'] else float('nan')
    summary_text += (
        f"\n{labels[key]}:\n"
        f"  Π/E_c = {final_pi:.4e}\n"
        f"  err: [{final_ux:.3f}, {final_uz:.3f}, {final_th:.3f}]\n"
    )
summary_text += f"\nTarget Π/E_c = {target_Pi:.4e}"
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow'))

plt.tight_layout()
plt.savefig('RESULTS/direct_energy_diagnostic.png',
            dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved: RESULTS/direct_energy_diagnostic.png")

# ════════════════════════════════════════════════════════
# DIAGNOSIS
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  DIAGNOSIS")
print(f"{'='*70}")

for key, h in results.items():
    pi_final = h['Pi'][-1]
    pi_best = min(h['Pi'])
    ux_final = h['err_ux'][-1]
    converged = abs(pi_final - target_Pi) / abs(target_Pi) < 0.05

    print(f"\n  {labels[key]}:")
    print(f"    Final Π/E_c:  {pi_final:.4e} "
          f"(target: {target_Pi:.4e})")
    print(f"    Best Π/E_c:   {pi_best:.4e}")
    print(f"    ux err:       {ux_final:.4f}")
    print(f"    {'✓ CONVERGED' if converged else '✗ FAILED'}")

print(f"\n{'='*70}")
print(f"  NEXT STEPS")
print(f"{'='*70}")
print(f"""
  If L-BFGS converges but Adam doesn't:
    → Energy formula is CORRECT
    → Problem is optimization conditioning
    → Fix: use L-BFGS-inspired preconditioning in GNN training

  If bending-only converges but full doesn't:
    → Axial stiffness creates ill-conditioning
    → Fix: longer axial curriculum ramp (5000+ steps)
    → Fix: normalize axial/bending contributions separately

  If NOTHING converges:
    → Energy formula may have a bug
    → Check: rotation sign convention (th_loc = -th)
    → Check: coordinate transformation (c, s ordering)

  If curriculum converges:
    → Use same curriculum in GNN training
    → Start axial_ramp MUCH later (after bending converges)
""")