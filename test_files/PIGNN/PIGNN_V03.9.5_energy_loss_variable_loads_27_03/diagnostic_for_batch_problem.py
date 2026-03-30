"""
diagnose.py — Verify data + energy loss are correct
"""

import torch
import os
from pathlib import Path

CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)

print("=" * 70)
print("  DIAGNOSTIC: Data + Energy Verification")
print("=" * 70)

# ── Load both datasets ──
raw_data = torch.load("DATA/graph_dataset.pt", weights_only=False)
norm_data = torch.load("DATA/graph_dataset_norm.pt", weights_only=False)

from normalizer import PhysicsScaler
if not hasattr(raw_data[0], 'u_c'):
    raw_data = PhysicsScaler.compute_and_store_list(raw_data)
if not hasattr(norm_data[0], 'u_c'):
    norm_data = PhysicsScaler.compute_and_store_list(norm_data)

from energy_loss import FrameEnergyLoss
loss_fn = FrameEnergyLoss()

# ══════════════════════════════════════════
# TEST 1: Connectivity check
# ══════════════════════════════════════════
print(f"\n{'─'*70}")
print(f"  TEST 1: CONNECTIVITY")
print(f"{'─'*70}")

d = raw_data[0]
conn = d.connectivity
coords = d.coords
N = d.num_nodes
E = d.n_elements

print(f"  Nodes: {N}, Elements: {E}")
print(f"  Connectivity range: [{conn.min()}, {conn.max()}]")
print(f"  Valid range: [0, {N-1}]")
ok = conn.min() >= 0 and conn.max() < N
print(f"  {'✓' if ok else '✗'} Connectivity indices valid")

# Check element lengths match
for e in range(min(5, E)):
    nA, nB = conn[e]
    computed_L = (coords[nB] - coords[nA]).norm().item()
    stored_L = d.elem_lengths[e].item()
    match = abs(computed_L - stored_L) < 1e-4
    print(f"  Elem {e}: nodes [{nA},{nB}], "
          f"L_computed={computed_L:.4f}, "
          f"L_stored={stored_L:.4f} "
          f"{'✓' if match else '✗'}")

# ══════════════════════════════════════════
# TEST 2: Energy with TRUE displacements
# ══════════════════════════════════════════
print(f"\n{'─'*70}")
print(f"  TEST 2: ENERGY WITH TRUE DISPLACEMENTS (raw data)")
print(f"{'─'*70}")

all_ok = True
for i in range(min(10, len(raw_data))):
    d = raw_data[i]
    u_true = d.y_node

    U = loss_fn._strain_energy(u_true, d)
    W = loss_fn._external_work(u_true, d)
    Pi = U - W
    E_c = (d.F_c * d.u_c).clamp(min=1e-30)
    Pi_norm = (Pi / E_c).item()
    UW = (U / W.abs().clamp(min=1e-30)).item()

    ok = (Pi.item() < 0 and 0.4 < UW < 0.6)
    if not ok:
        all_ok = False

    print(f"  Case {i:3d}: U={U.item():12.4e}  "
          f"W={W.item():12.4e}  "
          f"Π={Pi.item():12.4e}  "
          f"U/|W|={UW:.4f}  "
          f"Π_norm={Pi_norm:8.4f}  "
          f"{'✓' if ok else '✗ BAD'}")

print(f"\n  {'✓ ALL OK' if all_ok else '✗ SOME FAILED'}")

# ══════════════════════════════════════════
# TEST 3: Energy with NORMALIZED data + scales
# ══════════════════════════════════════════
print(f"\n{'─'*70}")
print(f"  TEST 3: ENERGY THROUGH FULL PIPELINE (norm data)")
print(f"  (Simulates what training does)")
print(f"{'─'*70}")

for i in range(min(5, len(norm_data))):
    d_norm = norm_data[i]
    d_raw = raw_data[i]

    # Simulate model outputting perfect predictions
    u_true_raw = d_raw.y_node  # (N, 3) physical

    # What the model should output (non-dimensional)
    pred_perfect = torch.zeros_like(u_true_raw)
    pred_perfect[:, 0] = u_true_raw[:, 0] / d_norm.u_c
    pred_perfect[:, 1] = u_true_raw[:, 1] / d_norm.u_c
    pred_perfect[:, 2] = u_true_raw[:, 2] / d_norm.theta_c

    # Convert back to physical (what loss function does)
    u_phys = torch.zeros_like(pred_perfect)
    u_phys[:, 0] = pred_perfect[:, 0] * d_norm.u_c
    u_phys[:, 1] = pred_perfect[:, 1] * d_norm.u_c
    u_phys[:, 2] = pred_perfect[:, 2] * d_norm.theta_c

    # Check roundtrip
    roundtrip_err = (u_phys - u_true_raw).abs().max().item()

    # Compute energy using NORMALIZED data's properties
    U = loss_fn._strain_energy(u_phys, d_norm)
    W = loss_fn._external_work(u_phys, d_norm)
    Pi = U - W
    UW = (U / W.abs().clamp(min=1e-30)).item()

    # Compare with raw data energy
    U_raw = loss_fn._strain_energy(u_true_raw, d_raw)
    W_raw = loss_fn._external_work(u_true_raw, d_raw)
    Pi_raw = U_raw - W_raw

    energy_match = abs(Pi.item() - Pi_raw.item()) / abs(Pi_raw.item()) < 0.01

    print(f"\n  Case {i}:")
    print(f"    Roundtrip error: {roundtrip_err:.2e}")
    print(f"    pred_perfect range: [{pred_perfect.min():.4f}, "
          f"{pred_perfect.max():.4f}]")
    print(f"    Raw:  U={U_raw.item():.4e}, W={W_raw.item():.4e}, "
          f"Π={Pi_raw.item():.4e}")
    print(f"    Norm: U={U.item():.4e}, W={W.item():.4e}, "
          f"Π={Pi.item():.4e}")
    print(f"    U/|W|={UW:.4f}")
    print(f"    Energy match: {'✓' if energy_match else '✗ MISMATCH!'}")

# ══════════════════════════════════════════
# TEST 4: Gradient direction check
# ══════════════════════════════════════════
print(f"\n{'─'*70}")
print(f"  TEST 4: GRADIENT DIRECTION AT u=0")
print(f"{'─'*70}")

for i in range(min(3, len(raw_data))):
    d = raw_data[i]
    u = torch.zeros(d.num_nodes, 3, requires_grad=True)

    U = loss_fn._strain_energy(u, d)
    W = loss_fn._external_work(u, d)
    Pi = U - W
    Pi.backward()

    grad = u.grad

    # At u=0: ∂Π/∂u = -F_ext (since ∂U/∂u = Ku = 0)
    print(f"\n  Case {i}:")
    print(f"    Π(0) = {Pi.item():.6e} (should be 0 or slightly negative)")
    print(f"    |∇Π| = {grad.norm().item():.4e}")

    for dof, name in [(0, 'ux/Fx'), (1, 'uz/Fz'), (2, 'θ/My')]:
        g = grad[:, dof]
        f = -d.F_ext[:, dof]
        loaded = f.abs() > 1e-10
        if loaded.any():
            match = torch.allclose(g[loaded], f[loaded], rtol=1e-3, atol=1e-6)
            max_err = (g[loaded] - f[loaded]).abs().max().item()
            print(f"    {name}: ∂Π/∂u = -F? {'✓' if match else '✗'} "
                  f"(max_err={max_err:.4e})")

# ══════════════════════════════════════════
# TEST 5: Stiffness magnitude check
# ══════════════════════════════════════════
print(f"\n{'─'*70}")
print(f"  TEST 5: STIFFNESS MAGNITUDES")
print(f"{'─'*70}")

d = raw_data[0]
EA = d.prop_E * d.prop_A
EI = d.prop_E * d.prop_I22
L = d.elem_lengths

ea_L = EA / L
ei_L3 = EI / L**3

print(f"  EA/L  range: [{ea_L.min():.4e}, {ea_L.max():.4e}]")
print(f"  EI/L³ range: [{ei_L3.min():.4e}, {ei_L3.max():.4e}]")
print(f"  Ratio EA/L ÷ EI/L³: {(ea_L.max()/ei_L3.max()).item():.1f}×")
print(f"  F_c = {d.F_c.item():.4e}")
print(f"  u_c = {d.u_c.item():.4e}")
print(f"  E_c = F_c*u_c = {(d.F_c*d.u_c).item():.4e}")
print(f"  Max K entry (EA/L): {ea_L.max().item():.4e}")
print(f"  K/E_c ratio: {(ea_L.max()/(d.F_c*d.u_c)).item():.1e}")
print(f"  ⚠ This ratio shows how many orders of magnitude")
print(f"    the stiffness exceeds the energy scale!")

# ══════════════════════════════════════════
# TEST 6: Normalized data properties match raw
# ══════════════════════════════════════════
print(f"\n{'─'*70}")
print(f"  TEST 6: NORM vs RAW PROPERTY CHECK")
print(f"{'─'*70}")

d_raw = raw_data[0]
d_norm = norm_data[0]

checks = [
    ('prop_E',    d_raw.prop_E,    d_norm.prop_E),
    ('prop_A',    d_raw.prop_A,    d_norm.prop_A),
    ('prop_I22',  d_raw.prop_I22,  d_norm.prop_I22),
    ('elem_lengths', d_raw.elem_lengths, d_norm.elem_lengths),
    ('F_ext',     d_raw.F_ext,     d_norm.F_ext),
    ('connectivity', d_raw.connectivity, d_norm.connectivity),
    ('bc_disp',   d_raw.bc_disp,   d_norm.bc_disp),
    ('bc_rot',    d_raw.bc_rot,    d_norm.bc_rot),
    ('elem_directions', d_raw.elem_directions, d_norm.elem_directions),
]

for name, raw_val, norm_val in checks:
    match = torch.equal(raw_val, norm_val)
    print(f"  {name:<18} {'✓ SAME' if match else '✗ DIFFERENT!'}")
    if not match:
        print(f"    Raw:  [{raw_val.min():.4e}, {raw_val.max():.4e}]")
        print(f"    Norm: [{norm_val.min():.4e}, {norm_val.max():.4e}]")

# ══════════════════════════════════════════
# TEST 7: Model forward pass sanity
# ══════════════════════════════════════════
print(f"\n{'─'*70}")
print(f"  TEST 7: MODEL FORWARD PASS")
print(f"{'─'*70}")

from model import PIGNN

model = PIGNN(node_in_dim=10, edge_in_dim=7, hidden_dim=128, n_layers=6)

with torch.no_grad():
    d = norm_data[0]
    pred = model(d)
    print(f"  pred shape: {pred.shape}")
    print(f"  pred range: [{pred.min():.6e}, {pred.max():.6e}]")
    print(f"  pred norm:  {pred.norm():.6e}")

    # Check BC enforcement
    bc_nodes = (d.bc_disp.squeeze() > 0.5).nonzero().squeeze()
    if bc_nodes.numel() > 0:
        bc_pred = pred[bc_nodes]
        print(f"  BC nodes {bc_nodes.tolist()}: "
              f"max pred = {bc_pred.abs().max():.6e} (should be 0)")

    # Energy at zero prediction
    u_phys = torch.zeros_like(pred)
    U_zero = loss_fn._strain_energy(u_phys, d)
    W_zero = loss_fn._external_work(u_phys, d)
    print(f"  Π(0) = U-W = {U_zero.item():.6e} - {W_zero.item():.6e} "
          f"= {(U_zero-W_zero).item():.6e}")

print(f"\n{'═'*70}")
print(f"  DIAGNOSTIC COMPLETE")
print(f"{'═'*70}")