"""
energy_fix.py — Fix energy loss for frame structures
=====================================================
Tests:
  1. K assembly with CORRECT θ sign → verify Ku=F matches Kratos
  2. Float64 L-BFGS → verify precision fix
  3. Matrix-form energy (not expanded) → numerical stability
  4. Per-DOF preconditioning → condition number fix
"""

import torch
import numpy as np

import os
import time
import numpy as np
from pathlib import Path

CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)

from step_2_grapg_constr import FrameData

# ── Load data ──
data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
from normalizer import PhysicsScaler
data_list = PhysicsScaler.compute_and_store_list(data_list)
data = data_list[0]

from energy_loss import FrameEnergyLoss
loss_fn = FrameEnergyLoss()

N = data.num_nodes
u_true = data.y_node

print(f"{'='*70}")
print(f"  ENERGY FIX VERIFICATION")
print(f"{'='*70}")


# ════════════════════════════════════════════════════════
# TEST 1: K assembly with CORRECT sign convention
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  TEST 1: K assembly with θ_loc = -θ_global (XZ plane)")
print(f"{'='*70}")

def assemble_K_F_corrected(data):
    """
    Assemble K with CORRECT sign convention for XZ plane.
    
    Key difference from standard XY formulation:
      θ_loc = -θ_y  (negative because XZ plane flips rotation sense)
    
    This is implemented as T[2,2] = -1, T[5,5] = -1
    """
    conn = data.connectivity.numpy()
    n_nodes = data.num_nodes
    n_dof = 3 * n_nodes

    K_global = np.zeros((n_dof, n_dof))
    F_global = np.zeros(n_dof)

    L = data.elem_lengths.numpy()
    EA = (data.prop_E * data.prop_A).numpy()
    EI = (data.prop_E * data.prop_I22).numpy()
    dirs = data.elem_directions.numpy()

    for e in range(len(conn)):
        nA, nB = conn[e]
        c = dirs[e, 0]
        s = dirs[e, 2]
        Le = L[e]

        ea_l = EA[e] / Le
        ei_l = EI[e] / Le
        ei_l2 = EI[e] / Le**2
        ei_l3 = EI[e] / Le**3

        # Local stiffness (standard)
        K_loc = np.zeros((6, 6))
        K_loc[0,0] =  ea_l;  K_loc[0,3] = -ea_l
        K_loc[3,0] = -ea_l;  K_loc[3,3] =  ea_l
        K_loc[1,1] =  12*ei_l3; K_loc[1,2] =  6*ei_l2
        K_loc[1,4] = -12*ei_l3; K_loc[1,5] =  6*ei_l2
        K_loc[2,1] =  6*ei_l2;  K_loc[2,2] =  4*ei_l
        K_loc[2,4] = -6*ei_l2;  K_loc[2,5] =  2*ei_l
        K_loc[4,1] = -12*ei_l3; K_loc[4,2] = -6*ei_l2
        K_loc[4,4] =  12*ei_l3; K_loc[4,5] = -6*ei_l2
        K_loc[5,1] =  6*ei_l2;  K_loc[5,2] =  2*ei_l
        K_loc[5,4] = -6*ei_l2;  K_loc[5,5] =  4*ei_l

        # ═══ CORRECTED transformation for XZ plane ═══
        T = np.zeros((6, 6))
        T[0,0] =  c;  T[0,1] = s
        T[1,0] = -s;  T[1,1] = c
        T[2,2] = -1.0   # ← KEY FIX: θ_loc = -θ_y
        T[3,3] =  c;  T[3,4] = s
        T[4,3] = -s;  T[4,4] = c
        T[5,5] = -1.0   # ← KEY FIX

        K_glob = T.T @ K_loc @ T

        dofs = [3*nA, 3*nA+1, 3*nA+2,
                3*nB, 3*nB+1, 3*nB+2]
        for i in range(6):
            for j in range(6):
                K_global[dofs[i], dofs[j]] += K_glob[i, j]

    # Force vector
    F_ext = data.F_ext.numpy()
    for n in range(n_nodes):
        F_global[3*n]   = F_ext[n, 0]
        F_global[3*n+1] = F_ext[n, 1]
        F_global[3*n+2] = F_ext[n, 2]

    return K_global, F_global


K, F = assemble_K_F_corrected(data)

# Identify fixed DOFs
bc_disp = data.bc_disp.numpy().flatten()
bc_rot = data.bc_rot.numpy().flatten()

fixed_dofs = []
for n in range(N):
    if bc_disp[n] > 0.5:
        fixed_dofs.extend([3*n, 3*n+1])
    if bc_rot[n] > 0.5:
        fixed_dofs.append(3*n+2)

all_dofs = list(range(3*N))
free_dofs = [d for d in all_dofs if d not in fixed_dofs]

K_ff = K[np.ix_(free_dofs, free_dofs)]
F_f = F[free_dofs]

# Condition number
eigvals = np.linalg.eigvalsh(K_ff)
cond = eigvals.max() / eigvals.min()

print(f"  Fixed DOFs: {len(fixed_dofs)}, Free DOFs: {len(free_dofs)}")
print(f"  K eigenvalues: [{eigvals.min():.4e}, {eigvals.max():.4e}]")
print(f"  Condition number: {cond:.0f}")

# Solve
u_solved = np.zeros(3*N)
u_solved[free_dofs] = np.linalg.solve(K_ff, F_f)
u_solved_2d = u_solved.reshape(N, 3)
u_true_np = u_true.numpy()

print(f"\n  Direct solve (corrected θ) vs Kratos:")
all_match = True
for dof, name in enumerate(['ux', 'uz', 'θ']):
    err = np.linalg.norm(u_solved_2d[:, dof] - u_true_np[:, dof])
    ref = np.linalg.norm(u_true_np[:, dof])
    rel = err / max(ref, 1e-15)
    ok = rel < 1e-3
    if not ok:
        all_match = False
    print(f"    {name}: rel_err = {rel:.6e}  "
          f"{'✓' if ok else '✗'}  "
          f"solved=[{u_solved_2d[:,dof].min():.6e}, "
          f"{u_solved_2d[:,dof].max():.6e}]")

# Energy check
u_solved_t = torch.tensor(u_solved_2d, dtype=torch.float32)
U_sol = loss_fn._strain_energy(u_solved_t, data)
W_sol = loss_fn._external_work(u_solved_t, data)
print(f"\n  Energy at corrected solution:")
print(f"    U = {U_sol.item():.6e}")
print(f"    W = {W_sol.item():.6e}")
print(f"    U/W = {(U_sol/W_sol).item():.6f} (should be 0.5)")
print(f"    {'✓' if abs((U_sol/W_sol).item() - 0.5) < 0.01 else '✗'} "
      f"Energy balance")

if all_match:
    print(f"\n  ✓ K assembly VERIFIED — sign convention is correct!")
    print(f"    θ_loc = -θ_y for XZ plane frames")
else:
    print(f"\n  ✗ K assembly still has issues")


# ════════════════════════════════════════════════════════
# TEST 2: Float64 Matrix-Form Energy + L-BFGS
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  TEST 2: Float64 matrix-form L-BFGS")
print(f"{'='*70}")

def matrix_energy_loss(u_phys, data64):
    """
    Energy using MATRIX form (d^T K d), computed in float64.
    This avoids numerical issues in the expanded formula.
    """
    conn = data64['connectivity']
    nA, nB = conn[:, 0].long(), conn[:, 1].long()
    n_elem = conn.shape[0]

    L  = data64['elem_lengths']
    EA = data64['prop_E'] * data64['prop_A']
    EI = data64['prop_E'] * data64['prop_I22']
    c  = data64['elem_directions'][:, 0]
    s  = data64['elem_directions'][:, 2]

    # Global to local transformation
    ux_A = u_phys[nA, 0]; uz_A = u_phys[nA, 1]; th_A = u_phys[nA, 2]
    ux_B = u_phys[nB, 0]; uz_B = u_phys[nB, 1]; th_B = u_phys[nB, 2]

    u_A_loc =  c * ux_A + s * uz_A
    w_A_loc = -s * ux_A + c * uz_A
    u_B_loc =  c * ux_B + s * uz_B
    w_B_loc = -s * ux_B + c * uz_B
    th_A_loc = -th_A    # XZ plane convention
    th_B_loc = -th_B

    d_local = torch.stack([
        u_A_loc, w_A_loc, th_A_loc,
        u_B_loc, w_B_loc, th_B_loc
    ], dim=1)  # (E, 6)

    # Local stiffness matrices
    ea_l = EA/L; ei_l = EI/L
    ei_l2 = EI/L**2; ei_l3 = EI/L**3

    K_loc = torch.zeros(n_elem, 6, 6, dtype=torch.float64)
    K_loc[:,0,0] =  ea_l;  K_loc[:,0,3] = -ea_l
    K_loc[:,3,0] = -ea_l;  K_loc[:,3,3] =  ea_l
    K_loc[:,1,1] =  12*ei_l3; K_loc[:,1,2] =  6*ei_l2
    K_loc[:,1,4] = -12*ei_l3; K_loc[:,1,5] =  6*ei_l2
    K_loc[:,2,1] =  6*ei_l2;  K_loc[:,2,2] =  4*ei_l
    K_loc[:,2,4] = -6*ei_l2;  K_loc[:,2,5] =  2*ei_l
    K_loc[:,4,1] = -12*ei_l3; K_loc[:,4,2] = -6*ei_l2
    K_loc[:,4,4] =  12*ei_l3; K_loc[:,4,5] = -6*ei_l2
    K_loc[:,5,1] =  6*ei_l2;  K_loc[:,5,2] =  2*ei_l
    K_loc[:,5,4] = -6*ei_l2;  K_loc[:,5,5] =  4*ei_l

    # U = ½ d^T K d per element
    Kd = torch.bmm(K_loc, d_local.unsqueeze(2))
    U_per_elem = 0.5 * torch.bmm(
        d_local.unsqueeze(1), Kd
    ).squeeze()
    U = U_per_elem.sum()

    # External work
    F_ext = data64['F_ext']
    W = (F_ext[:, 0] * u_phys[:, 0]
       + F_ext[:, 1] * u_phys[:, 1]
       + F_ext[:, 2] * u_phys[:, 2]).sum()

    return U - W, U, W


# Convert data to float64
data64 = {
    'connectivity': data.connectivity,
    'elem_lengths': data.elem_lengths.double(),
    'prop_E': data.prop_E.double(),
    'prop_A': data.prop_A.double(),
    'prop_I22': data.prop_I22.double(),
    'elem_directions': data.elem_directions.double(),
    'F_ext': data.F_ext.double(),
    'bc_disp': data.bc_disp.double(),
    'bc_rot': data.bc_rot.double(),
}

# Verify at true solution
u_true64 = u_true.double()
Pi_t, U_t, W_t = matrix_energy_loss(u_true64, data64)
print(f"  True solution (float64 matrix form):")
print(f"    U = {U_t.item():.10e}")
print(f"    W = {W_t.item():.10e}")
print(f"    Π = {Pi_t.item():.10e}")
print(f"    U/W = {(U_t/W_t).item():.8f}")

# ── L-BFGS optimization in float64 ──
u_opt = torch.zeros(N, 3, dtype=torch.float64, requires_grad=True)

optimizer = torch.optim.LBFGS(
    [u_opt], lr=1.0,
    max_iter=100,          # more inner iterations
    history_size=100,      # ≥ number of free DOFs
    line_search_fn='strong_wolfe',
    tolerance_grad=1e-14,
    tolerance_change=1e-16,
)

print(f"\n  L-BFGS float64 optimization:")
bc_d = data64['bc_disp']
bc_r = data64['bc_rot']

best_err = [1.0, 1.0, 1.0]

for step in range(200):
    def closure():
        optimizer.zero_grad()
        u = u_opt.clone()
        u[:, 0:2] = u[:, 0:2] * (1.0 - bc_d)
        u[:, 2:3] = u[:, 2:3] * (1.0 - bc_r)
        Pi, _, _ = matrix_energy_loss(u, data64)
        Pi.backward()
        return Pi

    loss_val = optimizer.step(closure)

    with torch.no_grad():
        u_pred = u_opt.clone()
        u_pred[:, 0:2] *= (1.0 - bc_d)
        u_pred[:, 2:3] *= (1.0 - bc_r)

        errs = []
        for d in range(3):
            e = (u_pred[:, d] - u_true64[:, d]).pow(2).sum().sqrt()
            r = u_true64[:, d].pow(2).sum().sqrt().clamp(min=1e-30)
            errs.append((e/r).item())

        g = u_opt.grad
        gn = g.abs().max().item() if g is not None else 0

        for d in range(3):
            best_err[d] = min(best_err[d], errs[d])

    if step % 10 == 0 or step < 5 or gn < 1e-8:
        print(f"  Step {step:3d}: Π={loss_val.item():14.8e}  "
              f"err=[{errs[0]:.6f}, {errs[1]:.6f}, {errs[2]:.6f}]  "
              f"|∇|={gn:.2e}")

    if gn < 1e-10:
        print(f"  ✓ Converged at step {step}!")
        break

print(f"\n  Best errors: [{best_err[0]:.6f}, {best_err[1]:.6f}, "
      f"{best_err[2]:.6f}]")

# Final comparison
with torch.no_grad():
    u_final = u_opt.clone()
    u_final[:, 0:2] *= (1.0 - bc_d)
    u_final[:, 2:3] *= (1.0 - bc_r)
    Pi_f, U_f, W_f = matrix_energy_loss(u_final, data64)

print(f"\n  Final solution:")
print(f"    U = {U_f.item():.10e}")
print(f"    W = {W_f.item():.10e}")
print(f"    U/W = {(U_f/W_f).item():.8f}")
print(f"    Π = {Pi_f.item():.10e}")
print(f"    Target Π = {Pi_t.item():.10e}")


# ════════════════════════════════════════════════════════
# TEST 3: Float64 Adam (GNN-relevant)
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  TEST 3: Float64 Adam (GNN-relevant)")
print(f"{'='*70}")

u_adam = torch.zeros(N, 3, dtype=torch.float64, requires_grad=True)
opt_adam = torch.optim.Adam([u_adam], lr=1e-4)

print(f"  Adam lr=1e-4, float64:")
for step in range(20000):
    opt_adam.zero_grad()
    u = u_adam.clone()
    u[:, 0:2] = u[:, 0:2] * (1.0 - bc_d)
    u[:, 2:3] = u[:, 2:3] * (1.0 - bc_r)
    Pi, U_val, W_val = matrix_energy_loss(u, data64)
    Pi.backward()
    torch.nn.utils.clip_grad_norm_([u_adam], 1.0)
    opt_adam.step()

    if step % 2000 == 0 or step == 19999:
        with torch.no_grad():
            u_pred = u_adam.clone()
            u_pred[:, 0:2] *= (1.0 - bc_d)
            u_pred[:, 2:3] *= (1.0 - bc_r)
            errs = []
            for d in range(3):
                e = (u_pred[:, d] - u_true64[:, d]).pow(2).sum().sqrt()
                r = u_true64[:, d].pow(2).sum().sqrt().clamp(min=1e-30)
                errs.append((e/r).item())
            g = u_adam.grad
            gn = [g[:, d].abs().mean().item() for d in range(3)]
        print(f"  {step:5d}: Π={Pi.item():11.4e}  "
              f"U/W={U_val.item()/max(W_val.item(),1e-30):.4f}  "
              f"err=[{errs[0]:.4f}, {errs[1]:.4f}, {errs[2]:.4f}]  "
              f"|∇|=[{gn[0]:.2e}, {gn[1]:.2e}, {gn[2]:.2e}]")


# ════════════════════════════════════════════════════════
# TEST 4: Preconditioned Adam (per-DOF learning rate)
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  TEST 4: Preconditioned Adam (diagonal K scaling)")
print(f"{'='*70}")

# Compute diagonal stiffness for preconditioning
K_diag = np.diag(K).reshape(N, 3)  # (N, 3)
K_scale = np.sqrt(np.abs(K_diag) + 1e-15)

# Normalize so max = 1
K_scale = K_scale / K_scale.max()
K_scale_t = torch.tensor(K_scale, dtype=torch.float64)

print(f"  K diagonal scales:")
print(f"    ux: [{K_scale_t[:,0].min():.4e}, {K_scale_t[:,0].max():.4e}]")
print(f"    uz: [{K_scale_t[:,1].min():.4e}, {K_scale_t[:,1].max():.4e}]")
print(f"    θ:  [{K_scale_t[:,2].min():.4e}, {K_scale_t[:,2].max():.4e}]")

# Optimize scaled variables: u_phys = u_scaled / K_scale
u_scaled = torch.zeros(N, 3, dtype=torch.float64, requires_grad=True)
opt_prec = torch.optim.Adam([u_scaled], lr=1e-3)

print(f"\n  Preconditioned Adam lr=1e-3, float64:")
for step in range(20000):
    opt_prec.zero_grad()

    # Unscale to physical
    u = u_scaled / K_scale_t.clamp(min=1e-10)
    u[:, 0:2] = u[:, 0:2] * (1.0 - bc_d)
    u[:, 2:3] = u[:, 2:3] * (1.0 - bc_r)

    Pi, U_val, W_val = matrix_energy_loss(u, data64)
    Pi.backward()
    torch.nn.utils.clip_grad_norm_([u_scaled], 10.0)
    opt_prec.step()

    if step % 2000 == 0 or step == 19999:
        with torch.no_grad():
            u_pred = (u_scaled / K_scale_t.clamp(min=1e-10)).clone()
            u_pred[:, 0:2] *= (1.0 - bc_d)
            u_pred[:, 2:3] *= (1.0 - bc_r)
            errs = []
            for d in range(3):
                e = (u_pred[:, d] - u_true64[:, d]).pow(2).sum().sqrt()
                r = u_true64[:, d].pow(2).sum().sqrt().clamp(min=1e-30)
                errs.append((e/r).item())
        print(f"  {step:5d}: Π={Pi.item():11.4e}  "
              f"U/W={U_val.item()/max(W_val.item(),1e-30):.4f}  "
              f"err=[{errs[0]:.4f}, {errs[1]:.4f}, {errs[2]:.4f}]")


# ════════════════════════════════════════════════════════
# TEST 5: Per-DOF separate optimizers
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  TEST 5: Per-DOF separate learning rates")
print(f"{'='*70}")

u_dof = torch.zeros(N, 3, dtype=torch.float64, requires_grad=True)

# Compute per-DOF learning rates inversely proportional to K_diag
K_diag_max = np.abs(K_diag).max(axis=0)  # max per DOF type
lr_per_dof = 1e-3 / (K_diag_max / K_diag_max.min())
print(f"  K_diag max: ux={K_diag_max[0]:.4e}, "
      f"uz={K_diag_max[1]:.4e}, θ={K_diag_max[2]:.4e}")
print(f"  LR per DOF: ux={lr_per_dof[0]:.4e}, "
      f"uz={lr_per_dof[1]:.4e}, θ={lr_per_dof[2]:.4e}")

opt_dof = torch.optim.Adam([u_dof], lr=1.0)  # lr=1 because we scale manually

lr_t = torch.tensor(lr_per_dof, dtype=torch.float64).unsqueeze(0)  # (1, 3)

for step in range(20000):
    opt_dof.zero_grad()

    u = u_dof.clone()
    u[:, 0:2] = u[:, 0:2] * (1.0 - bc_d)
    u[:, 2:3] = u[:, 2:3] * (1.0 - bc_r)

    Pi, U_val, W_val = matrix_energy_loss(u, data64)
    Pi.backward()

    # Manual per-DOF LR scaling
    with torch.no_grad():
        if u_dof.grad is not None:
            u_dof.grad *= lr_t
            torch.nn.utils.clip_grad_norm_([u_dof], 1e-4)

    opt_dof.step()

    if step % 2000 == 0 or step == 19999:
        with torch.no_grad():
            u_pred = u_dof.clone()
            u_pred[:, 0:2] *= (1.0 - bc_d)
            u_pred[:, 2:3] *= (1.0 - bc_r)
            errs = []
            for d in range(3):
                e = (u_pred[:, d] - u_true64[:, d]).pow(2).sum().sqrt()
                r = u_true64[:, d].pow(2).sum().sqrt().clamp(min=1e-30)
                errs.append((e/r).item())
        print(f"  {step:5d}: Π={Pi.item():11.4e}  "
              f"err=[{errs[0]:.4f}, {errs[1]:.4f}, {errs[2]:.4f}]")


# ════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  SUMMARY & DECISION MATRIX")
print(f"{'='*70}")
print(f"""
  Expected results:

  TEST 1 (K assembly fix):
    If ux, uz, θ all < 1e-4  → sign convention CONFIRMED
    Then energy formula is VERIFIED correct

  TEST 2 (Float64 L-BFGS):
    If err < 0.001           → float64 is SUFFICIENT
    If still stalls          → need preconditioning too

  TEST 3 (Float64 Adam):
    If err < 0.05 in 20K     → GNN training FEASIBLE with float64
    If stalls                → need preconditioning for Adam

  TEST 4 (Preconditioned Adam):
    If converges faster      → preconditioning HELPS
    This is what GNN training needs

  TEST 5 (Per-DOF LR):
    If converges             → per-DOF decoder scaling needed in GNN

  ──────────────────────────────────────────────────────────
  DECISION TREE:
  ──────────────────────────────────────────────────────────

  Test 2 converges?
  ├─ YES → Float64 energy loss + matrix form
  │   ├─ Test 3 also converges?
  │   │   ├─ YES → Use float64 energy in GNN, keep Adam
  │   │   └─ NO  → Need preconditioning (Test 4/5)
  │   └─ Implementation: mixed precision
  │       (model float32, energy float64)
  │
  └─ NO → Fundamental problem with approach
          → Consider: equilibrium residual loss instead of energy
          → Or: supervised pre-training + energy fine-tuning
""")