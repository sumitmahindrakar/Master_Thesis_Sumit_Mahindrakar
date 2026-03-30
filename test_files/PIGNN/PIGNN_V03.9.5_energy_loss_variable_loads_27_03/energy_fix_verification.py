"""
energy_fix_verification.py
==========================
1. Assembles K globally, solves Ku=F exactly → verifies energy formula
2. Tests optimization in physical space (not pred_raw) → fixes conditioning
3. Tests single-scale u_c (not separate ux_c/uz_c) → reduces condition number
"""

import os
import time
import numpy as np
from pathlib import Path

CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)

from step_2_grapg_constr import FrameData

import torch
import numpy as np

# ── Load data ──
data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
from normalizer import PhysicsScaler
data_list = PhysicsScaler.compute_and_store_list(data_list)
data = data_list[0]

from energy_loss import FrameEnergyLoss
loss_fn = FrameEnergyLoss()

N = data.num_nodes
E_elem = data.connectivity.shape[0]
u_true = data.y_node

print(f"{'='*70}")
print(f"  ENERGY VERIFICATION AND FIX")
print(f"{'='*70}")
print(f"  Nodes: {N}, Elements: {E_elem}")
print(f"  DOFs: {3*N} total, ~{3*N - 6} free (approx)")
print(f"  ux_c={data.ux_c.item():.4e}, uz_c={data.uz_c.item():.4e}")
print(f"  ratio: {(data.ux_c/data.uz_c).item():.1f}×")


# ════════════════════════════════════════════════════════
# TEST 1: Assemble K globally, solve Ku=F
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  TEST 1: Direct K assembly and solve")
print(f"{'='*70}")

def assemble_global_K_and_F(data):
    """Assemble global stiffness matrix and force vector."""
    conn = data.connectivity.numpy()
    coords = data.coords.numpy()
    n_nodes = data.num_nodes
    n_dof = 3 * n_nodes  # ux, uz, θ per node

    K_global = np.zeros((n_dof, n_dof))
    F_global = np.zeros(n_dof)

    L = data.elem_lengths.numpy()
    EA = (data.prop_E * data.prop_A).numpy()
    EI = (data.prop_E * data.prop_I22).numpy()
    dirs = data.elem_directions.numpy()

    for e in range(len(conn)):
        nA, nB = conn[e]
        c = dirs[e, 0]  # cos
        s = dirs[e, 2]  # sin
        Le = L[e]

        ea_l = EA[e] / Le
        ei_l = EI[e] / Le
        ei_l2 = EI[e] / Le**2
        ei_l3 = EI[e] / Le**3

        # Local stiffness (6×6)
        K_loc = np.zeros((6, 6))

        # Axial
        K_loc[0,0] =  ea_l;  K_loc[0,3] = -ea_l
        K_loc[3,0] = -ea_l;  K_loc[3,3] =  ea_l

        # Bending
        K_loc[1,1] =  12*ei_l3; K_loc[1,2] =  6*ei_l2
        K_loc[1,4] = -12*ei_l3; K_loc[1,5] =  6*ei_l2
        K_loc[2,1] =  6*ei_l2;  K_loc[2,2] =  4*ei_l
        K_loc[2,4] = -6*ei_l2;  K_loc[2,5] =  2*ei_l
        K_loc[4,1] = -12*ei_l3; K_loc[4,2] = -6*ei_l2
        K_loc[4,4] =  12*ei_l3; K_loc[4,5] = -6*ei_l2
        K_loc[5,1] =  6*ei_l2;  K_loc[5,2] =  2*ei_l
        K_loc[5,4] = -6*ei_l2;  K_loc[5,5] =  4*ei_l

        # Transformation matrix T (6×6)
        # Local DOFs: [u_loc, w_loc, θ_loc] for each node
        # Global DOFs: [ux, uz, θ] for each node
        # u_loc =  c*ux + s*uz
        # w_loc = -s*ux + c*uz
        # θ_loc = θ  (NOT negated for standard assembly!)
        T = np.zeros((6, 6))
        T[0,0] =  c;  T[0,1] = s
        T[1,0] = -s;  T[1,1] = c
        T[2,2] = 1.0
        T[3,3] =  c;  T[3,4] = s
        T[4,3] = -s;  T[4,4] = c
        T[5,5] = 1.0

        # Global element stiffness: K_glob = T^T K_loc T
        K_glob = T.T @ K_loc @ T

        # Assembly
        dofs = [3*nA, 3*nA+1, 3*nA+2,
                3*nB, 3*nB+1, 3*nB+2]

        for i in range(6):
            for j in range(6):
                K_global[dofs[i], dofs[j]] += K_glob[i, j]

    # Force vector
    F_ext = data.F_ext.numpy()
    for n in range(n_nodes):
        F_global[3*n]   = F_ext[n, 0]   # Fx
        F_global[3*n+1] = F_ext[n, 1]   # Fz
        F_global[3*n+2] = F_ext[n, 2]   # My

    return K_global, F_global


K_global, F_global = assemble_global_K_and_F(data)

# Identify free and fixed DOFs
bc_disp = data.bc_disp.numpy().flatten()
bc_rot = data.bc_rot.numpy().flatten()

fixed_dofs = []
for n in range(N):
    if bc_disp[n] > 0.5:
        fixed_dofs.extend([3*n, 3*n+1])  # ux, uz fixed
    if bc_rot[n] > 0.5:
        fixed_dofs.append(3*n+2)           # θ fixed

all_dofs = list(range(3*N))
free_dofs = [d for d in all_dofs if d not in fixed_dofs]

print(f"  Total DOFs: {3*N}")
print(f"  Fixed DOFs: {len(fixed_dofs)}")
print(f"  Free DOFs:  {len(free_dofs)}")

# Extract submatrices
K_ff = K_global[np.ix_(free_dofs, free_dofs)]
F_f = F_global[free_dofs]

# Condition number
eigvals = np.linalg.eigvalsh(K_ff)
cond = eigvals.max() / eigvals.min()
print(f"\n  K eigenvalues: [{eigvals.min():.4e}, {eigvals.max():.4e}]")
print(f"  CONDITION NUMBER: {cond:.0f}")
print(f"  (This is what optimizers must handle)")

# Solve Ku=F
u_solved = np.zeros(3*N)
u_solved[free_dofs] = np.linalg.solve(K_ff, F_f)

# Reshape to (N, 3)
u_solved_2d = u_solved.reshape(N, 3)

# Compare with Kratos true solution
u_true_np = u_true.numpy()

print(f"\n  Direct solve vs Kratos:")
for dof, name in enumerate(['ux', 'uz', 'θ']):
    err = np.linalg.norm(u_solved_2d[:, dof] - u_true_np[:, dof])
    ref = np.linalg.norm(u_true_np[:, dof])
    rel = err / max(ref, 1e-15)
    print(f"    {name}: rel_err = {rel:.6e}  "
          f"(solved=[{u_solved_2d[:,dof].min():.6e}, "
          f"{u_solved_2d[:,dof].max():.6e}])")

# Energy at solved solution
u_solved_t = torch.tensor(u_solved_2d, dtype=torch.float32)
U_solved = loss_fn._strain_energy(u_solved_t, data)
W_solved = loss_fn._external_work(u_solved_t, data)
Pi_solved = U_solved - W_solved

print(f"\n  Energy at direct solution:")
print(f"    U = {U_solved.item():.6e}")
print(f"    W = {W_solved.item():.6e}")
print(f"    Π = {Pi_solved.item():.6e}")
print(f"    U/W = {(U_solved/W_solved).item():.6f}")

U_true = loss_fn._strain_energy(u_true, data)
W_true = loss_fn._external_work(u_true, data)
print(f"\n  Energy at Kratos solution:")
print(f"    U = {U_true.item():.6e}")
print(f"    W = {W_true.item():.6e}")
print(f"    U/W = {(U_true/W_true).item():.6f}")

# Check if they match
energy_match = abs(U_solved.item() - U_true.item()) / abs(U_true.item()) < 0.01
print(f"\n  {'✓' if energy_match else '✗'} "
      f"Energy formula matches direct solve")

# ═══ Check sign convention ═══
# Test: does _nondim_energy give same result as _strain_energy - _external_work?
print(f"\n  Sign convention check:")

# Need a fake model that returns pred_raw
class FakeModel(torch.nn.Module):
    def __init__(self, pred_raw, bc_disp, bc_rot):
        super().__init__()
        self.pred = pred_raw  # not a parameter, just fixed
        self.bc_disp = bc_disp
        self.bc_rot = bc_rot

    def forward(self, data):
        pred = self.pred.clone()
        pred[:, 0:2] *= (1.0 - self.bc_disp)
        pred[:, 2:3] *= (1.0 - self.bc_rot)
        return pred

# Convert true solution to pred_raw with SEPARATE scales
pred_true_sep = torch.zeros_like(u_true)
pred_true_sep[:, 0] = u_true[:, 0] / data.ux_c
pred_true_sep[:, 1] = u_true[:, 1] / data.uz_c
pred_true_sep[:, 2] = u_true[:, 2] / data.theta_c

fake_model = FakeModel(pred_true_sep, data.bc_disp, data.bc_rot)
with torch.no_grad():
    Pi_nondim, ld, _, _ = loss_fn(fake_model, data)

E_c = (data.F_c * data.ux_c).item()
Pi_physical_from_nondim = Pi_nondim.item() * E_c
Pi_physical_from_direct = (U_true - W_true).item()

print(f"  _nondim_energy at true sol:   {Pi_nondim.item():.6e} "
      f"(×E_c = {Pi_physical_from_nondim:.6e})")
print(f"  U - W at true sol:            "
      f"{Pi_physical_from_direct:.6e}")
print(f"  Match: "
      f"{'✓' if abs(Pi_physical_from_nondim - Pi_physical_from_direct) / abs(Pi_physical_from_direct) < 0.01 else '✗'}")


# ═══ Check force vector ═══
print(f"\n  Force vector check:")
F_ext = data.F_ext.numpy()
for i in range(3):
    name = ['Fx', 'Fz', 'My'][i]
    n_nonzero = np.sum(np.abs(F_ext[:, i]) > 1e-10)
    if n_nonzero > 0:
        vals = F_ext[np.abs(F_ext[:, i]) > 1e-10, i]
        print(f"    {name}: {n_nonzero} nodes, "
              f"range [{vals.min():.4f}, {vals.max():.4f}]")
    else:
        print(f"    {name}: ALL ZERO ← no gradient signal for this DOF!")


# ════════════════════════════════════════════════════════
# TEST 2: Physical-space L-BFGS (condition number ~6400)
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  TEST 2: Physical-space optimization")
print(f"{'='*70}")

class PhysicalPredictor(torch.nn.Module):
    """Predicts physical displacements directly (not non-dim)."""
    def __init__(self, N, bc_disp, bc_rot, ux_c, uz_c, theta_c):
        super().__init__()
        self.u_phys = torch.nn.Parameter(torch.zeros(N, 3))
        self.bc_disp = bc_disp
        self.bc_rot = bc_rot
        self.ux_c = ux_c
        self.uz_c = uz_c
        self.theta_c = theta_c

    def forward(self, data):
        """Returns pred_raw (non-dim) but optimizes in physical space."""
        u = self.u_phys.clone()
        u[:, 0:2] *= (1.0 - self.bc_disp)
        u[:, 2:3] *= (1.0 - self.bc_rot)

        # Convert to pred_raw for energy_loss compatibility
        pred_raw = torch.zeros_like(u)
        pred_raw[:, 0] = u[:, 0] / self.ux_c
        pred_raw[:, 1] = u[:, 1] / self.uz_c
        pred_raw[:, 2] = u[:, 2] / self.theta_c
        return pred_raw

model_phys = PhysicalPredictor(
    N, data.bc_disp, data.bc_rot,
    data.ux_c, data.uz_c, data.theta_c
)

optimizer_lbfgs = torch.optim.LBFGS(
    model_phys.parameters(),
    lr=1.0,
    max_iter=50,
    history_size=50,
    line_search_fn='strong_wolfe',
    tolerance_grad=1e-12,
    tolerance_change=1e-15,
)

print(f"\n  L-BFGS in physical space:")
for step in range(100):
    def closure():
        optimizer_lbfgs.zero_grad()
        loss, _, _, _ = loss_fn(model_phys, data)
        loss.backward()
        return loss

    loss_val = optimizer_lbfgs.step(closure)

    if step % 10 == 0 or step < 5:
        with torch.no_grad():
            u_pred = model_phys.u_phys.clone()
            u_pred[:, 0:2] *= (1.0 - data.bc_disp)
            u_pred[:, 2:3] *= (1.0 - data.bc_rot)

            errs = []
            for d in range(3):
                e = (u_pred[:, d] - u_true[:, d]).pow(2).sum().sqrt()
                r = u_true[:, d].pow(2).sum().sqrt().clamp(min=1e-15)
                errs.append((e/r).item())

            # Check gradient magnitude
            g = model_phys.u_phys.grad
            grad_norms = [g[:, d].abs().max().item() for d in range(3)] if g is not None else [0,0,0]

            print(f"  Step {step:3d}: Π={loss_val.item():11.4e}  "
                  f"err=[{errs[0]:.4f}, {errs[1]:.4f}, {errs[2]:.4f}]  "
                  f"|∇|=[{grad_norms[0]:.2e}, {grad_norms[1]:.2e}, {grad_norms[2]:.2e}]")

    # Check convergence
    if step > 5:
        g = model_phys.u_phys.grad
        if g is not None and g.abs().max().item() < 1e-8:
            print(f"  Converged at step {step}!")
            break


# ════════════════════════════════════════════════════════
# TEST 3: Single u_c scale (not separate ux_c/uz_c)
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  TEST 3: Single u_c scale (pred_raw space)")
print(f"{'='*70}")

class SingleScalePredictor(torch.nn.Module):
    """Uses SINGLE u_c for both ux and uz."""
    def __init__(self, N, bc_disp, bc_rot, u_c, theta_c):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(N, 3))
        self.bc_disp = bc_disp
        self.bc_rot = bc_rot
        self.u_c = u_c
        self.theta_c = theta_c

    def forward(self, data):
        pred = self.p.clone()
        pred[:, 0:2] *= (1.0 - self.bc_disp)
        pred[:, 2:3] *= (1.0 - self.bc_rot)
        return pred

# Modify data to use single scale
data_single = data.clone()
data_single.ux_c = data.u_c  # use max(ux_c, uz_c)
data_single.uz_c = data.u_c  # SAME scale for both
# theta_c stays the same

print(f"  Using u_c = {data.u_c.item():.4e} for BOTH ux and uz")
print(f"  (was: ux_c={data.ux_c.item():.4e}, uz_c={data.uz_c.item():.4e})")

# True pred_raw with single scale
pred_true_single = torch.zeros_like(u_true)
pred_true_single[:, 0] = u_true[:, 0] / data.u_c
pred_true_single[:, 1] = u_true[:, 1] / data.u_c  # now much smaller!
pred_true_single[:, 2] = u_true[:, 2] / data.theta_c

print(f"\n  True pred_raw ranges (single scale):")
print(f"    DOF 0 (ux/u_c): [{pred_true_single[:,0].min():.4f}, "
      f"{pred_true_single[:,0].max():.4f}]")
print(f"    DOF 1 (uz/u_c): [{pred_true_single[:,1].min():.6f}, "
      f"{pred_true_single[:,1].max():.6f}]  ← small but OK")
print(f"    DOF 2 (θ/θ_c):  [{pred_true_single[:,2].min():.4f}, "
      f"{pred_true_single[:,2].max():.4f}]")

# Test with L-BFGS
model_single = SingleScalePredictor(
    N, data_single.bc_disp, data_single.bc_rot,
    data_single.u_c, data_single.theta_c
)

opt_single = torch.optim.LBFGS(
    model_single.parameters(), lr=1.0,
    max_iter=50, history_size=50,
    line_search_fn='strong_wolfe',
    tolerance_grad=1e-12,
)

print(f"\n  L-BFGS with single u_c:")
for step in range(100):
    def closure():
        opt_single.zero_grad()
        loss, _, _, _ = loss_fn(model_single, data_single)
        loss.backward()
        return loss

    loss_val = opt_single.step(closure)

    if step % 10 == 0 or step < 5:
        with torch.no_grad():
            pred = model_single(data_single)
            u_pred = torch.zeros_like(pred)
            u_pred[:, 0] = pred[:, 0] * data_single.u_c
            u_pred[:, 1] = pred[:, 1] * data_single.u_c
            u_pred[:, 2] = pred[:, 2] * data_single.theta_c

            errs = []
            for d in range(3):
                e = (u_pred[:, d] - u_true[:, d]).pow(2).sum().sqrt()
                r = u_true[:, d].pow(2).sum().sqrt().clamp(min=1e-15)
                errs.append((e/r).item())

            print(f"  Step {step:3d}: Π={loss_val.item():11.4e}  "
                  f"err=[{errs[0]:.4f}, {errs[1]:.4f}, {errs[2]:.4f}]")

    g = model_single.p.grad
    if g is not None and g.abs().max().item() < 1e-8:
        print(f"  Converged at step {step}!")
        break


# ════════════════════════════════════════════════════════
# TEST 4: Adam in physical space (what GNN will use)
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  TEST 4: Adam in physical space (GNN-relevant)")
print(f"{'='*70}")

model_adam = PhysicalPredictor(
    N, data.bc_disp, data.bc_rot,
    data.ux_c, data.uz_c, data.theta_c
)

opt_adam = torch.optim.Adam(model_adam.parameters(), lr=1e-4)

print(f"\n  Adam lr=1e-4 in physical space:")
for step in range(10000):
    opt_adam.zero_grad()
    loss, ld, _, _ = loss_fn(model_adam, data)

    if torch.isnan(loss):
        print(f"  NaN at step {step}")
        break

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_adam.parameters(), 1.0)
    opt_adam.step()

    if step % 1000 == 0 or step == 9999:
        with torch.no_grad():
            u_pred = model_adam.u_phys.clone()
            u_pred[:, 0:2] *= (1.0 - data.bc_disp)
            u_pred[:, 2:3] *= (1.0 - data.bc_rot)

            errs = []
            for d in range(3):
                e = (u_pred[:, d] - u_true[:, d]).pow(2).sum().sqrt()
                r = u_true[:, d].pow(2).sum().sqrt().clamp(min=1e-15)
                errs.append((e/r).item())

            g = model_adam.u_phys.grad
            gn = [g[:, d].abs().mean().item() for d in range(3)] if g is not None else [0]*3

            print(f"  Step {step:5d}: Π={ld['Pi']:11.4e}  "
                  f"err=[{errs[0]:.4f}, {errs[1]:.4f}, {errs[2]:.4f}]  "
                  f"|∇|=[{gn[0]:.2e}, {gn[1]:.2e}, {gn[2]:.2e}]")


# ════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  DIAGNOSIS SUMMARY")
print(f"{'='*70}")
print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │ ROOT CAUSE 1: Conditioning                                  │
  │   Separate ux_c/uz_c creates condition number ~10⁶          │
  │   Physical K has condition number ~{cond:.0f}              │
  │   Fix: use SINGLE u_c or optimize in physical space         │
  │                                                             │
  │ ROOT CAUSE 2: Zero gradient for uz, θ                       │
  │   Only Fx forces applied → ∂W/∂uz = 0, ∂W/∂θ = 0          │
  │   These DOFs learn ONLY through K coupling (slow)           │
  │   Fix: optimize in physical space (better conditioning)     │
  │        or use Newton/L-BFGS (handles coupling naturally)    │
  │                                                             │
  │ EXPECTED RESULTS:                                           │
  │   Test 1 (Ku=F): err < 1e-6 → energy formula correct       │
  │   Test 2 (phys L-BFGS): err < 0.01 → conditioning fixed    │
  │   Test 3 (single u_c): err < 0.01 → scale fix works        │
  │   Test 4 (phys Adam): err < 0.05 → GNN-compatible          │
  └─────────────────────────────────────────────────────────────┘
""")