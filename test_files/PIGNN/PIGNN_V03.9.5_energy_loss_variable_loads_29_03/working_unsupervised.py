"""
working_unsupervised.py
========================
Fixes the TWO bugs that killed all previous approaches:

BUG 1: Residual normalized by K_diag (≈10⁹) makes loss=0 at u=0
FIX:   Normalize by ||F||² → loss=1 at u=0, loss=0 at solution

BUG 2: Gradient ∝ K (≈10⁹) causes divergence or tiny effective lr
FIX:   Gradient normalization (divide by gradient norm before step)

Also fixes:
- Decoder init gain=0.1 (not 0.01)
- Float64 for residual computation
- Single u_c scale (avoids 37.5× conditioning)
"""

import torch
import torch.nn as nn
import numpy as np
import os
import time
from pathlib import Path

CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)

from model import PIGNN
from step_2_grapg_constr import FrameData

# ════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════

print(f"{'='*70}")
print(f"  UNSUPERVISED PIGNN — Fixed Residual Loss")
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
print(f"  {len(data_list)} graphs, "
      f"{data_list[0].num_nodes} nodes, "
      f"{data_list[0].connectivity.shape[0]} elements")


# ════════════════════════════════════════════════════════
# FORCE-NORMALIZED EQUILIBRIUM RESIDUAL LOSS
# ════════════════════════════════════════════════════════

class ForceNormalizedResidualLoss(nn.Module):
    """
    Loss = ||Ku - F||² / ||F||²
    
    At u=0:        loss = ||F||²/||F||² = 1       ← has gradient ✓
    At solution:   loss = 0                        ← correct minimum ✓
    
    Previous bug:  loss = ||R/K_diag||² → loss ≈ 0 at u=0 ← WRONG
    
    Gradient ∂L/∂u = 2K^T(Ku-F)/||F||² ∝ K → huge magnitude
    Fix: use gradient normalization (not clipping)
    """

    def __init__(self, use_single_uc=True):
        super().__init__()
        self.use_single_uc = use_single_uc

    def forward(self, model, data):
        pred_raw = model(data)  # (N, 3) non-dim

        # ── Convert to physical ──
        if self.use_single_uc:
            u_c = data.u_c
            theta_c = data.theta_c
            if hasattr(data, 'batch') and data.batch is not None:
                u_c = u_c[data.batch]
                theta_c = theta_c[data.batch]
            u_phys = torch.zeros_like(pred_raw)
            u_phys[:, 0] = pred_raw[:, 0] * u_c
            u_phys[:, 1] = pred_raw[:, 1] * u_c    # SAME scale
            u_phys[:, 2] = pred_raw[:, 2] * theta_c
        else:
            ux_c = data.ux_c
            uz_c = data.uz_c
            theta_c = data.theta_c
            if hasattr(data, 'batch') and data.batch is not None:
                ux_c = ux_c[data.batch]
                uz_c = uz_c[data.batch]
                theta_c = theta_c[data.batch]
            u_phys = torch.zeros_like(pred_raw)
            u_phys[:, 0] = pred_raw[:, 0] * ux_c
            u_phys[:, 1] = pred_raw[:, 1] * uz_c
            u_phys[:, 2] = pred_raw[:, 2] * theta_c

        # ── Upcast to float64 for stiffness operations ──
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

        # ── f_local = K_local @ d_local ──
        f_local = torch.bmm(K_loc, d_local.unsqueeze(2)).squeeze(2)

        # ── Transform to global ──
        f_global_A = torch.zeros(n_elem, 3,
                                 dtype=torch.float64,
                                 device=u64.device)
        f_global_B = torch.zeros_like(f_global_A)

        f_global_A[:, 0] =  c*f_local[:,0] - s*f_local[:,1]
        f_global_A[:, 1] =  s*f_local[:,0] + c*f_local[:,1]
        f_global_A[:, 2] = -f_local[:, 2]

        f_global_B[:, 0] =  c*f_local[:,3] - s*f_local[:,4]
        f_global_B[:, 1] =  s*f_local[:,3] + c*f_local[:,4]
        f_global_B[:, 2] = -f_local[:, 5]

        # ── Assemble ──
        F_int = torch.zeros(n_nodes, 3,
                            dtype=torch.float64,
                            device=u64.device)
        F_int.scatter_add_(
            0, nA.unsqueeze(1).expand_as(f_global_A),
            f_global_A)
        F_int.scatter_add_(
            0, nB.unsqueeze(1).expand_as(f_global_B),
            f_global_B)

        # ── Residual R = Ku - F ──
        R = F_int - F_ext

        # ── Free DOF mask ──
        bc_d = data.bc_disp.double().to(u64.device)
        bc_r = data.bc_rot.double().to(u64.device)
        free_mask = torch.cat([1-bc_d, 1-bc_d, 1-bc_r], dim=1)

        R_free = R * free_mask

        # ═══════════════════════════════════════════════
        # KEY FIX: Normalize by ||F||², NOT by K_diag
        # ═══════════════════════════════════════════════
        F_free = F_ext * free_mask
        F_norm_sq = (F_free**2).sum().clamp(min=1e-30)

        loss = (R_free**2).sum() / F_norm_sq

        # ── Per-graph averaging ──
        n_graphs = data.num_graphs if (
            hasattr(data, 'batch') and data.batch is not None
        ) else 1
        loss = loss / n_graphs

        # ── Diagnostics ──
        with torch.no_grad():
            R_ux = (R_free[:, 0]**2).sum().sqrt().item()
            R_uz = (R_free[:, 1]**2).sum().sqrt().item()
            R_My = (R_free[:, 2]**2).sum().sqrt().item()
            F_norm = F_norm_sq.sqrt().item()

        loss_dict = {
            'loss': loss.item(),
            'R_ux': R_ux / n_graphs,
            'R_uz': R_uz / n_graphs,
            'R_My': R_My / n_graphs,
            'F_norm': F_norm,
            'R_over_F': ((R_free**2).sum().sqrt() / F_norm_sq.sqrt()).item(),
            'pred_range': [pred_raw.min().item(),
                          pred_raw.max().item()],
            'u_range': [u_phys.min().item(),
                       u_phys.max().item()],
        }

        return loss.float(), loss_dict, pred_raw, u_phys


# ════════════════════════════════════════════════════════
# GRADIENT NORMALIZATION (not clipping!)
# ════════════════════════════════════════════════════════

def gradient_normalize_(model, max_norm=1.0):
    """
    Normalize gradient to have norm = max_norm.
    
    Unlike clip_grad_norm_ which only clips when > max_norm,
    this ALWAYS normalizes to max_norm.
    
    This is critical because ∂L/∂u ∝ K ≈ 10⁹,
    and we need CONSISTENT step sizes regardless of K.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.float().norm().item()**2
    total_norm = total_norm**0.5

    if total_norm > 1e-15:
        scale = max_norm / total_norm
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(scale)

    return total_norm


# ════════════════════════════════════════════════════════
# ERROR COMPUTATION
# ════════════════════════════════════════════════════════

def compute_errors(model, norm_data, raw_data, device,
                   use_single_uc=True):
    model.eval()
    errs = {'ux': [], 'uz': [], 'th': [], 'total': []}
    with torch.no_grad():
        for nd, rd in zip(norm_data, raw_data):
            nd = nd.clone().to(device)
            pred = model(nd)

            u_pred = torch.zeros_like(pred)
            if use_single_uc:
                u_pred[:, 0] = pred[:, 0] * nd.u_c
                u_pred[:, 1] = pred[:, 1] * nd.u_c
                u_pred[:, 2] = pred[:, 2] * nd.theta_c
            else:
                u_pred[:, 0] = pred[:, 0] * nd.ux_c
                u_pred[:, 1] = pred[:, 1] * nd.uz_c
                u_pred[:, 2] = pred[:, 2] * nd.theta_c

            u_true = rd.y_node.to(device)
            for d, name in enumerate(['ux', 'uz', 'th']):
                e = (u_pred[:,d]-u_true[:,d]).pow(2).sum().sqrt()
                r = u_true[:,d].pow(2).sum().sqrt().clamp(1e-15)
                errs[name].append((e/r).item())

            e_t = (u_pred-u_true).pow(2).sum().sqrt()
            r_t = u_true.pow(2).sum().sqrt().clamp(1e-15)
            errs['total'].append((e_t/r_t).item())

    return {k: np.mean(v) for k, v in errs.items()}


# ════════════════════════════════════════════════════════
# VERIFICATION: What does the loss look like at u=0
#               and at u=u_true?
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  LOSS VERIFICATION")
print(f"{'='*70}")

loss_fn = ForceNormalizedResidualLoss(use_single_uc=True)

# Test at u=0
class ZeroModel(nn.Module):
    def forward(self, data):
        N = data.x.shape[0]
        pred = torch.zeros(N, 3, device=data.x.device)
        pred[:, 0:2] *= (1.0 - data.bc_disp)
        pred[:, 2:3] *= (1.0 - data.bc_rot)
        return pred

# Test at u=true
class TrueModel(nn.Module):
    def __init__(self, raw_data, data_list):
        super().__init__()
        self.raw_data = raw_data
        self.data_list = data_list
        self.idx = 0

    def forward(self, data):
        rd = self.raw_data[self.idx]
        nd = self.data_list[self.idx]
        u_true = rd.y_node.to(data.x.device)
        u_c = nd.u_c.to(data.x.device)
        theta_c = nd.theta_c.to(data.x.device)

        pred = torch.zeros_like(u_true)
        pred[:, 0] = u_true[:, 0] / u_c
        pred[:, 1] = u_true[:, 1] / u_c
        pred[:, 2] = u_true[:, 2] / theta_c

        pred[:, 0:2] *= (1.0 - data.bc_disp.to(data.x.device))
        pred[:, 2:3] *= (1.0 - data.bc_rot.to(data.x.device))
        return pred

zero_model = ZeroModel()
true_model = TrueModel(raw_data, data_list)

print(f"\n  Loss at u=0:")
for i in range(min(3, len(data_list))):
    d = data_list[i].clone().to(device)
    loss_0, ld0, _, _ = loss_fn(zero_model, d)
    print(f"    Case {i}: loss={loss_0.item():.6f}  "
          f"R/F={ld0['R_over_F']:.4f}")

print(f"\n  Loss at u=true:")
for i in range(min(3, len(data_list))):
    d = data_list[i].clone().to(device)
    true_model.idx = i
    loss_t, ldt, _, _ = loss_fn(true_model, d)
    print(f"    Case {i}: loss={loss_t.item():.6e}  "
          f"R/F={ldt['R_over_F']:.6e}")

print(f"\n  ✓ Loss should be ~1.0 at u=0 and ~0 at u=true")


# ════════════════════════════════════════════════════════
# MODEL CREATION
# ════════════════════════════════════════════════════════

def make_model(hidden=64, layers=3, decoder_gain=0.1):
    model = PIGNN(
        node_in_dim=10, edge_in_dim=7,
        hidden_dim=hidden, n_layers=layers,
    ).to(device)

    with torch.no_grad():
        for dec in [model.decoder_ux, model.decoder_uz,
                    model.decoder_th]:
            last = dec.layers[-1]
            nn.init.xavier_uniform_(last.weight,
                                    gain=decoder_gain)
            last.bias.zero_()

    return model


# ════════════════════════════════════════════════════════
# TEST 1: Force-normalized residual + gradient norm
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  TEST 1: Force-normalized residual + gradient normalization")
print(f"  (single u_c, Adam lr=1e-3, grad_norm=1.0)")
print(f"{'='*70}")

model1 = make_model(hidden=64, layers=3, decoder_gain=0.1)
opt1 = torch.optim.Adam(model1.parameters(), lr=1e-3)

hist1 = {'step': [], 'loss': [], 'ux': [], 'uz': [], 'th': [],
         'grad_raw': [], 'R_over_F': []}

t0 = time.time()
for step in range(5000):
    model1.train()

    total_loss = 0.0
    last_dict = None

    for nd in data_list:
        nd = nd.clone().to(device)
        opt1.zero_grad()
        loss, ld, _, _ = loss_fn(model1, nd)
        loss.backward()
        total_loss += loss.item()
        last_dict = ld

    # ═══ GRADIENT NORMALIZATION (not clipping!) ═══
    raw_gn = gradient_normalize_(model1, max_norm=1.0)
    opt1.step()

    if step % 250 == 0 or step == 4999 or step < 5:
        errs = compute_errors(model1, data_list, raw_data,
                              device, use_single_uc=True)
        avg_loss = total_loss / len(data_list)

        hist1['step'].append(step)
        hist1['loss'].append(avg_loss)
        hist1['ux'].append(errs['ux'])
        hist1['uz'].append(errs['uz'])
        hist1['th'].append(errs['th'])
        hist1['grad_raw'].append(raw_gn)
        hist1['R_over_F'].append(last_dict['R_over_F'])

        print(f"  {step:5d}: loss={avg_loss:9.4f}  "
              f"R/F={last_dict['R_over_F']:.4f}  "
              f"err=[{errs['ux']:.4f}, {errs['uz']:.4f}, "
              f"{errs['th']:.4f}]  "
              f"|∇|_raw={raw_gn:.2e}  "
              f"pred=[{last_dict['pred_range'][0]:.4f}, "
              f"{last_dict['pred_range'][1]:.4f}]")

t1 = time.time()
print(f"  Time: {t1-t0:.1f}s")


# ════════════════════════════════════════════════════════
# TEST 2: Same but with separate ux_c/uz_c
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  TEST 2: Force-normalized residual + separate ux_c/uz_c")
print(f"{'='*70}")

loss_fn_sep = ForceNormalizedResidualLoss(use_single_uc=False)
model2 = make_model(hidden=64, layers=3, decoder_gain=0.1)
opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

hist2 = {'step': [], 'loss': [], 'ux': [], 'uz': [], 'th': []}

for step in range(5000):
    model2.train()
    total_loss = 0.0
    last_dict = None

    for nd in data_list:
        nd = nd.clone().to(device)
        opt2.zero_grad()
        loss, ld, _, _ = loss_fn_sep(model2, nd)
        loss.backward()
        total_loss += loss.item()
        last_dict = ld

    raw_gn = gradient_normalize_(model2, max_norm=1.0)
    opt2.step()

    if step % 500 == 0 or step == 4999:
        errs = compute_errors(model2, data_list, raw_data,
                              device, use_single_uc=False)
        avg_loss = total_loss / len(data_list)
        hist2['step'].append(step)
        hist2['loss'].append(avg_loss)
        hist2['ux'].append(errs['ux'])
        hist2['uz'].append(errs['uz'])
        hist2['th'].append(errs['th'])

        print(f"  {step:5d}: loss={avg_loss:9.4f}  "
              f"err=[{errs['ux']:.4f}, {errs['uz']:.4f}, "
              f"{errs['th']:.4f}]")


# ════════════════════════════════════════════════════════
# TEST 3: Larger model
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  TEST 3: Force-normalized + larger model (H=128, L=6)")
print(f"{'='*70}")

model3 = make_model(hidden=128, layers=6, decoder_gain=0.1)
opt3 = torch.optim.Adam(model3.parameters(), lr=5e-4)
print(f"  Parameters: {model3.count_params():,}")

hist3 = {'step': [], 'loss': [], 'ux': [], 'uz': [], 'th': []}

for step in range(5000):
    model3.train()
    total_loss = 0.0

    for nd in data_list:
        nd = nd.clone().to(device)
        opt3.zero_grad()
        loss, ld, _, _ = loss_fn(model3, nd)
        loss.backward()
        total_loss += loss.item()

    raw_gn = gradient_normalize_(model3, max_norm=1.0)
    opt3.step()

    if step % 500 == 0 or step == 4999:
        errs = compute_errors(model3, data_list, raw_data,
                              device, use_single_uc=True)
        avg_loss = total_loss / len(data_list)
        hist3['step'].append(step)
        hist3['loss'].append(avg_loss)
        hist3['ux'].append(errs['ux'])
        hist3['uz'].append(errs['uz'])
        hist3['th'].append(errs['th'])

        print(f"  {step:5d}: loss={avg_loss:9.4f}  "
              f"err=[{errs['ux']:.4f}, {errs['uz']:.4f}, "
              f"{errs['th']:.4f}]  "
              f"|∇|={raw_gn:.2e}")


# ════════════════════════════════════════════════════════
# TEST 4: Higher decoder gain + higher lr
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  TEST 4: Decoder gain=0.5, lr=5e-3")
print(f"{'='*70}")

model4 = make_model(hidden=64, layers=3, decoder_gain=0.5)
opt4 = torch.optim.Adam(model4.parameters(), lr=5e-3)

hist4 = {'step': [], 'loss': [], 'ux': [], 'uz': [], 'th': []}

for step in range(5000):
    model4.train()
    total_loss = 0.0

    for nd in data_list:
        nd = nd.clone().to(device)
        opt4.zero_grad()
        loss, ld, _, _ = loss_fn(model4, nd)
        loss.backward()
        total_loss += loss.item()

    raw_gn = gradient_normalize_(model4, max_norm=1.0)
    opt4.step()

    if step % 500 == 0 or step == 4999:
        errs = compute_errors(model4, data_list, raw_data,
                              device, use_single_uc=True)
        avg_loss = total_loss / len(data_list)
        hist4['step'].append(step)
        hist4['loss'].append(avg_loss)
        hist4['ux'].append(errs['ux'])
        hist4['uz'].append(errs['uz'])
        hist4['th'].append(errs['th'])

        print(f"  {step:5d}: loss={avg_loss:9.4f}  "
              f"err=[{errs['ux']:.4f}, {errs['uz']:.4f}, "
              f"{errs['th']:.4f}]")


# ════════════════════════════════════════════════════════
# TEST 5: Gradient norm=0.1 (smaller steps)
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  TEST 5: Gradient norm=0.1, lr=1e-3, 10K steps")
print(f"{'='*70}")

model5 = make_model(hidden=64, layers=3, decoder_gain=0.1)
opt5 = torch.optim.Adam(model5.parameters(), lr=1e-3)

hist5 = {'step': [], 'loss': [], 'ux': [], 'uz': [], 'th': []}

for step in range(10000):
    model5.train()
    total_loss = 0.0

    for nd in data_list:
        nd = nd.clone().to(device)
        opt5.zero_grad()
        loss, ld, _, _ = loss_fn(model5, nd)
        loss.backward()
        total_loss += loss.item()

    raw_gn = gradient_normalize_(model5, max_norm=0.1)
    opt5.step()

    if step % 1000 == 0 or step == 9999:
        errs = compute_errors(model5, data_list, raw_data,
                              device, use_single_uc=True)
        avg_loss = total_loss / len(data_list)
        hist5['step'].append(step)
        hist5['loss'].append(avg_loss)
        hist5['ux'].append(errs['ux'])
        hist5['uz'].append(errs['uz'])
        hist5['th'].append(errs['th'])

        print(f"  {step:5d}: loss={avg_loss:9.4f}  "
              f"err=[{errs['ux']:.4f}, {errs['uz']:.4f}, "
              f"{errs['th']:.4f}]")


# ════════════════════════════════════════════════════════
# COMPARISON PLOT
# ════════════════════════════════════════════════════════

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Unsupervised PIGNN: Force-Normalized Residual',
             fontsize=14, fontweight='bold')

tests = [
    ('1: base (H64,L3)', hist1, 'blue'),
    ('2: sep ux_c/uz_c', hist2, 'red'),
    ('3: large (H128,L6)', hist3, 'green'),
    ('4: gain=0.5,lr=5e-3', hist4, 'purple'),
    ('5: gnorm=0.1, 10K', hist5, 'orange'),
]

# Loss
ax = axes[0, 0]
for name, h, color in tests:
    if h['loss']:
        ax.semilogy(h['step'], h['loss'],
                    color=color, label=name, lw=1.5)
ax.axhline(y=1.0, color='gray', ls=':', label='u=0 baseline')
ax.set_title('Loss (||R||²/||F||²)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Per-DOF errors
for d, (name, ax_idx) in enumerate([
    ('ux error', (0,1)), ('uz error', (0,2)),
    ('θ error', (1,0))
]):
    ax = axes[ax_idx]
    key = ['ux', 'uz', 'th'][d]
    for tname, h, color in tests:
        if h[key]:
            ax.semilogy(h['step'], h[key],
                        color=color, label=tname, lw=1.5)
    ax.axhline(y=0.05, color='gray', ls=':', label='5%')
    ax.axhline(y=1.0, color='red', ls=':', alpha=0.5,
               label='u=0 level')
    ax.set_title(name)
    ax.set_ylim(1e-3, 5)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# Gradient norm
ax = axes[1, 1]
if hist1['grad_raw']:
    ax.semilogy(hist1['step'], hist1['grad_raw'],
                'b-', lw=1.5, label='Raw |∇|')
ax.set_title('Raw gradient norm (before normalization)')
ax.legend()
ax.grid(True, alpha=0.3)

# R/F ratio
ax = axes[1, 2]
if hist1['R_over_F']:
    ax.plot(hist1['step'], hist1['R_over_F'],
            'b-', lw=1.5)
ax.axhline(y=1.0, color='gray', ls=':', label='u=0')
ax.axhline(y=0.0, color='green', ls=':', label='converged')
ax.set_title('||R||/||F|| (Test 1)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("RESULTS", exist_ok=True)
plt.savefig('RESULTS/unsupervised_fixed.png',
            dpi=150, bbox_inches='tight')
plt.show()


# ════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  FINAL COMPARISON")
print(f"{'='*70}")

for name, h, _ in tests:
    if h['ux']:
        best = {k: min(h[k]) for k in ['ux', 'uz', 'th']}
        final_loss = h['loss'][-1] if h['loss'] else float('nan')
        print(f"\n  {name}:")
        print(f"    Final loss: {final_loss:.4f}  "
              f"(should decrease from ~1.0)")
        print(f"    Best err: ux={best['ux']:.4f}, "
              f"uz={best['uz']:.4f}, θ={best['th']:.4f}")
        worst = max(best.values())
        if worst < 0.05:
            print(f"    ✓ CONVERGED (<5%)")
        elif worst < 0.2:
            print(f"    ~ PARTIAL")
        elif worst < 0.95:
            print(f"    ~ SOME LEARNING (below u=0 level)")
        else:
            print(f"    ✗ NOT converged")

print(f"\n  Key diagnostic:")
print(f"    Loss at u=0: ~1.0")
print(f"    Loss should DECREASE during training")
print(f"    If loss stays ~1.0: gradient signal exists but")
print(f"       network can't find the right direction")
print(f"    If loss < 1.0 but err ~1.0: network found a")
print(f"       different solution than the true one (possible")
print(f"       for underdetermined systems)")