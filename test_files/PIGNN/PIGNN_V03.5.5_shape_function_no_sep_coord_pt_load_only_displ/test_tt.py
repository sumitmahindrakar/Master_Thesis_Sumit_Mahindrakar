"""
plot_results.py — Plot and analyze from saved history
No training needed. Just loads history.pt and plots.
"""

import os
import numpy as np
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

RESULTS_DIR = "RESULTS"
HISTORY_PATH = os.path.join(RESULTS_DIR, "history.pt")

# ── Load history ──
history = torch.load(HISTORY_PATH, weights_only=False)
print(f"Loaded history: {len(history['epoch'])} epochs")
print(f"Keys: {list(history.keys())}")

# ════════════════════════════════════════
# PLOT
# ════════════════════════════════════════

import matplotlib.pyplot as plt

epochs = history['epoch']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Energy PIGNN — Training', fontsize=14, fontweight='bold')

# Pi
ax = axes[0, 0]
ax.plot(epochs, history['Pi'], 'k-', lw=1.5)
ax.axhline(y=0, color='gray', ls='--', alpha=0.5)
ax.set_title('Pi (potential energy)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Pi / E_c')
ax.grid(True, alpha=0.3)

# U_axial and U_bend
ax = axes[0, 1]
ax.semilogy(epochs, [max(v, 1e-15) for v in history['U_axial']],
            'tab:orange', lw=1.5, label='U_axial')
ax.semilogy(epochs, [max(v, 1e-15) for v in history['U_bend']],
            'tab:red', lw=1.5, label='U_bend')
ax.set_title('Strain Energy Components')
ax.set_xlabel('Epoch')
ax.legend()
ax.grid(True, alpha=0.3)

# W_ext
ax = axes[1, 0]
ax.plot(epochs, history['W_ext'], 'tab:blue', lw=1.5)
ax.set_title('External Work W_ext')
ax.set_xlabel('Epoch')
ax.grid(True, alpha=0.3)

# EA factor
ax = axes[1, 1]
if 'ea_factor' in history:
    ax.semilogy(epochs, history['ea_factor'], 'tab:green', lw=1.5)
    ax.set_title('EA Curriculum Factor')
    ax.set_xlabel('Epoch')
else:
    ax.text(0.5, 0.5, 'No EA factor', ha='center', transform=ax.transAxes)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'loss_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {RESULTS_DIR}/loss_curves.png")

# Displacement error
if history['val_disp_error']:
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    n_val = len(history['val_disp_error'])
    validate_every = 100
    val_epochs = [
        epochs[i] for i in range(len(epochs))
        if i == 0 or (epochs[i] % validate_every == 0)
    ][:n_val]

    ax2.plot(val_epochs, history['val_disp_error'], 'ro-', ms=4, lw=1.5)
    ax2.axhline(y=1.0, color='gray', ls='--', alpha=0.5, label='zero solution')
    ax2.axhline(y=0.1, color='green', ls='--', alpha=0.5, label='10% error')
    ax2.set_title('Displacement Error vs Kratos')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Relative L2 Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'disp_error.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {RESULTS_DIR}/disp_error.png")

# ════════════════════════════════════════
# ANALYZE
# ════════════════════════════════════════

print(f"\n{'='*80}")
print(f"  RESULT ANALYSIS")
print(f"{'='*80}")

# Final values
print(f"\n  Final values (epoch {epochs[-1]}):")
print(f"  {'-'*50}")
for key in ['Pi', 'U_axial', 'U_bend', 'W_ext', 'L_bc', 'total']:
    if key in history:
        print(f"    {key:<10} {history[key][-1]:>12.6e}")

# Best total
best_idx = int(np.argmin(history['total']))
print(f"\n  Best total at epoch {epochs[best_idx]}:")
print(f"    total = {history['total'][best_idx]:.6e}")
print(f"    Pi    = {history['Pi'][best_idx]:.6e}")

# Displacement error
if history['val_disp_error']:
    de = history['val_disp_error']
    print(f"\n  Displacement error:")
    print(f"    Best:   {min(de):.6e}")
    print(f"    Final:  {de[-1]:.6e}")
    if min(de) < 0.01:
        print(f"    EXCELLENT (< 1%)")
    elif min(de) < 0.05:
        print(f"    GOOD (< 5%)")
    elif min(de) < 0.20:
        print(f"    MODERATE (< 20%)")
    elif min(de) < 1.0:
        print(f"    POOR but improving")
    else:
        print(f"    VERY POOR")

print(f"\n{'='*80}")