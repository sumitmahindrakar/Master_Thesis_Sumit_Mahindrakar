"""
verify_corotational.py — Run this BEFORE training
"""
import torch
import os
from pathlib import Path
from corotational import CorotationalBeam2D, verify_beam_element

CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)

# Load RAW (un-normalized) data
data_list = torch.load("DATA/graph_dataset.pt",
                       weights_only=False)
data = data_list[0]

# Run full verification
result = verify_beam_element(data)

# Quick summary
beam = CorotationalBeam2D()
true_disp = data.y_node.clone()
result = beam(true_disp, data)

# Check equilibrium
residual = result['nodal_forces'] - data.F_ext
free = data.bc_disp.squeeze(-1) < 0.5

print(f"\n── SUMMARY ──")
print(f"  Equilibrium Max|R|: {residual[free].abs().max():.6e}")

if data.y_element is not None:
    kratos_M = data.y_element[:, 1]
    kratos_V = data.y_element[:, 2]
    
    err_M_mid = (result['M_mid'] - kratos_M).abs().max()
    err_V = (result['V_e'] - kratos_V).abs().max()
    
    print(f"  M_mid error:        {err_M_mid:.6e}")
    print(f"  V error:            {err_V:.6e}")
    
    if err_M_mid / kratos_M.abs().max() < 0.01:
        print(f"  ✓ M_mid matches Kratos")
    else:
        print(f"  ✗ M_mid does NOT match — investigate further")