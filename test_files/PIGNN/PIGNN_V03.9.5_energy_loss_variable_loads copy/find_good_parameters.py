"""
find_good_parameters.py
Compute expected displacements for different E, A, I, Load
to find parameters that give u ~ O(1)
"""
import numpy as np

# Frame geometry
H_total = 18.0      # total height
W_total = 6.0       # total width
L_col = 0.6         # column segment length
L_beam = 1.2        # beam segment length
n_floors = 6        # number of floors
n_col_segments = 30 # column segments per column (18/0.6)

# Rough estimate: lateral stiffness of frame
# Each column contributes 12EI/L³ in shear
# n_columns × n_floors columns contribute

def estimate_disp(E, A, I22, F_lateral):
    EA = E * A
    EI = E * I22
    
    # Lateral stiffness of one column segment
    k_col = 12 * EI / L_col**3
    
    # Total lateral stiffness (rough: 2 columns per floor)
    # Each floor has 2 column segments resisting shear
    # Total floors of columns
    k_total = 2 * k_col / n_floors  # very rough
    
    # More accurate: cantilever approximation
    # Treat frame as equivalent cantilever
    # Total I_equiv ≈ 2 × A_col × (W/2)² (parallel axis)
    I_equiv = 2 * A * (W_total / 2)**2
    EI_equiv = E * I_equiv
    u_cantilever = F_lateral * H_total**3 / (3 * EI_equiv)
    
    # Beam theory with distributed stiffness
    # u ≈ F × H³ / (3 × EI_frame)
    # where EI_frame accounts for portal action
    
    # Simple estimate: n_floors point loads
    # Each floor has force F/n_floors
    # Deflection ≈ sum of shear deformation
    u_shear = F_lateral * H_total / k_total
    
    theta_max = F_lateral * H_total**2 / (2 * EI_equiv)
    
    return {
        'EA': EA,
        'EI': EI,
        'EA_over_EI': EA / EI,
        'k_col_segment': k_col,
        'u_cantilever': u_cantilever,
        'u_shear': u_shear,
        'theta_max': theta_max,
        'EA_over_L': EA / L_col,
        'EI_over_L3': 12 * EI / L_col**3,
    }

print(f"{'E':>8} {'A':>8} {'I22':>8} {'F':>6} | "
      f"{'EA':>8} {'EI':>8} {'EA/EI':>8} | "
      f"{'u_est':>10} {'θ_est':>10} | "
      f"{'EA/L':>10} {'12EI/L³':>10}")
print("-" * 120)

test_params = [
    # E,    A,     I22,   F
    (1,     1,     1,     1),      # your current
    (10,    1,     1,     1),
    (100,   1,     0.1,   1),
    (100,   1,     1,     1),
    (1000,  1,     0.1,   1),
    (1000,  1,     1,     1),
    (10,    1,     0.1,   1),
    (10,    10,    0.1,   1),
    (100,   10,    0.1,   1),
    (1e4,   1,     0.01,  1),
    (1e4,   1,     0.1,   1),
    (1e4,   1,     1,     10),
    (1e6,   0.01,  0.001, 10),
]

for E, A, I22, F in test_params:
    r = estimate_disp(E, A, I22, F)
    status = ""
    if 0.01 < abs(r['u_cantilever']) < 10:
        status += " ← u OK"
    if 1 < r['EA_over_EI'] < 100:
        status += " ← ratio OK"
    if 0.01 < abs(r['theta_max']) < 10:
        status += " ← θ OK"
    
    print(f"{E:8.1f} {A:8.3f} {I22:8.4f} {F:6.1f} | "
          f"{r['EA']:8.1f} {r['EI']:8.1f} "
          f"{r['EA_over_EI']:8.1f} | "
          f"{r['u_cantilever']:10.4f} "
          f"{r['theta_max']:10.4f} | "
          f"{r['EA_over_L']:10.1f} "
          f"{r['EI_over_L3']:10.1f}"
          f"{status}")

print(f"\n\nTARGET: u ~ 0.01 to 1.0")
print(f"TARGET: θ ~ 0.001 to 0.1")
print(f"TARGET: EA/EI ~ 1 to 100")
print(f"TARGET: EA/L and 12EI/L³ both ~ 1 to 1000")