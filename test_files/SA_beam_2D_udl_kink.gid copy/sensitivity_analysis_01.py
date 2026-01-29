"""
Simple Sensitivity Calculator from VTK Results
∂M/∂(EI) using General Influence Method
"""

import numpy as np

def compute_moment_sensitivity():
    """
    Compute ∂M/∂(EI) from VTK output values
    """
    
    # Material Properties
    E = 2.1e11      # Pa
    I = 5e-6        # m^4
    EI = E * I
    
    print("=" * 60)
    print("MOMENT SENSITIVITY: ∂M/∂(EI)")
    print("=" * 60)
    print(f"\nEI = {EI:.4e} N·m²")
    
    # Element lengths
    L = 2.0 / 3.0  # 0.6667 m for each element
    
    # Primary moments (from your VTK CELL_DATA)
    M_primary = {
        1: 740.74237,
        2: -1481.4814,
        3: 740.74237
    }
    
    # Dual moments (from your VTK CELL_DATA)
    M_dual = {
        1: -0.33333334,
        2: 0.66666669,
        3: -0.33333334
    }
    
    print("\n" + "-" * 60)
    print(f"{'Elem':^6} {'M':^12} {'M̄':^12} {'∂M/∂(EI)':^18}")
    print("-" * 60)
    
    total = 0.0
    
    for eid in [1, 2, 3]:
        # ∂M/∂(EI)_k = -(M · M̄ · L) / EI²
        dM_dEI = -(M_primary[eid] * M_dual[eid] * L) / (EI ** 2)
        total += dM_dEI
        
        print(f"{eid:^6} {M_primary[eid]:^+12.2f} {M_dual[eid]:^+12.4f} "
              f"{dM_dEI:^+18.6e}")
    
    print("-" * 60)
    print(f"{'TOTAL':^6} {' ':^12} {' ':^12} {total:^+18.6e}")
    print("-" * 60)
    
    # Example: 10% increase
    print(f"\n10% increase in EI → ΔM₂ ≈ {total * EI * 0.1:+.4f} N·m")
    
    return total

if __name__ == "__main__":
    compute_moment_sensitivity()