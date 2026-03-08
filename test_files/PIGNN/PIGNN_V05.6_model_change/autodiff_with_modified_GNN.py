from pathlib import Path

print(Path.cwd())

from pathlib import Path

CURRENT_SUBFOLDER = Path(__file__).resolve().parent
print(CURRENT_SUBFOLDER)



# Component    Loss Value    Issue
# ─────────    ──────────    ─────
# Axial:       ~10^7         ← ENORMOUS (EA ≈ 4.3×10⁹ amplifies everything)
# Bending:     ~1300         ← Stuck (overwhelmed by axial gradient)
# Kinematic:   ~0.004        ← Trivially small

# Problem: Optimizer only sees axial gradients.
# Bending never gets a chance to improve.Why? Your axial PDE residual is:

# r
# a
# x
# i
# a
# l
# =
# E
# A
# ⏟
# ≈
# 4.3
# ×
# 10
# 9
# ⋅
# d
# 2
# u
# x
# d
# x
# 2
# +
# f
# x
# r 
# axial
# ​
#  = 
# ≈4.3×10 
# 9
 
# EA
# ​
 
# ​
#  ⋅ 
# dx 
# 2
 
# d 
# 2
#  u 
# x
# ​
 
# ​
#  +f 
# x
# ​
 

# Even a tiny 
# d
# 2
# u
# x
# /
# d
# x
# 2
# ≈
# 10
# −
# 3
# d 
# 2
#  u 
# x
# ​
#  /dx 
# 2
#  ≈10 
# −3
#   gives 
# r
# ≈
# 4.3
# ×
# 10
# 6
# r≈4.3×10 
# 6
#  , and 
# r
# 2
# ≈
# 10
# 13
# r 
# 2
#  ≈10 
# 13
#  .

# The Fix: Non-Dimensionalize the PDEs
# Divide each PDE by its characteristic stiffness before squaring:

# text

# Before:  r = EA · d²u/dx² + f        →  r² ~ (EA)² · (d²u)² ~ 10^13
# After:   r = d²u/dx² + f/EA          →  r² ~ (d²u)²          ~ 10^-6
# Both PDEs become "strain-scale" quantities (~10⁻³ to 10⁻⁵), so the optimizer can balance them.