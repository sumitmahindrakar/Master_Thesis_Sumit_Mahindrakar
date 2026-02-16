import numpy as np
import matplotlib.pyplot as plt
import os
# import sys

# Set working directory to the project folder
os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\SA_Kratos_adj_V2_HDF5")
print(f"Working directory: {os.getcwd()}")

# Node coordinates (x positions)
x = np.array([0.000004, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# Displacement Z from VTK
disp_z = np.array([
    0.0000000e+00,
    1.5673381e-03,
    5.3966199e-03,
    1.0178316e-02,
    1.4602900e-02,
    1.7360844e-02,
    1.7460078e-02,
    1.5178377e-02,
    1.1110974e-02,
    5.8531039e-03,
    0.0000000e+00
])

# Rotation Y from VTK
rot_y = np.array([
    0.0000000e+00,
    -2.9165646e-02,
    -4.5237437e-02,
    -4.8213951e-02,
    -3.8095180e-02,
    -1.4881131e-02,
    1.1904391e-02,
    3.2737575e-02,
    4.7618419e-02,
    5.6546926e-02,
    5.9523094e-02
])

# Reaction Z from VTK
reaction_z = np.array([
    -2.7500090e+01,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -1.2499910e+01
])

# Reaction Moment Y from VTK
reaction_moment_y = np.array([
    7.4999800e+00,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
])

# Point Load Z from VTK
point_load_z = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0
])

# ---- Compute Shear Force and Bending Moment Diagrams ----
# Using equilibrium from left side
n_points = 200
x_cont = np.linspace(x[0], x[-1], n_points)

R1 = 27.5
M1 = -7.5  # clamped end moment (negative because reaction moment is positive = counterclockwise)
P = 40.0
x_load = 0.5

shear = np.zeros(n_points)
moment = np.zeros(n_points)

for i, xi in enumerate(x_cont):
    # Start from left
    V = R1
    M = M1

    # Add effect of point load if we've passed it
    if xi >= x_load:
        V -= P

    shear[i] = V
    moment[i] = M + R1 * xi - P * max(0, xi - x_load)

# ---- PLOTTING ----
fig, axes = plt.subplots(3, 2, figsize=(14, 15))
fig.suptitle('Primal Analysis Results — Propped Cantilever Beam\n'
             'E=2100, I=0.01, A=10, L=1.0, P=40 at x=0.5',
             fontsize=14, fontweight='bold')

# 1. Displacement Z
ax = axes[0, 0]
ax.plot(x, disp_z, 'b-o', linewidth=2, markersize=6)
ax.fill_between(x, disp_z, alpha=0.15, color='blue')
ax.set_xlabel('x [m]')
ax.set_ylabel('Displacement Z [m]')
ax.set_title('Vertical Displacement (Z)')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
# Mark max displacement
idx_max = np.argmax(disp_z)
ax.annotate(f'max = {disp_z[idx_max]:.6f} m\nat x = {x[idx_max]:.1f}',
            xy=(x[idx_max], disp_z[idx_max]),
            xytext=(x[idx_max] + 0.15, disp_z[idx_max] - 0.003),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')

# 2. Rotation Y
ax = axes[0, 1]
ax.plot(x, rot_y, 'r-o', linewidth=2, markersize=6)
ax.fill_between(x, rot_y, alpha=0.15, color='red')
ax.set_xlabel('x [m]')
ax.set_ylabel('Rotation Y [rad]')
ax.set_title('Rotation about Y-axis')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)

# 3. Shear Force Diagram
ax = axes[1, 0]
ax.plot(x_cont, shear, 'g-', linewidth=2)
ax.fill_between(x_cont, shear, alpha=0.15, color='green')
ax.set_xlabel('x [m]')
ax.set_ylabel('Shear Force [N]')
ax.set_title('Shear Force Diagram (V)')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.annotate(f'V_left = {R1:.1f} N', xy=(0.05, R1), fontsize=10, color='green')
ax.annotate(f'V_right = {R1 - P:.1f} N', xy=(0.6, R1 - P), fontsize=10, color='green')

# 4. Bending Moment Diagram
ax = axes[1, 1]
ax.plot(x_cont, moment, 'm-', linewidth=2)
ax.fill_between(x_cont, moment, alpha=0.15, color='purple')
ax.set_xlabel('x [m]')
ax.set_ylabel('Bending Moment [N·m]')
ax.set_title('Bending Moment Diagram (M)')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
# Mark max moment
idx_max_m = np.argmax(np.abs(moment))
ax.annotate(f'M_max = {moment[idx_max_m]:.2f} N·m',
            xy=(x_cont[idx_max_m], moment[idx_max_m]),
            xytext=(x_cont[idx_max_m] + 0.15, moment[idx_max_m] - 1),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')

# 5. Reactions
ax = axes[2, 0]
ax.bar([0, 1], [-reaction_z[0], -reaction_z[-1]], width=0.3,
       color=['steelblue', 'coral'], edgecolor='black')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Node 1 (Clamped)\nx=0.0', 'Node 11 (Pinned)\nx=1.0'])
ax.set_ylabel('Reaction Force Z [N]')
ax.set_title('Support Reactions')
ax.grid(True, alpha=0.3, axis='y')
for i, val in enumerate([-reaction_z[0], -reaction_z[-1]]):
    ax.text(i, val + 0.5, f'{val:.2f} N', ha='center', fontweight='bold', fontsize=11)

# 6. Beam Schematic
ax = axes[2, 1]
ax.set_xlim(-0.1, 1.15)
ax.set_ylim(-0.3, 0.4)
ax.set_aspect('equal')
ax.set_title('Beam Configuration')

# Draw beam
ax.plot([0, 1], [0, 0], 'k-', linewidth=4)

# Draw nodes
ax.plot(x, np.zeros_like(x), 'ko', markersize=5)

# Clamped support at node 1
ax.fill_between([-0.05, 0], [-0.15, -0.15], [0.15, 0.15], color='gray', alpha=0.5)
ax.plot([0, 0], [-0.15, 0.15], 'k-', linewidth=3)

# Pinned support at node 11
triangle_x = [0.95, 1.0, 1.05, 0.95]
triangle_y = [-0.1, 0.0, -0.1, -0.1]
ax.fill(triangle_x, triangle_y, color='gray', alpha=0.5)
ax.plot(triangle_x, triangle_y, 'k-', linewidth=2)

# Load arrow
ax.annotate('', xy=(0.5, 0.0), xytext=(0.5, 0.3),
            arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
ax.text(0.5, 0.32, 'P = 40 N', ha='center', fontsize=11, color='red', fontweight='bold')

# Labels
ax.text(0.0, -0.25, 'Fixed', ha='center', fontsize=9)
ax.text(1.0, -0.25, 'Pinned', ha='center', fontsize=9)
ax.set_xlabel('x [m]')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('primal_results.png', dpi=200, bbox_inches='tight')
plt.show()

print("\n===== RESULTS SUMMARY =====")
print(f"Max displacement Z: {np.max(disp_z):.6f} m at x = {x[np.argmax(disp_z)]:.1f} m")
print(f"Reaction at Node 1 (Z):  {-reaction_z[0]:.4f} N")
print(f"Reaction at Node 11 (Z): {-reaction_z[-1]:.4f} N")
print(f"Moment at Node 1 (Y):    {reaction_moment_y[0]:.4f} N·m")
print(f"Sum of reactions:         {-reaction_z[0] + (-reaction_z[-1]):.4f} N (should be 40.0)")
print(f"Applied load:             {P:.1f} N")
print("============================")