"""
Generate diagram for Simply Supported Beam with UDL
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_beam_system():
    """Draw a simply supported beam with UDL diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))#14,8
    
    # Beam parameters
    L = 2.0  # Length
    beam_height = 0.15
    beam_y = 2.0
    
    # Colors
    beam_color = '#2E86AB'
    load_color = '#E63946'
    support_color = '#1D3557'
    dimension_color = '#6C757D'
    
    # ==================== DRAW BEAM ====================
    beam = patches.FancyBboxPatch(
        (0, beam_y - beam_height/2), L, beam_height,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=beam_color, edgecolor='black', linewidth=2
    )
    ax.add_patch(beam)
    
    # ==================== DRAW UDL ====================
    n_arrows = 15
    arrow_spacing = L / (n_arrows + 1)
    arrow_length = 0.05
    load_top = beam_y + beam_height/2 + arrow_length + 0.15
    
    # UDL distribution line
    ax.plot([0+L/2 / (n_arrows + 1)-0.01, L-L / 2/(n_arrows + 1)+0.01], [load_top, load_top], color=load_color, linewidth=2)
    
    # UDL arrows
    for i in range(1, n_arrows + 1):
        x = i * arrow_spacing
        ax.annotate('', 
                   xy=(x, beam_y + beam_height/2 + 0.02),
                   xytext=(x, load_top),
                   arrowprops=dict(arrowstyle='->', color=load_color, lw=1.5))
    
    # End arrows for UDL
    ax.annotate('', xy=(0.05, beam_y + beam_height/2 + 0.02),
               xytext=(0.05, load_top),
               arrowprops=dict(arrowstyle='->', color=load_color, lw=1.5))
    ax.annotate('', xy=(L-0.05, beam_y + beam_height/2 + 0.02),
               xytext=(L-0.05, load_top),
               arrowprops=dict(arrowstyle='->', color=load_color, lw=1.5))
    
    # UDL label
    ax.text(L/2, load_top + 0.15, 'w = 10 kN/m', fontsize=14, ha='center', 
            fontweight='bold', color=load_color)
    
    # ==================== DRAW SUPPORTS ====================
    support_width = 0.15
    support_height = 0.15
    
    # Left support (pinned - triangle)
    left_support = plt.Polygon(
        [(0, beam_y - beam_height/2),
         (-support_width/2, beam_y - beam_height/2 - support_height),
         (support_width/2, beam_y - beam_height/2 - support_height)],
        facecolor=support_color, edgecolor='black', linewidth=2
    )
    ax.add_patch(left_support)
    
    # Left support ground line
    ax.plot([-support_width, support_width], 
            [beam_y - beam_height/2 - support_height - 0.02]*2,
            color='black', linewidth=2)
    
    # Left support hatching (ground)
    for i in range(5):
        x_start = -support_width + i * 0.08
        ax.plot([x_start, x_start + 0.06], 
                [beam_y - beam_height/2 - support_height - 0.02,
                 beam_y - beam_height/2 - support_height - 0.1],
                color='black', linewidth=1)
    
    # Right support (roller - triangle with circle)
    right_support = plt.Polygon(
        [(L, beam_y - beam_height/2),
         (L - support_width/2, beam_y - beam_height/2 - support_height),
         (L + support_width/2, beam_y - beam_height/2 - support_height)],
        facecolor=support_color, edgecolor='black', linewidth=2
    )
    ax.add_patch(right_support)
    
    # Roller circle
    roller = plt.Circle((L, beam_y - beam_height/2 - support_height - 0.06), 
                         0.05, facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(roller)
    
    # Right support ground line
    ax.plot([L - support_width, L + support_width], 
            [beam_y - beam_height/2 - support_height - 0.13]*2,
            color='black', linewidth=2)
    
    # Right support hatching
    for i in range(5):
        x_start = L - support_width + i * 0.08
        ax.plot([x_start, x_start + 0.06], 
                [beam_y - beam_height/2 - support_height - 0.13,
                 beam_y - beam_height/2 - support_height - 0.21],
                color='black', linewidth=1)
    
    # ==================== DRAW NODES ====================
    node_positions = [0, 2.0]  # Your 3 nodes
    for i, x in enumerate(node_positions):
        circle = plt.Circle((x, beam_y), 0.03, facecolor='black', 
                            edgecolor='black', linewidth=2, zorder=5)
        ax.add_patch(circle)
        # ax.text(x, beam_y - 0.35, f'Node {i+1}', fontsize=11, ha='center',
        #         fontweight='bold')
    
    # ==================== DIMENSIONS ====================
    dim_y = beam_y - 0.6
    
    # Total length dimension
    ax.annotate('', xy=(0, dim_y), xytext=(L, dim_y),
               arrowprops=dict(arrowstyle='<->', color=dimension_color, lw=1.5))
    ax.text(L/2, dim_y - 0.12, 'L = 2.0 m', fontsize=12, ha='center',
            color=dimension_color, fontweight='bold')
    
    # Element dimensions
    # elem_dim_y = beam_y - 1.1
    # ax.annotate('', xy=(0, elem_dim_y), xytext=(1.0, elem_dim_y),
    #            arrowprops=dict(arrowstyle='<->', color='gray', lw=1))
    # ax.text(0.5, elem_dim_y - 0.1, '1.0 m', fontsize=10, ha='center', color='gray')
    
    # ax.annotate('', xy=(1.0, elem_dim_y), xytext=(2.0, elem_dim_y),
    #            arrowprops=dict(arrowstyle='<->', color='gray', lw=1))
    # ax.text(1.5, elem_dim_y - 0.1, '1.0 m', fontsize=10, ha='center', color='gray')
    
    # ==================== ELEMENT LABELS ====================
    # ax.text(0.5, beam_y + 0.02, 'Element 1', fontsize=10, ha='center',
    #         color='white', fontweight='bold')
    # ax.text(1.5, beam_y + 0.02, 'Element 2', fontsize=10, ha='center',
            # color='white', fontweight='bold')
    
    # ==================== REACTIONS ====================
    reaction_y = beam_y - beam_height/2 - 0.5
    
    # Left reaction (Ry only, pinned)
    ax.annotate('', xy=(0, beam_y - beam_height/2 - 0.35),
               xytext=(0, reaction_y - 0.1),
               arrowprops=dict(arrowstyle='-', color='green', lw=2))
    # ax.text(0.15, reaction_y - 0.1, '$R_A$ = 10 kN', fontsize=11, color='green',
    #         fontweight='bold')
    
    # Right reaction (Ry only, roller)
    ax.annotate('', xy=(L, beam_y - beam_height/2 - 0.35),
               xytext=(L, reaction_y - 0.1),
               arrowprops=dict(arrowstyle='-', color='green', lw=2))
    # ax.text(L - 0.35, reaction_y - 0.2, '$R_B$ = 10 kN', fontsize=11, color='green',
    #         fontweight='bold')
    
    # ==================== COORDINATE SYSTEM ====================
    ax.annotate('', xy=(-0.75, 1.49), xytext=(-0.75, 2.0),
               arrowprops=dict(arrowstyle='<-', color='black', lw=1.5))
    ax.annotate('', xy=(-0.76, 1.5), xytext=(-0.26, 1.5),
               arrowprops=dict(arrowstyle='<-', color='black', lw=1.5))
    ax.text(-0.8, 2.0, 'Y', fontsize=11, fontweight='bold')
    ax.text(-0.26, 1.55, 'X', fontsize=11, fontweight='bold')
    
    # ==================== TITLE AND INFO BOX ====================
    ax.set_title('Simply Supported Beam with Uniformly Distributed Load (UDL)\n',
                fontsize=20, fontweight='bold')#16
    
    # Info box
    info_text = (
        'Beam Properties:\n'
        '─────────────────\n'
        'E = 210 GPa\n'
        'I = 5×10⁻⁶ m⁴\n'
        'L = 2.0 m\n'
        'A = 0.00287 m^2\n'
        'w = 10 kN/m\n'
    )
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray')
    ax.text(2.1, 3.2, info_text, fontsize=12, verticalalignment='top',
            fontfamily='monospace', bbox=props)#10
    
    # ==================== BOUNDARY CONDITIONS ====================
    bc_text = (
        'Boundary Conditions:\n'
        '─────────────────────\n'
        'Node 1 (x=0):\n'
        '  • $u_x$ = 0 (fixed)\n'
        '  • $u_y$ = 0 (fixed)\n'
        '  • $θ_z$ = free\n\n'
        'Node 3 (x=L):\n'
        '  • $u_x$ = free\n'
        '  • $u_y$ = 0 (fixed)\n'
        '  • $θ_z$ = free'
    )
    ax.text(-0.7, 3.2, bc_text, fontsize=12, verticalalignment='top',
            fontfamily='monospace', 
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9, edgecolor='gray'))#9
    
    # ==================== FORMATTING ====================
    ax.set_xlim(-1.25, 3.25)#-1, 3.5
    ax.set_ylim(1.0, 3.2)#0.0, 3.2
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_files/beam_2D_test_udl.gid/plots/beam_system_diagram.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: beam_system_diagram.png")
    
    plt.show()


def draw_analytical_diagrams():
    """Draw SFD, BMD, and deflection diagrams"""
    
    # Beam parameters
    L = 2.0
    w = 10000  # N/m
    E = 210e9
    I = 5e-6
    
    # Create x array
    x = np.linspace(0, L, 200)
    
    # Analytical solutions
    V = w * L / 2 - w * x  # Shear force
    M = w * L * x / 2 - w * x**2 / 2  # Bending moment
    delta = w * x * (L**3 - 2*L*x**2 + x**3) / (24 * E * I)  # Deflection
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    fig.suptitle('Simply Supported Beam with UDL - Analytical Diagrams', 
                 fontsize=14, fontweight='bold')
    
    # ==================== SHEAR FORCE DIAGRAM ====================
    # ax1 = axes[0]
    # ax1.fill_between(x, 0, V/1000, alpha=0.3, color='#E63946')
    # ax1.plot(x, V/1000, color='#E63946', linewidth=2)
    # ax1.axhline(y=0, color='black', linewidth=1)
    # ax1.set_ylabel('Shear Force (kN)', fontsize=11)
    # ax1.set_title('Shear Force Diagram (SFD)', fontsize=12, fontweight='bold')
    # ax1.grid(True, linestyle='--', alpha=0.7)
    
    # # Add values
    # ax1.annotate(f'+{w*L/2/1000:.1f} kN', xy=(0, V[0]/1000), xytext=(0.1, V[0]/1000 + 1),
    #             fontsize=10, fontweight='bold', color='#E63946')
    # ax1.annotate(f'{V[-1]/1000:.1f} kN', xy=(L, V[-1]/1000), xytext=(L-0.3, V[-1]/1000 - 1.5),
    #             fontsize=10, fontweight='bold', color='#E63946')
    # ax1.plot([L/2], [0], 'ko', markersize=8)
    # ax1.annotate('V = 0', xy=(L/2, 0), xytext=(L/2 + 0.1, 1), fontsize=10)
    
    # ==================== DEFLECTION DIAGRAM ====================
    ax2 = axes[0]#2
    ax2.fill_between(x, 0, -delta*1000, alpha=0.3, color='#2E86AB')
    ax2.plot(x, -delta*1000, color='#2E86AB', linewidth=2)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_xlabel('Position along beam (m)', fontsize=11)
    ax2.set_ylabel('Deflection (mm)', fontsize=11)
    ax2.set_title('Deflection Diagram', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add max deflection value
    max_delta = 5 * w * L**4 / (384 * E * I)
    ax2.plot([L/2], [-max_delta*1000], 'ko', markersize=8)
    ax2.annotate(f'$δ_{{max}}$ = {max_delta*1000:.4f} mm', xy=(L/2, -max_delta*1000), 
                xytext=(L/2 + 0.25, -max_delta*1000 ),#L/2 + 0.15, -max_delta*1000 - 0.3
                fontsize=11, fontweight='bold', color='#2E86AB')

    # ==================== BENDING MOMENT DIAGRAM ====================
    ax1 = axes[1]#1
    ax1.fill_between(x, 0, M/1000, alpha=0.3, color='#F18F01')
    ax1.plot(x, M/1000, color='#F18F01', linewidth=2)
    ax1.axhline(y=0, color='black', linewidth=1)
    ax1.set_ylabel('Bending Moment (kN·m)', fontsize=11)
    ax1.set_title('Bending Moment Diagram (BMD)', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add max moment value
    max_M = w * L**2 / 8
    ax1.plot([L/2], [max_M/1000], 'ko', markersize=8)
    ax1.annotate(f'$M_{{max}}$ = {max_M/1000:.2f} kN·m', xy=(L/2, max_M/1000), 
                xytext=(L/2 + 0.25, max_M/1000 - 0.25),#L/2 + 0.15, max_M/1000 + 0.3
                fontsize=11, fontweight='bold', color='#F18F01')
    
    
    # Add formulas
    formula_text = (
        'Analytical Formulas:\n'
        '────────────────────\n'
        f'$V(x) = wL/2 - wx$\n'
        f'$M(x) = wLx/2 - wx²/2$\n'
        f'$M_{{max}} = wL²/8$ = {max_M/1000:.2f} kN·m\n'
        f'$δ_{{max}} = 5wL⁴/(384EI)$ = {max_delta*1000:.4f} mm'
    )
    
    plt.tight_layout()
    plt.savefig('test_files/beam_2D_test_udl.gid/plots/beam_analytical_diagrams.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: beam_analytical_diagrams.png")
    
    plt.show()


def draw_complete_summary():
    """Draw complete beam summary with system and results"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.25)
    
    # Beam parameters
    L = 2.0
    w = 10000
    E = 210e9
    I = 5e-6
    
    x = np.linspace(0, L, 200)
    V = w * L / 2 - w * x
    M = w * L * x / 2 - w * x**2 / 2
    delta = w * x * (L**3 - 2*L*x**2 + x**3) / (24 * E * I)
    theta = w * (L**3 - 6*L*x**2 + 4*x**3) / (24 * E * I)
    
    # ==================== BEAM SYSTEM (TOP) ====================
    ax_beam = fig.add_subplot(gs[0, :])
    
    beam_y = 0.5
    beam_height = 0.08
    
    # Draw beam
    beam = patches.FancyBboxPatch(
        (0, beam_y - beam_height/2), L, beam_height,
        boxstyle="round,pad=0.005,rounding_size=0.01",
        facecolor='#2E86AB', edgecolor='black', linewidth=2
    )
    ax_beam.add_patch(beam)
    
    # Draw UDL
    n_arrows = 20
    for i in range(n_arrows + 1):
        xi = i * L / n_arrows
        ax_beam.annotate('', xy=(xi, beam_y + beam_height/2 + 0.01),
                        xytext=(xi, beam_y + beam_height/2 + 0.2),
                        arrowprops=dict(arrowstyle='->', color='#E63946', lw=1))
    ax_beam.plot([0, L], [beam_y + beam_height/2 + 0.2]*2, color='#E63946', lw=2)
    ax_beam.text(L/2, beam_y + beam_height/2 + 0.28, 'w = 10 kN/m', 
                ha='center', fontsize=12, fontweight='bold', color='#E63946')
    
    # Supports
    support_size = 0.06
    # Left pinned
    left_tri = plt.Polygon([(0, beam_y - beam_height/2),
                            (-support_size, beam_y - beam_height/2 - support_size*1.5),
                            (support_size, beam_y - beam_height/2 - support_size*1.5)],
                           facecolor='#1D3557', edgecolor='black', lw=2)
    ax_beam.add_patch(left_tri)
    
    # Right roller
    right_tri = plt.Polygon([(L, beam_y - beam_height/2),
                             (L-support_size, beam_y - beam_height/2 - support_size*1.5),
                             (L+support_size, beam_y - beam_height/2 - support_size*1.5)],
                            facecolor='#1D3557', edgecolor='black', lw=2)
    ax_beam.add_patch(right_tri)
    roller = plt.Circle((L, beam_y - beam_height/2 - support_size*1.5 - 0.025), 
                        0.02, facecolor='white', edgecolor='black', lw=2)
    ax_beam.add_patch(roller)
    
    # Nodes
    for i, xi in enumerate([0, 1, 2]):
        ax_beam.plot(xi, beam_y, 'o', markersize=10, 
                    markerfacecolor='white', markeredgecolor='black', markeredgewidth=2)
        ax_beam.text(xi, beam_y - 0.18, f'{i+1}', ha='center', fontsize=11, fontweight='bold')
    
    # Dimension
    ax_beam.annotate('', xy=(0, beam_y - 0.3), xytext=(L, beam_y - 0.3),
                    arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax_beam.text(L/2, beam_y - 0.38, 'L = 2.0 m', ha='center', fontsize=11, color='gray')
    
    # Reactions
    ax_beam.annotate('', xy=(0, beam_y - beam_height/2 - 0.12),
                    xytext=(0, beam_y - beam_height/2 - 0.35),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax_beam.text(0.08, beam_y - beam_height/2 - 0.28, '$R_A$=10kN', fontsize=10, color='green')
    
    ax_beam.annotate('', xy=(L, beam_y - beam_height/2 - 0.18),
                    xytext=(L, beam_y - beam_height/2 - 0.42),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax_beam.text(L-0.22, beam_y - beam_height/2 - 0.35, '$R_B$=10kN', fontsize=10, color='green')
    
    ax_beam.set_xlim(-0.4, 2.8)
    ax_beam.set_ylim(-0.1, 1.0)
    ax_beam.set_aspect('equal')
    ax_beam.axis('off')
    ax_beam.set_title('Simply Supported Beam with UDL', fontsize=14, fontweight='bold')
    
    # Properties box
    props_text = 'E = 210 GPa\nI = 5×10⁻⁶ m⁴\nL = 2.0 m\nw = 10 kN/m'
    ax_beam.text(2.35, 0.7, props_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
                verticalalignment='top', fontfamily='monospace')
    
    # ==================== SHEAR FORCE ====================
    ax_shear = fig.add_subplot(gs[1, 0])
    ax_shear.fill_between(x, 0, V/1000, alpha=0.3, color='#E63946')
    ax_shear.plot(x, V/1000, color='#E63946', linewidth=2.5)
    ax_shear.axhline(y=0, color='black', linewidth=1)
    ax_shear.set_xlabel('Position (m)', fontsize=11)
    ax_shear.set_ylabel('Shear Force (kN)', fontsize=11)
    ax_shear.set_title('Shear Force Diagram (SFD)', fontsize=12, fontweight='bold')
    ax_shear.grid(True, linestyle='--', alpha=0.7)
    ax_shear.text(0.1, 8, '+10 kN', fontsize=10, fontweight='bold', color='#E63946')
    ax_shear.text(1.7, -8, '-10 kN', fontsize=10, fontweight='bold', color='#E63946')
    
    # ==================== BENDING MOMENT ====================
    ax_moment = fig.add_subplot(gs[1, 1])
    ax_moment.fill_between(x, 0, M/1000, alpha=0.3, color='#F18F01')
    ax_moment.plot(x, M/1000, color='#F18F01', linewidth=2.5)
    ax_moment.axhline(y=0, color='black', linewidth=1)
    ax_moment.set_xlabel('Position (m)', fontsize=11)
    ax_moment.set_ylabel('Bending Moment (kN·m)', fontsize=11)
    ax_moment.set_title('Bending Moment Diagram (BMD)', fontsize=12, fontweight='bold')
    ax_moment.grid(True, linestyle='--', alpha=0.7)
    ax_moment.annotate(f'$M_{{max}}$ = 5.0 kN·m', xy=(1, 5), xytext=(1.2, 4.2),
                      fontsize=11, fontweight='bold', color='#F18F01',
                      arrowprops=dict(arrowstyle='->', color='#F18F01'))
    
    # ==================== DEFLECTION ====================
    ax_defl = fig.add_subplot(gs[2, 0])
    ax_defl.fill_between(x, 0, -delta*1000, alpha=0.3, color='#2E86AB')
    ax_defl.plot(x, -delta*1000, color='#2E86AB', linewidth=2.5)
    ax_defl.axhline(y=0, color='black', linewidth=1)
    ax_defl.set_xlabel('Position (m)', fontsize=11)
    ax_defl.set_ylabel('Deflection (mm)', fontsize=11)
    ax_defl.set_title('Deflection Diagram', fontsize=12, fontweight='bold')
    ax_defl.grid(True, linestyle='--', alpha=0.7)
    max_delta = 5 * w * L**4 / (384 * E * I)
    ax_defl.annotate(f'$δ_{{max}}$ = {max_delta*1000:.4f} mm', 
                    xy=(1, -max_delta*1000), xytext=(1.2, -max_delta*1000*0.5),
                    fontsize=11, fontweight='bold', color='#2E86AB',
                    arrowprops=dict(arrowstyle='->', color='#2E86AB'))
    
    # ==================== ROTATION ====================
    ax_rot = fig.add_subplot(gs[2, 1])
    ax_rot.fill_between(x, 0, theta*1000, alpha=0.3, color='#A23B72')
    ax_rot.plot(x, theta*1000, color='#A23B72', linewidth=2.5)
    ax_rot.axhline(y=0, color='black', linewidth=1)
    ax_rot.set_xlabel('Position (m)', fontsize=11)
    ax_rot.set_ylabel('Rotation (mrad)', fontsize=11)
    ax_rot.set_title('Rotation Diagram', fontsize=12, fontweight='bold')
    ax_rot.grid(True, linestyle='--', alpha=0.7)
    end_theta = w * L**3 / (24 * E * I)
    ax_rot.annotate(f'$θ_A$ = {end_theta*1000:.4f} mrad', 
                   xy=(0, end_theta*1000), xytext=(0.3, end_theta*1000*0.7),
                   fontsize=10, fontweight='bold', color='#A23B72')
    ax_rot.annotate(f'$θ_B$ = -{end_theta*1000:.4f} mrad', 
                   xy=(2, -end_theta*1000), xytext=(1.3, -end_theta*1000*0.7),
                   fontsize=10, fontweight='bold', color='#A23B72')
    
    plt.suptitle('Simply Supported Beam Analysis Summary', fontsize=16, fontweight='bold', y=0.98)
    
    # plt.savefig('test_files/beam_2D_test_udl.gid/plots/beam_complete_summary.png', dpi=200, bbox_inches='tight',
    #             facecolor='white', edgecolor='none')
    # print("Saved: beam_complete_summary.png")
    
    # plt.show()


if __name__ == "__main__":
    print("Generating beam diagrams...")
    print("-" * 50)
    
    # Generate all diagrams
    draw_beam_system()
    draw_analytical_diagrams()
    draw_complete_summary()
    
    print("\nAll diagrams generated successfully!")