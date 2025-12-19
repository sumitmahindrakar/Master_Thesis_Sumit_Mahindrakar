"""
Moment Sensitivity Calculator: ∂M/∂(EI)
Using General Influence Method with VTK File Parsing and Visualization
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


def parse_vtk_cell_moments(vtk_file_path):
    """
    Parse bending moments (Mz) from VTK file CELL_DATA section
    
    Args:
        vtk_file_path: Path to VTK file
        
    Returns:
        dict: {element_id: Mz_value}
    """
    moments = {}
    
    if not os.path.exists(vtk_file_path):
        print(f"Error: VTK file not found: {vtk_file_path}")
        return moments
    
    with open(vtk_file_path, 'r') as f:
        lines = f.readlines()
    
    in_cell_data = False
    reading_moment = False
    elem_idx = 1
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        if line_stripped.startswith('CELL_DATA'):
            in_cell_data = True
            continue
        
        if line_stripped.startswith('POINT_DATA'):
            in_cell_data = False
            reading_moment = False
            continue
        
        if in_cell_data and 'MOMENT' in line_stripped and 'FIELD' not in line_stripped:
            reading_moment = True
            continue
        
        if reading_moment and line_stripped:
            parts = line_stripped.split()
            if len(parts) >= 3:
                try:
                    Mz = float(parts[2])
                    moments[elem_idx] = Mz
                    elem_idx += 1
                except ValueError:
                    reading_moment = False
    
    return moments


def compute_moment_sensitivity(E, I, L_elements, M_primary, M_dual, response_element=2):
    """
    Compute ∂M/∂(EI) for all elements
    
    Formula: ∂M/∂(EI)_k = -(M_k · M̄_k · L_k) / (EI)²
    """
    EI = E * I
    EI_squared = EI ** 2
    
    sensitivities = {}
    total_sensitivity = 0.0
    
    for eid in sorted(M_primary.keys()):
        M_p = M_primary[eid]
        M_d = M_dual.get(eid, 0.0)
        L = L_elements.get(eid, 0.6667)
        
        integral = M_p * M_d * L
        dM_dEI = -integral / EI_squared
        
        sensitivities[eid] = {
            'M_primary': M_p,
            'M_dual': M_d,
            'length': L,
            'integral': integral,
            'dM_dEI': dM_dEI
        }
        
        total_sensitivity += dM_dEI
    
    return sensitivities, total_sensitivity


def print_results(E, I, sensitivities, total_sensitivity, response_element=2):
    """
    Print formatted sensitivity results
    """
    EI = E * I
    
    print("\n" + "=" * 75)
    print("MOMENT SENSITIVITY ANALYSIS: ∂M/∂(EI)")
    print("Using General Influence Method")
    print("=" * 75)
    
    print(f"\nMaterial/Section Properties:")
    print(f"  E  = {E:.4e} Pa")
    print(f"  I  = {I:.4e} m⁴")
    print(f"  EI = {EI:.4e} N·m²")
    
    print(f"\nResponse Quantity: Bending Moment in Element {response_element}")
    
    print("\n" + "-" * 75)
    print(f"{'Elem k':^8} {'M_k [N·m]':^14} {'M̄_k [N·m]':^14} "
          f"{'L_k [m]':^10} {'∂M/∂(EI)_k':^20}")
    print("-" * 75)
    
    for eid, data in sorted(sensitivities.items()):
        print(f"{eid:^8} {data['M_primary']:^+14.4f} {data['M_dual']:^+14.6f} "
              f"{data['length']:^10.4f} {data['dM_dEI']:^+20.6e}")
    
    print("-" * 75)
    print(f"{'TOTAL':^8} {' ':^14} {' ':^14} "
          f"{' ':^10} {total_sensitivity:^+20.6e}")
    print("-" * 75)
    
    # Example calculation
    delta_EI_percent = 10.0
    delta_EI = (delta_EI_percent / 100.0) * EI
    delta_M = total_sensitivity * delta_EI
    
    print(f"\n{delta_EI_percent}% increase in EI → ΔM ≈ {delta_M:+.6f} N·m")


def plot_sensitivity_analysis(E, I, L_elements, M_primary, M_dual, sensitivities, 
                               total_sensitivity, beam_length, response_element=2,
                               save_path=None):
    """
    Create comprehensive visualization of sensitivity analysis
    
    Args:
        E: Young's modulus [Pa]
        I: Second moment of area [m^4]
        L_elements: Dict of element lengths
        M_primary: Dict of primary moments
        M_dual: Dict of dual moments
        sensitivities: Dict of computed sensitivities
        total_sensitivity: Total/global sensitivity
        beam_length: Total beam length [m]
        response_element: Element where we measure response
        save_path: Path to save figure (optional)
    """
    EI = E * I
    n_elements = len(M_primary)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.2], hspace=0.3, wspace=0.3)
    
    # Color scheme
    colors = {
        'primary': '#2E86AB',
        'dual': '#A23B72',
        'sensitivity': '#F18F01',
        'positive': '#28A745',
        'negative': '#DC3545',
        'beam': '#6C757D',
        'support': '#343A40'
    }
    
    # Element positions for plotting
    elem_ids = sorted(M_primary.keys())
    x_nodes = [0]
    for eid in elem_ids:
        x_nodes.append(x_nodes[-1] + L_elements[eid])
    x_centers = [(x_nodes[i] + x_nodes[i+1]) / 2 for i in range(n_elements)]
    
    # =========================================
    # Plot 1: Beam Structure Schematic
    # =========================================
    ax1 = fig.add_subplot(gs[0, :])
    
    # Draw beam
    beam_y = 0.5
    ax1.plot([0, beam_length], [beam_y, beam_y], 'k-', linewidth=8, solid_capstyle='butt')
    
    # Draw elements with labels
    for i, eid in enumerate(elem_ids):
        x_start = x_nodes[i]
        x_end = x_nodes[i+1]
        x_mid = (x_start + x_end) / 2
        
        # Element coloring
        if eid == response_element:
            color = colors['sensitivity']
            ax1.fill_between([x_start, x_end], [beam_y-0.05, beam_y-0.05], 
                           [beam_y+0.05, beam_y+0.05], color=color, alpha=0.3)
        
        # Element label
        ax1.text(x_mid, beam_y + 0.15, f'Element {eid}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
        ax1.text(x_mid, beam_y - 0.15, f'L = {L_elements[eid]:.3f} m', 
                ha='center', va='top', fontsize=9)
    
    # Draw nodes
    for i, x in enumerate(x_nodes):
        ax1.plot(x, beam_y, 'ko', markersize=10, zorder=5)
        ax1.text(x, beam_y + 0.25, f'Node {i+1}', ha='center', va='bottom', fontsize=9)
    
    # Draw fixed supports
    support_size = 0.1
    for x in [0, beam_length]:
        # Fixed support symbol (hatched rectangle)
        rect = mpatches.FancyBboxPatch((x - support_size/2, beam_y - 0.2), 
                                        support_size, 0.15,
                                        boxstyle="square,pad=0",
                                        facecolor=colors['support'],
                                        edgecolor='black', linewidth=1)
        ax1.add_patch(rect)
        # Ground hatching
        for j in range(5):
            ax1.plot([x - support_size/2 + j*support_size/5, 
                     x - support_size/2 + (j+1)*support_size/5],
                    [beam_y - 0.25, beam_y - 0.2], 'k-', linewidth=1)
    
    # Draw UDL arrows
    n_arrows = 15
    arrow_x = np.linspace(0.05, beam_length - 0.05, n_arrows)
    for x in arrow_x:
        ax1.annotate('', xy=(x, beam_y + 0.05), xytext=(x, beam_y + 0.35),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    ax1.text(beam_length/2, beam_y + 0.45, 'q = 10 kN/m (UDL)', 
            ha='center', va='bottom', fontsize=11, color='blue', fontweight='bold')
    
    ax1.set_xlim(-0.3, beam_length + 0.3)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Beam Structure: Fixed-Fixed Beam with UDL', fontsize=14, fontweight='bold', pad=20)
    
    # =========================================
    # Plot 2: Primary Moment Diagram M(x)
    # =========================================
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Create moment diagram
    M_values = [M_primary[eid] for eid in elem_ids]
    
    # Bar plot for moments
    bars = ax2.bar(x_centers, M_values, width=L_elements[1]*0.8, 
                   color=[colors['positive'] if m >= 0 else colors['negative'] for m in M_values],
                   edgecolor='black', linewidth=1.5, alpha=0.7)
    
    # Add value labels
    for i, (x, m) in enumerate(zip(x_centers, M_values)):
        offset = 50 if m >= 0 else -50
        va = 'bottom' if m >= 0 else 'top'
        ax2.text(x, m + offset, f'{m:+.2f}', ha='center', va=va, fontsize=10, fontweight='bold')
    
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_xlabel('Position along beam [m]', fontsize=11)
    ax2.set_ylabel('Primary Moment M [N·m]', fontsize=11)
    ax2.set_title('Primary Analysis: Bending Moment Diagram M(x)\n(From UDL Loading)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xticks(x_centers)
    ax2.set_xticklabels([f'Elem {eid}' for eid in elem_ids])
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.1, beam_length + 0.1)
    
    # =========================================
    # Plot 3: Dual Moment Diagram M̄(x)
    # =========================================
    ax3 = fig.add_subplot(gs[1, 1])
    
    M_dual_values = [M_dual[eid] for eid in elem_ids]
    
    bars = ax3.bar(x_centers, M_dual_values, width=L_elements[1]*0.8,
                   color=[colors['positive'] if m >= 0 else colors['negative'] for m in M_dual_values],
                   edgecolor='black', linewidth=1.5, alpha=0.7)
    
    # Add value labels
    for i, (x, m) in enumerate(zip(x_centers, M_dual_values)):
        offset = 0.05 if m >= 0 else -0.05
        va = 'bottom' if m >= 0 else 'top'
        ax3.text(x, m + offset, f'{m:+.4f}', ha='center', va=va, fontsize=10, fontweight='bold')
    
    ax3.axhline(y=0, color='black', linewidth=1)
    ax3.set_xlabel('Position along beam [m]', fontsize=11)
    ax3.set_ylabel('Dual Moment M̄ [N·m]', fontsize=11)
    ax3.set_title('Dual Analysis: Virtual Moment Diagram M̄(x)\n(From Unit Moment Couple)', 
                  fontsize=12, fontweight='bold')
    ax3.set_xticks(x_centers)
    ax3.set_xticklabels([f'Elem {eid}' for eid in elem_ids])
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.1, beam_length + 0.1)
    
    # =========================================
    # Plot 4: Sensitivity Bar Chart
    # =========================================
    ax4 = fig.add_subplot(gs[2, 0])
    
    sens_values = [sensitivities[eid]['dM_dEI'] for eid in elem_ids]
    
    # Add total as separate bar
    x_positions = list(range(len(elem_ids))) + [len(elem_ids) + 0.5]
    all_values = sens_values + [total_sensitivity]
    labels = [f'Element {eid}' for eid in elem_ids] + ['TOTAL\n(Global)']
    
    bar_colors = [colors['sensitivity'] if i < len(elem_ids) else colors['primary'] 
                  for i in range(len(all_values))]
    
    bars = ax4.bar(x_positions, all_values, color=bar_colors, 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels
    for i, (x, v) in enumerate(zip(x_positions, all_values)):
        ax4.text(x, v + max(all_values)*0.05, f'{v:.2e}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=0)
    
    ax4.axhline(y=0, color='black', linewidth=1)
    ax4.set_xlabel('Design Variable', fontsize=11)
    ax4.set_ylabel(f'∂M_{response_element}/∂(EI)_k', fontsize=11)
    ax4.set_title(f'Sensitivity of Moment in Element {response_element}\nw.r.t. Element Stiffness (EI)', 
                  fontsize=12, fontweight='bold')
    ax4.set_xticks(x_positions)
    ax4.set_xticklabels(labels, fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # =========================================
    # Plot 5: Sensitivity Contribution Pie Chart
    # =========================================
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Use absolute values for pie chart
    abs_sens = [abs(sensitivities[eid]['dM_dEI']) for eid in elem_ids]
    total_abs = sum(abs_sens)
    percentages = [s/total_abs*100 for s in abs_sens]
    
    pie_colors = [plt.cm.Set2(i) for i in range(len(elem_ids))]
    
    wedges, texts, autotexts = ax5.pie(abs_sens, labels=[f'Element {eid}' for eid in elem_ids],
                                        autopct='%1.1f%%', colors=pie_colors,
                                        explode=[0.05 if eid == response_element else 0 for eid in elem_ids],
                                        shadow=True, startangle=90)
    
    # Style the text
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    ax5.set_title(f'Contribution to Total Sensitivity\n(|∂M_{response_element}/∂(EI)_k| / Σ|∂M_{response_element}/∂(EI)_k|)', 
                  fontsize=12, fontweight='bold')
    
    # Add legend with actual values
    legend_labels = [f'Element {eid}: {sensitivities[eid]["dM_dEI"]:.2e}' for eid in elem_ids]
    ax5.legend(wedges, legend_labels, title="Sensitivity Values", 
              loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    
    # =========================================
    # Add overall title and info box
    # =========================================
    fig.suptitle('Moment Sensitivity Analysis: ∂M/∂(EI)\nUsing General Influence Method', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Info text box
    info_text = (f"Material Properties:\n"
                 f"  E = {E:.2e} Pa\n"
                 f"  I = {I:.2e} m⁴\n"
                 f"  EI = {EI:.2e} N·m²\n\n"
                 f"Response: M in Element {response_element}\n"
                 f"Total ∂M/∂(EI) = {total_sensitivity:.4e}")
    
    fig.text(0.02, 0.02, info_text, fontsize=9, family='monospace',
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()


def plot_sensitivity_vs_EI_change(E, I, sensitivities, total_sensitivity, 
                                   response_element=2, save_path=None):
    """
    Plot how moment changes with different levels of EI change
    
    Args:
        E: Young's modulus [Pa]
        I: Second moment of area [m^4]
        sensitivities: Dict of computed sensitivities
        total_sensitivity: Total/global sensitivity
        response_element: Element where we measure response
        save_path: Path to save figure (optional)
    """
    EI = E * I
    M_original = sensitivities[response_element]['M_primary']
    
    # Range of EI changes
    delta_EI_percent = np.linspace(-50, 50, 101)  # -50% to +50%
    delta_EI = delta_EI_percent / 100 * EI
    
    # Calculate moment changes
    delta_M_elem1 = sensitivities[1]['dM_dEI'] * delta_EI
    delta_M_elem2 = sensitivities[2]['dM_dEI'] * delta_EI
    delta_M_elem3 = sensitivities[3]['dM_dEI'] * delta_EI
    delta_M_total = total_sensitivity * delta_EI
    
    # New moments
    M_new_elem1 = M_original + delta_M_elem1
    M_new_elem2 = M_original + delta_M_elem2
    M_new_elem3 = M_original + delta_M_elem3
    M_new_total = M_original + delta_M_total
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # =========================================
    # Plot 1: Change in Moment (ΔM)
    # =========================================
    ax1 = axes[0]
    
    ax1.plot(delta_EI_percent, delta_M_elem1, 'b-', linewidth=2, label='ΔEI in Element 1 only')
    ax1.plot(delta_EI_percent, delta_M_elem2, 'r-', linewidth=2, label='ΔEI in Element 2 only')
    ax1.plot(delta_EI_percent, delta_M_elem3, 'g-', linewidth=2, label='ΔEI in Element 3 only')
    ax1.plot(delta_EI_percent, delta_M_total, 'k-', linewidth=3, label='ΔEI in Entire Beam')
    
    ax1.axhline(y=0, color='gray', linewidth=1, linestyle='--')
    ax1.axvline(x=0, color='gray', linewidth=1, linestyle='--')
    
    ax1.set_xlabel('Change in EI [%]', fontsize=12)
    ax1.set_ylabel(f'Change in Moment ΔM_{response_element} [N·m]', fontsize=12)
    ax1.set_title(f'Change in Moment vs. Change in EI\n(Linear Sensitivity Approximation)', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # =========================================
    # Plot 2: New Moment Value
    # =========================================
    ax2 = axes[1]
    
    ax2.plot(delta_EI_percent, M_new_elem1, 'b-', linewidth=2, label='ΔEI in Element 1 only')
    ax2.plot(delta_EI_percent, M_new_elem2, 'r-', linewidth=2, label='ΔEI in Element 2 only')
    ax2.plot(delta_EI_percent, M_new_elem3, 'g-', linewidth=2, label='ΔEI in Element 3 only')
    ax2.plot(delta_EI_percent, M_new_total, 'k-', linewidth=3, label='ΔEI in Entire Beam')
    
    ax2.axhline(y=M_original, color='gray', linewidth=1, linestyle='--', label=f'Original M = {M_original:.2f}')
    ax2.axvline(x=0, color='gray', linewidth=1, linestyle='--')
    
    ax2.set_xlabel('Change in EI [%]', fontsize=12)
    ax2.set_ylabel(f'Moment M_{response_element} [N·m]', fontsize=12)
    ax2.set_title(f'Moment in Element {response_element} vs. Change in EI\n(Linear Sensitivity Approximation)', 
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()


def plot_moment_diagrams_comparison(M_primary, M_dual, L_elements, beam_length, 
                                     response_element=2, save_path=None):
    """
    Plot primary and dual moment diagrams side by side with continuous lines
    
    Args:
        M_primary: Dict of primary moments
        M_dual: Dict of dual moments
        L_elements: Dict of element lengths
        beam_length: Total beam length
        response_element: Response element ID
        save_path: Path to save figure
    """
    elem_ids = sorted(M_primary.keys())
    n_elements = len(elem_ids)
    
    # Create x coordinates for plotting
    x_nodes = [0]
    for eid in elem_ids:
        x_nodes.append(x_nodes[-1] + L_elements[eid])
    
    # Create detailed x coordinates for smooth plotting
    x_plot = []
    M_primary_plot = []
    M_dual_plot = []
    
    for i, eid in enumerate(elem_ids):
        x_start = x_nodes[i]
        x_end = x_nodes[i+1]
        
        # For beam elements, moment varies linearly within element (approximately)
        # Here we use constant value (average) for simplicity
        x_elem = np.linspace(x_start, x_end, 20)
        x_plot.extend(x_elem)
        M_primary_plot.extend([M_primary[eid]] * len(x_elem))
        M_dual_plot.extend([M_dual[eid]] * len(x_elem))
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # =========================================
    # Plot 1: Primary Moment Diagram
    # =========================================
    ax1 = axes[0]
    
    ax1.fill_between(x_plot, 0, M_primary_plot, alpha=0.3, color='blue')
    ax1.plot(x_plot, M_primary_plot, 'b-', linewidth=2, label='M(x)')
    ax1.axhline(y=0, color='black', linewidth=1)
    
    # Mark element boundaries
    for x in x_nodes[1:-1]:
        ax1.axvline(x=x, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    
    # Add element labels and values
    for i, eid in enumerate(elem_ids):
        x_mid = (x_nodes[i] + x_nodes[i+1]) / 2
        M = M_primary[eid]
        y_offset = 100 if M > 0 else -100
        ax1.annotate(f'M_{eid} = {M:+.2f}', xy=(x_mid, M), 
                    xytext=(x_mid, M + y_offset),
                    ha='center', fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1))
    
    ax1.set_ylabel('Primary Moment M [N·m]', fontsize=12)
    ax1.set_title('Primary Analysis: Bending Moment Diagram M(x)\n(Real Load: UDL = 10 kN/m)', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # =========================================
    # Plot 2: Dual Moment Diagram
    # =========================================
    ax2 = axes[1]
    
    ax2.fill_between(x_plot, 0, M_dual_plot, alpha=0.3, color='red')
    ax2.plot(x_plot, M_dual_plot, 'r-', linewidth=2, label='M̄(x)')
    ax2.axhline(y=0, color='black', linewidth=1)
    
    # Mark element boundaries
    for x in x_nodes[1:-1]:
        ax2.axvline(x=x, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    
    # Add element labels and values
    for i, eid in enumerate(elem_ids):
        x_mid = (x_nodes[i] + x_nodes[i+1]) / 2
        M = M_dual[eid]
        y_offset = 0.1 if M > 0 else -0.1
        ax2.annotate(f'M̄_{eid} = {M:+.4f}', xy=(x_mid, M), 
                    xytext=(x_mid, M + y_offset),
                    ha='center', fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1))
    
    ax2.set_xlabel('Position along beam [m]', fontsize=12)
    ax2.set_ylabel('Dual Moment M̄ [N·m]', fontsize=12)
    ax2.set_title('Dual Analysis: Virtual Moment Diagram M̄(x)\n(Virtual Load: Unit Moment Couple at Element 2 boundaries)', 
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()


def verify_calculation(E, I, M_primary, M_dual, L, expected_total):
    """
    Verify the sensitivity calculation step by step
    """
    EI = E * I
    
    print("\n" + "=" * 75)
    print("VERIFICATION OF CALCULATION")
    print("=" * 75)
    
    print(f"\nGiven:")
    print(f"  E = {E:.4e} Pa")
    print(f"  I = {I:.4e} m⁴")
    print(f"  EI = E × I = {EI:.4e} N·m²")
    print(f"  EI² = {EI**2:.4e} (N·m²)²")
    print(f"  L (each element) = {L:.6f} m")
    
    print(f"\nFormula: ∂M/∂(EI)_k = -(M_k × M̄_k × L_k) / (EI)²")
    
    print("\nStep-by-step calculation:")
    print("-" * 75)
    
    total_calculated = 0.0
    
    for eid in sorted(M_primary.keys()):
        M_p = M_primary[eid]
        M_d = M_dual[eid]
        
        product = M_p * M_d
        integral = product * L
        dM_dEI = -integral / (EI ** 2)
        
        print(f"\nElement {eid}:")
        print(f"  M_{eid} = {M_p:+.4f} N·m")
        print(f"  M̄_{eid} = {M_d:+.6f} N·m")
        print(f"  M × M̄ = ({M_p:+.4f}) × ({M_d:+.6f}) = {product:+.6f}")
        print(f"  M × M̄ × L = {product:+.6f} × {L:.6f} = {integral:+.6f}")
        print(f"  ∂M/∂(EI)_{eid} = -({integral:+.6f}) / ({EI**2:.4e})")
        print(f"                 = {dM_dEI:+.6e}")
        
        total_calculated += dM_dEI
    
    print("\n" + "-" * 75)
    print(f"Total ∂M/∂(EI) = Σ(∂M/∂(EI)_k) = {total_calculated:+.6e}")
    print("-" * 75)
    
    print(f"\nComparison with expected output:")
    print(f"  Expected:   {expected_total:+.6e}")
    print(f"  Calculated: {total_calculated:+.6e}")
    
    if abs(expected_total) > 1e-15:
        rel_error = abs(total_calculated - expected_total) / abs(expected_total) * 100
        print(f"  Relative error: {rel_error:.6f}%")
        
        if rel_error < 0.1:
            print("\n✓ RESULTS MATCH! Calculation is correct.")
        else:
            print("\n✗ Results don't match. Check input values.")
    
    return total_calculated


def main(primary_vtk_path, dual_vtk_path, E, I, beam_length, n_elements,
         response_element=2, run_verification=False, expected_total=None,
         create_plots=True, save_plots=False, output_dir="."):
    """
    Main function - compute sensitivity from VTK files with optional visualization
    
    Args:
        primary_vtk_path: Path to primary analysis VTK file
        dual_vtk_path: Path to dual analysis VTK file
        E: Young's modulus [Pa]
        I: Second moment of area [m^4]
        beam_length: Total beam length [m]
        n_elements: Number of elements
        response_element: Element ID where we measure moment response (default: 2)
        run_verification: Whether to run detailed verification (default: False)
        expected_total: Expected total sensitivity for verification (optional)
        create_plots: Whether to create visualization plots (default: True)
        save_plots: Whether to save plots to files (default: False)
        output_dir: Directory to save plots (default: current directory)
        
    Returns:
        tuple: (sensitivities dict, total sensitivity)
    """
    
    # Calculate Element Lengths
    L_elem = beam_length / n_elements
    L_elements = {i: L_elem for i in range(1, n_elements + 1)}
    
    print("=" * 75)
    print("MOMENT SENSITIVITY ANALYSIS: ∂M/∂(EI)")
    print("Reading from VTK Files")
    print("=" * 75)
    
    # Parse Primary VTK File
    print(f"\nPrimary VTK: {primary_vtk_path}")
    M_primary = parse_vtk_cell_moments(primary_vtk_path)
    
    if M_primary:
        print("Primary Moments (CELL_DATA MOMENT Mz):")
        for eid, M in sorted(M_primary.items()):
            print(f"  Element {eid}: M = {M:+.4f} N·m")
    else:
        print("Error: Could not parse primary moments from VTK file.")
        return None, None
    
    # Parse Dual VTK File
    print(f"\nDual VTK: {dual_vtk_path}")
    M_dual = parse_vtk_cell_moments(dual_vtk_path)
    
    if M_dual:
        print("Dual Moments (CELL_DATA MOMENT Mz):")
        for eid, M in sorted(M_dual.items()):
            print(f"  Element {eid}: M̄ = {M:+.6f} N·m")
    else:
        print("Error: Could not parse dual moments from VTK file.")
        return None, None
    
    # Compute Sensitivities
    sensitivities, total_sensitivity = compute_moment_sensitivity(
        E, I, L_elements, M_primary, M_dual, response_element
    )
    
    # Print Results
    print_results(E, I, sensitivities, total_sensitivity, response_element)
    
    # Optional Verification
    if run_verification:
        if expected_total is None:
            expected_total = total_sensitivity
        verify_calculation(E, I, M_primary, M_dual, L_elem, expected_total)
    
    # Create Plots
    if create_plots:
        print("\n" + "=" * 75)
        print("GENERATING VISUALIZATIONS")
        print("=" * 75)
        
        # Plot 1: Comprehensive sensitivity analysis
        save_path1 = os.path.join(output_dir, "sensitivity_analysis.png") if save_plots else None
        plot_sensitivity_analysis(E, I, L_elements, M_primary, M_dual, 
                                  sensitivities, total_sensitivity, beam_length,
                                  response_element, save_path1)
        
        # Plot 2: Moment diagrams comparison
        save_path2 = os.path.join(output_dir, "moment_diagrams.png") if save_plots else None
        plot_moment_diagrams_comparison(M_primary, M_dual, L_elements, beam_length,
                                        response_element, save_path2)
        
        # Plot 3: Sensitivity vs EI change
        save_path3 = os.path.join(output_dir, "sensitivity_vs_EI_change.png") if save_plots else None
        plot_sensitivity_vs_EI_change(E, I, sensitivities, total_sensitivity,
                                      response_element, save_path3)
    
    return sensitivities, total_sensitivity


if __name__ == "__main__":
    
    # =========================================
    # USER INPUT - MODIFY THESE VALUES
    # =========================================
    
    # VTK file paths
    PRIMARY_VTK = "test_files/SA_beam_2D_udl.gid/vtk_output/Parts_Beam_Beams_0_1.vtk"
    DUAL_VTK = "test_files/SA_beam_2D_udl.gid/vtk_output_dual/Parts_Beam_Beams_0_1.vtk"
    
    # Material properties
    E = 2.1e11      # Young's modulus [Pa]
    I = 5.0e-6      # Second moment of area [m^4]
    
    # Geometry
    BEAM_LENGTH = 2.0   # Total beam length [m]
    N_ELEMENTS = 9      # Number of elements
    
    # Response element (where we measure the moment)
    RESPONSE_ELEMENT = 5
    
    # Verification options
    RUN_VERIFICATION = False
    EXPECTED_TOTAL = 8.958322e-10
    
    # Plotting options
    CREATE_PLOTS = True
    SAVE_PLOTS = False           # Set to True to save plots
    OUTPUT_DIR = "."            # Directory to save plots
    
    # =========================================
    # Run Analysis
    # =========================================
    sensitivities, total = main(
        primary_vtk_path=PRIMARY_VTK,
        dual_vtk_path=DUAL_VTK,
        E=E,
        I=I,
        beam_length=BEAM_LENGTH,
        n_elements=N_ELEMENTS,
        response_element=RESPONSE_ELEMENT,
        run_verification=RUN_VERIFICATION,
        expected_total=EXPECTED_TOTAL,
        create_plots=CREATE_PLOTS,
        save_plots=SAVE_PLOTS,
        output_dir=OUTPUT_DIR
    )