"""
Moment Sensitivity Calculator: ∂M/∂(EI)
Using General Influence Method with VTK File Parsing and Visualization

Description:
    Computes the sensitivity of bending moment at a response location
    with respect to flexural rigidity (EI) of beam elements using the
    adjoint/dual method.
    
Formula:
    ∂M_response/∂(EI)_k = -(1/(EI)²) × ∫ M_k(x) × M̄_k(x) dx
    
    Approximated as:
    ∂M_response/∂(EI)_k = -(M_k × M̄_k × L_k) / (EI)²
    
    Where:
    - M_k: Primary moment in element k (from actual loading)
    - M̄_k: Dual/adjoint moment in element k (from unit virtual load)
    - L_k: Length of element k
    - EI: Flexural rigidity (assumed uniform for all elements)
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches


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
        
        # Detect CELL_DATA section
        if line_stripped.startswith('CELL_DATA'):
            in_cell_data = True
            continue
        
        # Reset if we hit POINT_DATA (we've passed CELL_DATA)
        if line_stripped.startswith('POINT_DATA'):
            in_cell_data = False
            reading_moment = False
            continue
        
        # Look for MOMENT field in CELL_DATA
        if in_cell_data and 'MOMENT' in line_stripped and 'FIELD' not in line_stripped:
            reading_moment = True
            continue
        
        # Read moment values
        if reading_moment and line_stripped:
            parts = line_stripped.split()
            if len(parts) >= 3:
                try:
                    # Moment is [Mx, My, Mz] - we want Mz (3rd component)
                    Mz = float(parts[2])
                    moments[elem_idx] = Mz
                    elem_idx += 1
                except ValueError:
                    # Not a number line, stop reading
                    reading_moment = False
    
    return moments


def compute_moment_sensitivity(E, I, L_elements, M_primary, M_dual):
    """
    Compute ∂M/∂(EI) for all elements using the adjoint method
    
    Formula: ∂M_response/∂(EI)_k = -(M_k · M̄_k · L_k) / (EI)²
    
    This represents how the response moment changes when the stiffness
    of element k changes, assuming all elements currently have the same EI.
    
    Args:
        E: Young's modulus [Pa]
        I: Second moment of area [m^4]
        L_elements: Dict of element lengths {elem_id: length}
        M_primary: Dict of primary moments {elem_id: moment}
        M_dual: Dict of dual moments {elem_id: moment}
        
    Returns:
        tuple: (sensitivities dict, total sensitivity)
    """
    EI = E * I
    EI_squared = EI ** 2
    
    sensitivities = {}
    total_sensitivity = 0.0
    
    for eid in sorted(M_primary.keys()):
        M_p = M_primary[eid]
        M_d = M_dual.get(eid, 0.0)
        L = L_elements.get(eid, 1.0)
        
        # Virtual work integral (approximated)
        integral = M_p * M_d * L
        
        # Sensitivity formula: ∂M/∂(EI)_k = -∫(M·M̄)dx / (EI)²
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


def print_results(E, I, sensitivities, total_sensitivity, response_element=None):
    """
    Print formatted sensitivity results
    """
    EI = E * I
    
    print("\n" + "=" * 75)
    print("MOMENT SENSITIVITY ANALYSIS: ∂M/∂(EI)")
    print("Using General Influence (Adjoint) Method")
    print("=" * 75)
    
    print(f"\nMaterial/Section Properties:")
    print(f"  E  = {E:.4e} Pa")
    print(f"  I  = {I:.4e} m⁴")
    print(f"  EI = {EI:.4e} N·m²")
    
    if response_element:
        print(f"\nResponse: Bending moment in Element {response_element}")
    
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
    
    # Example: effect of 10% EI change
    delta_EI_percent = 10.0
    delta_EI = (delta_EI_percent / 100.0) * EI
    delta_M = total_sensitivity * delta_EI
    
    print(f"\nLinear Approximation:")
    print(f"  If EI increases by {delta_EI_percent}%: ΔM ≈ {delta_M:+.6f} N·m")
    
    return


def plot_beam_schematic(ax, beam_length, n_elements, L_elements, response_element=None):
    """
    Plot beam structure schematic
    """
    beam_y = 0.5
    beam_height = 0.08
    
    # Draw beam body
    beam_rect = Rectangle((0, beam_y - beam_height/2), beam_length, beam_height,
                          facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(beam_rect)
    
    # Element boundaries and labels
    elem_ids = sorted(L_elements.keys())
    x_nodes = [0]
    for eid in elem_ids:
        x_nodes.append(x_nodes[-1] + L_elements[eid])
    
    # Draw element divisions
    for i, x in enumerate(x_nodes):
        ax.axvline(x=x, ymin=0.35, ymax=0.65, color='black', linewidth=1.5, linestyle='-')
        ax.plot(x, beam_y, 'ko', markersize=8, zorder=5)
        ax.text(x, beam_y - 0.15, f'N{i+1}', ha='center', fontsize=9)
    
    # Element labels
    for i, eid in enumerate(elem_ids):
        x_mid = (x_nodes[i] + x_nodes[i+1]) / 2
        color = 'red' if eid == response_element else 'black'
        weight = 'bold' if eid == response_element else 'normal'
        ax.text(x_mid, beam_y + 0.12, f'E{eid}', ha='center', fontsize=10, 
               color=color, fontweight=weight)
        
        # Highlight response element
        if eid == response_element:
            highlight = Rectangle((x_nodes[i], beam_y - beam_height/2 - 0.02), 
                                  L_elements[eid], beam_height + 0.04,
                                  facecolor='none', edgecolor='red', 
                                  linewidth=2, linestyle='--')
            ax.add_patch(highlight)
    
    # Fixed supports
    support_width = 0.05
    support_height = 0.1
    
    # Left support
    left_support = Rectangle((-support_width, beam_y - beam_height/2 - support_height),
                            support_width, support_height + beam_height/2,
                            facecolor='gray', edgecolor='black', linewidth=1, hatch='///')
    ax.add_patch(left_support)
    
    # Right support  
    right_support = Rectangle((beam_length, beam_y - beam_height/2 - support_height),
                             support_width, support_height + beam_height/2,
                             facecolor='gray', edgecolor='black', linewidth=1, hatch='///')
    ax.add_patch(right_support)
    
    # UDL arrows
    n_arrows = min(15, n_elements * 3)
    arrow_spacing = beam_length / (n_arrows + 1)
    for i in range(1, n_arrows + 1):
        x = i * arrow_spacing
        ax.annotate('', xy=(x, beam_y + beam_height/2 + 0.02),
                   xytext=(x, beam_y + beam_height/2 + 0.15),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=1.2))
    
    # UDL line
    ax.plot([arrow_spacing, beam_length - arrow_spacing], 
           [beam_y + beam_height/2 + 0.15, beam_y + beam_height/2 + 0.15],
           'b-', linewidth=2)
    ax.text(beam_length/2, beam_y + beam_height/2 + 0.22, 'UDL: q', 
           ha='center', fontsize=10, color='blue')
    
    ax.set_xlim(-0.2, beam_length + 0.2)
    ax.set_ylim(0.1, 0.9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Beam Structure Schematic', fontsize=12, fontweight='bold')


def plot_moment_diagram(ax, L_elements, moments, title, color='blue', ylabel='Moment [N·m]'):
    """
    Plot bending moment diagram as a step function with filled areas
    """
    elem_ids = sorted(moments.keys())
    
    # Build x coordinates
    x_nodes = [0]
    for eid in elem_ids:
        x_nodes.append(x_nodes[-1] + L_elements[eid])
    
    # Plot each element's moment as a filled rectangle
    for i, eid in enumerate(elem_ids):
        x_start = x_nodes[i]
        x_end = x_nodes[i+1]
        M = moments[eid]
        
        # Fill color based on sign
        fill_color = 'lightgreen' if M >= 0 else 'lightcoral'
        edge_color = 'green' if M >= 0 else 'red'
        
        # Create filled region
        ax.fill_between([x_start, x_end], [0, 0], [M, M], 
                       color=fill_color, alpha=0.6, edgecolor=edge_color, linewidth=1.5)
        
        # Add value label
        x_mid = (x_start + x_end) / 2
        y_offset = M * 0.1 if abs(M) > 0 else 0.1
        va = 'bottom' if M >= 0 else 'top'
        ax.text(x_mid, M + np.sign(M) * abs(max(moments.values())) * 0.05, 
               f'{M:+.2f}', ha='center', va=va, fontsize=9, fontweight='bold')
    
    # Draw baseline
    ax.axhline(y=0, color='black', linewidth=1.5)
    
    # Element boundaries
    for x in x_nodes:
        ax.axvline(x=x, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Position along beam [m]', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis limits with small padding
    ax.set_xlim(-0.05 * x_nodes[-1], x_nodes[-1] * 1.05)


def plot_sensitivity_bar(ax, sensitivities, total_sensitivity, ylabel='∂M/∂(EI)'):
    """
    Plot sensitivity values as a bar chart with total
    """
    elem_ids = sorted(sensitivities.keys())
    sens_values = [sensitivities[eid]['dM_dEI'] for eid in elem_ids]
    
    # Positions for bars
    x_elem = list(range(len(elem_ids)))
    x_total = len(elem_ids) + 0.5
    
    # Colors
    colors_elem = ['#FF6B6B' if v < 0 else '#4ECDC4' for v in sens_values]
    color_total = '#2C3E50'
    
    # Plot element sensitivities
    bars_elem = ax.bar(x_elem, sens_values, color=colors_elem, 
                       edgecolor='black', linewidth=1.2, alpha=0.8, label='Element')
    
    # Plot total
    bar_total = ax.bar(x_total, total_sensitivity, color=color_total,
                       edgecolor='black', linewidth=1.5, alpha=0.9, label='Total')
    
    # Add value labels
    for i, (x, v) in enumerate(zip(x_elem, sens_values)):
        va = 'bottom' if v >= 0 else 'top'
        offset = max(abs(v) for v in sens_values + [total_sensitivity]) * 0.05
        y_pos = v + (offset if v >= 0 else -offset)
        ax.text(x, y_pos, f'{v:.2e}', ha='center', va=va, fontsize=8, rotation=45)
    
    # Total label
    va = 'bottom' if total_sensitivity >= 0 else 'top'
    offset = max(abs(v) for v in sens_values + [total_sensitivity]) * 0.05
    y_pos = total_sensitivity + (offset if total_sensitivity >= 0 else -offset)
    ax.text(x_total, y_pos, f'{total_sensitivity:.2e}', 
           ha='center', va=va, fontsize=9, fontweight='bold', rotation=45)
    
    # Baseline
    ax.axhline(y=0, color='black', linewidth=1)
    
    # X-axis labels
    x_labels = [f'Elem {eid}' for eid in elem_ids] + ['TOTAL']
    ax.set_xticks(x_elem + [x_total])
    ax.set_xticklabels(x_labels, fontsize=9)
    
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title('Sensitivity: ∂M/∂(EI) by Element', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Legend
    positive_patch = mpatches.Patch(color='#4ECDC4', label='Positive sensitivity')
    negative_patch = mpatches.Patch(color='#FF6B6B', label='Negative sensitivity')
    total_patch = mpatches.Patch(color='#2C3E50', label='Total (sum)')
    ax.legend(handles=[positive_patch, negative_patch, total_patch], 
             loc='best', fontsize=8)


def create_sensitivity_plots(E, I, L_elements, M_primary, M_dual, sensitivities, 
                             total_sensitivity, beam_length, response_element=None,
                             save_path=None):
    """
    Create comprehensive visualization with all plots
    """
    n_elements = len(M_primary)
    EI = E * I
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 12))
    
    # Layout: 2x2 grid
    ax1 = fig.add_subplot(2, 2, 1)  # Beam schematic
    ax2 = fig.add_subplot(2, 2, 2)  # Sensitivity bar chart
    ax3 = fig.add_subplot(2, 2, 3)  # Primary moment diagram
    ax4 = fig.add_subplot(2, 2, 4)  # Dual moment diagram
    
    # Plot 1: Beam Schematic
    plot_beam_schematic(ax1, beam_length, n_elements, L_elements, response_element)
    
    # Plot 2: Sensitivity Bar Chart
    plot_sensitivity_bar(ax2, sensitivities, total_sensitivity)
    
    # Plot 3: Primary Moment Diagram
    plot_moment_diagram(ax3, L_elements, M_primary, 
                       'Primary Bending Moment M(x)\n[From Applied Load]',
                       color='blue', ylabel='Moment M [N·m]')
    
    # Plot 4: Dual Moment Diagram
    plot_moment_diagram(ax4, L_elements, M_dual,
                       'Dual/Adjoint Moment M̄(x)\n[From Unit Virtual Load]',
                       color='red', ylabel='Dual Moment M̄ [N·m]')
    
    # Add overall title
    title_text = 'Moment Sensitivity Analysis: ∂M/∂(EI)\nUsing Adjoint Method'
    if response_element:
        title_text += f' | Response: Element {response_element}'
    fig.suptitle(title_text, fontsize=14, fontweight='bold', y=0.98)
    
    # Add info text box
    info_text = (f'E = {E:.2e} Pa\n'
                f'I = {I:.2e} m⁴\n'
                f'EI = {EI:.2e} N·m²\n'
                f'Total ∂M/∂(EI) = {total_sensitivity:.4e}')
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    fig.text(0.02, 0.02, info_text, fontsize=9, family='monospace',
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()
    
    return fig


def create_detailed_sensitivity_plot(sensitivities, total_sensitivity, E, I,
                                     save_path=None):
    """
    Create a detailed sensitivity plot showing the breakdown
    """
    EI = E * I
    elem_ids = sorted(sensitivities.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Data extraction
    M_values = [sensitivities[eid]['M_primary'] for eid in elem_ids]
    M_bar_values = [sensitivities[eid]['M_dual'] for eid in elem_ids]
    dM_dEI_values = [sensitivities[eid]['dM_dEI'] for eid in elem_ids]
    
    x = np.arange(len(elem_ids))
    width = 0.6
    
    # Plot 1: Primary Moments
    ax1 = axes[0]
    colors1 = ['green' if v >= 0 else 'red' for v in M_values]
    bars1 = ax1.bar(x, M_values, width, color=colors1, edgecolor='black', alpha=0.7)
    ax1.axhline(y=0, color='black', linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'E{eid}' for eid in elem_ids])
    ax1.set_ylabel('Primary Moment M [N·m]')
    ax1.set_title('Primary Moments M_k', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, M_values):
        ax1.text(bar.get_x() + bar.get_width()/2, val, f'{val:+.1f}', 
                ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)
    
    # Plot 2: Dual Moments
    ax2 = axes[1]
    colors2 = ['green' if v >= 0 else 'red' for v in M_bar_values]
    bars2 = ax2.bar(x, M_bar_values, width, color=colors2, edgecolor='black', alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'E{eid}' for eid in elem_ids])
    ax2.set_ylabel('Dual Moment M̄ [N·m]')
    ax2.set_title('Dual Moments M̄_k', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, M_bar_values):
        ax2.text(bar.get_x() + bar.get_width()/2, val, f'{val:+.4f}', 
                ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)
    
    # Plot 3: Sensitivities with Total
    ax3 = axes[2]
    x_sens = list(range(len(elem_ids))) + [len(elem_ids) + 0.5]
    all_sens = dM_dEI_values + [total_sensitivity]
    colors3 = ['#4ECDC4' if v >= 0 else '#FF6B6B' for v in dM_dEI_values] + ['#2C3E50']
    bars3 = ax3.bar(x_sens, all_sens, width, color=colors3, edgecolor='black', alpha=0.8)
    ax3.axhline(y=0, color='black', linewidth=1)
    ax3.set_xticks(x_sens)
    ax3.set_xticklabels([f'E{eid}' for eid in elem_ids] + ['TOTAL'])
    ax3.set_ylabel('∂M/∂(EI)')
    ax3.set_title('Sensitivity ∂M/∂(EI)_k', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars3, all_sens):
        ax3.text(bar.get_x() + bar.get_width()/2, val, f'{val:.2e}', 
                ha='center', va='bottom' if val >= 0 else 'top', fontsize=8, rotation=0)
    
    fig.suptitle(f'Sensitivity Analysis Breakdown | EI = {EI:.2e} N·m²', 
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nDetailed plot saved to: {save_path}")
    
    plt.show()
    
    return fig


def main(primary_vtk_path, dual_vtk_path, E, I, beam_length, n_elements,
         response_element=None, create_plots=True, save_plots=False, output_dir="."):
    """
    Main function - compute sensitivity from VTK files with visualization
    
    Args:
        primary_vtk_path: Path to primary analysis VTK file
        dual_vtk_path: Path to dual analysis VTK file
        E: Young's modulus [Pa]
        I: Second moment of area [m^4]
        beam_length: Total beam length [m]
        n_elements: Number of elements (will be updated from VTK if different)
        response_element: Element ID for response (optional, for labeling)
        create_plots: Whether to create visualization plots (default: True)
        save_plots: Whether to save plots to files (default: False)
        output_dir: Directory to save plots (default: current directory)
        
    Returns:
        tuple: (sensitivities dict, total sensitivity)
    """
    
    print("=" * 75)
    print("MOMENT SENSITIVITY ANALYSIS: ∂M/∂(EI)")
    print("Using Adjoint Method with VTK File Parsing")
    print("=" * 75)
    
    # =========================================
    # Parse Primary VTK File
    # =========================================
    print(f"\nPrimary VTK: {primary_vtk_path}")
    M_primary = parse_vtk_cell_moments(primary_vtk_path)
    
    if M_primary:
        print("Primary Moments (CELL_DATA MOMENT Mz):")
        for eid, M in sorted(M_primary.items()):
            print(f"  Element {eid}: M = {M:+.4f} N·m")
    else:
        print("Error: Could not parse primary moments from VTK file.")
        return None, None
    
    # =========================================
    # Parse Dual VTK File
    # =========================================
    print(f"\nDual VTK: {dual_vtk_path}")
    M_dual = parse_vtk_cell_moments(dual_vtk_path)
    
    if M_dual:
        print("Dual Moments (CELL_DATA MOMENT Mz):")
        for eid, M in sorted(M_dual.items()):
            print(f"  Element {eid}: M̄ = {M:+.6f} N·m")
    else:
        print("Error: Could not parse dual moments from VTK file.")
        return None, None
    
    # =========================================
    # Update element count and lengths from VTK data
    # =========================================
    actual_n_elements = len(M_primary)
    if actual_n_elements != n_elements:
        print(f"\nNote: Found {actual_n_elements} elements in VTK (expected {n_elements})")
        n_elements = actual_n_elements
    
    L_elem = beam_length / n_elements
    L_elements = {i: L_elem for i in range(1, n_elements + 1)}
    
    print(f"\nElement lengths: L = {L_elem:.6f} m each")
    
    # Validate response element
    elem_ids = sorted(M_primary.keys())
    if response_element is not None and response_element not in elem_ids:
        print(f"Warning: Response element {response_element} not in data. Available: {elem_ids}")
        response_element = elem_ids[len(elem_ids)//2]
        print(f"Using element {response_element} as response element.")
    
    # =========================================
    # Compute Sensitivities
    # =========================================
    sensitivities, total_sensitivity = compute_moment_sensitivity(
        E, I, L_elements, M_primary, M_dual
    )
    
    # =========================================
    # Print Results
    # =========================================
    print_results(E, I, sensitivities, total_sensitivity, response_element)
    
    # =========================================
    # Create Visualizations
    # =========================================
    if create_plots:
        print("\n" + "=" * 75)
        print("GENERATING VISUALIZATIONS")
        print("=" * 75)
        
        # Main comprehensive plot
        save_path1 = os.path.join(output_dir, "sensitivity_analysis.png") if save_plots else None
        create_sensitivity_plots(E, I, L_elements, M_primary, M_dual,
                                sensitivities, total_sensitivity, beam_length,
                                response_element, save_path1)
        
        # Detailed breakdown plot
        save_path2 = os.path.join(output_dir, "sensitivity_breakdown.png") if save_plots else None
        create_detailed_sensitivity_plot(sensitivities, total_sensitivity, E, I, save_path2)
    
    return sensitivities, total_sensitivity


# =========================================
# Run Analysis
# =========================================
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
    N_ELEMENTS = 4      # Number of elements (will auto-detect from VTK)
    
    # Response element (optional - for labeling in plots)
    RESPONSE_ELEMENT = None  # Set to None if not applicable
    
    # Plotting options
    CREATE_PLOTS = True
    SAVE_PLOTS = True          # Set to True to save plots
    OUTPUT_DIR = "test_files/SA_beam_2D_udl.gid/plots"            # Directory to save plots
    
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
        create_plots=CREATE_PLOTS,
        save_plots=SAVE_PLOTS,
        output_dir=OUTPUT_DIR
    )