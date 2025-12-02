"""
Quick runner for frame analysis
"""

from frame_plotter import (
    parse_vtk_file, analyze_frame_geometry, calculate_frame_errors,
    print_frame_results, plot_frame_system_diagram, plot_analytical_diagrams,
    plot_fem_vs_analytical, plot_error_distribution, FrameParameters
)
import matplotlib.pyplot as plt
import os

# =============== MODIFY THESE ===============
VTK_FILE = "test_files/frame_2D_test_udl.gid/vtk_output/Parts_Beam_Beams_0_1.vtk"  # Your VTK file

# Frame parameters
params = FrameParameters(
    L = 2.0,              # Beam span [m]
    H = 2.0,              # Column height [m]  
    w = 10000.0,          # UDL on beam [N/m]
    P = 0.0,              # Point load [N]
    E = 210e9,            # Young's modulus [Pa]
    I_beam = 5e-6,        # Beam I [m^4]
    I_col = 5e-6,         # Column I [m^4]
    A_beam = 0.00287,     # Beam area [m^2]
    A_col = 0.00287,      # Column area [m^2]
    frame_type = 'portal_pinned',  # 'portal_pinned' or 'portal_fixed'
    load_type = 'udl_beam'
)

OUTPUT_PREFIX = "test_files/frame_2D_test_udl.gid/plots/frame_results"
# ============================================

# Find VTK file
if not os.path.exists(VTK_FILE):
    print(f"File not found: {VTK_FILE}")
    for root, dirs, files in os.walk('.'):
        for f in files:
            if f.endswith('.vtk'):
                VTK_FILE = os.path.join(root, f)
                print(f"Found: {VTK_FILE}")
                break

if os.path.exists(VTK_FILE):
    print(f"Processing: {VTK_FILE}")
    
    vtk_data = parse_vtk_file(VTK_FILE)
    members, frame_info = analyze_frame_geometry(vtk_data)
    
    # Update dimensions
    params.L = frame_info['span']
    params.H = frame_info['height']
    
    errors = calculate_frame_errors(vtk_data, members, params, frame_info)
    print_frame_results(vtk_data, members, errors, params)
    
    plot_frame_system_diagram(params, OUTPUT_PREFIX)
    plot_analytical_diagrams(params, errors, OUTPUT_PREFIX)
    plot_fem_vs_analytical(vtk_data, members, errors, params, OUTPUT_PREFIX)
    plot_error_distribution(vtk_data, errors, params, OUTPUT_PREFIX)
    
    plt.show()
else:
    print("No VTK file found!")