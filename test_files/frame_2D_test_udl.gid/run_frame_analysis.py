"""
Quick runner for frame analysis
"""

from frame_plotter import (
    parse_vtk_file, analyze_frame_geometry, calculate_frame_errors,
    print_frame_results, plot_frame_system_diagram, plot_analytical_diagrams,
    plot_fem_vs_analytical, plot_error_distribution, FrameParameters,
    plot_column_diagrams, plot_column_deflections, plot_fem_column_comparison,
    identify_column_groups
)
import matplotlib.pyplot as plt
import os

# =============== MODIFY THESE ===============
VTK_FILE = "test_files/frame_2D_test_udl.gid/vtk_output/Parts_Beam_Beams_0_1.vtk"

params = FrameParameters(
    L = 2.0,
    H = 2.0,
    w = 10000.0,
    P = 0.0,
    E = 210e9,
    I_beam = 5e-6,
    I_col = 5e-6,
    A_beam = 0.00287,
    A_col = 0.00287,
    frame_type = 'portal_pinned',
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

    # Multi-bay column recognition
    column_groups = identify_column_groups(members, frame_info)
    
    # Update dimensions
    params.L = frame_info['span']
    params.H = frame_info['height']
    
    errors = calculate_frame_errors(vtk_data, members, params, frame_info)
    print_frame_results(vtk_data, members, errors, params)
    
    plot_frame_system_diagram(params, OUTPUT_PREFIX)
    plot_analytical_diagrams(params, errors, OUTPUT_PREFIX)
    plot_fem_vs_analytical(vtk_data, members, errors, params, OUTPUT_PREFIX)
    plot_error_distribution(vtk_data, errors, params, OUTPUT_PREFIX)

    # NEW column plots
    plot_column_diagrams(vtk_data, members, column_groups, params, OUTPUT_PREFIX)
    plot_column_deflections(vtk_data, members, column_groups, params, OUTPUT_PREFIX)
    plot_fem_column_comparison(vtk_data, members, column_groups, params, OUTPUT_PREFIX)

    plt.show()
else:
    print("No VTK file found!")
