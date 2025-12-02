"""
Quick runner for beam analysis with error verification
"""

from beam_plotter import (
    parse_vtk_file, calculate_errors, print_results_with_errors,
    plot_results_with_analytical, plot_error_distribution,
    BeamParameters
)
import matplotlib.pyplot as plt
import os

# =============== MODIFY THESE ===============
VTK_FILE = "test_files/beam_2D_test_udl.gid/vtk_output/Parts_Beam_Beams_0_1.vtk"

# Beam parameters
params = BeamParameters(
    L = 2.0,              # Beam length [m]
    w = 10000.0,          # UDL [N/m]
    P = 0.0,              # Point load [N]
    E = 210e9,            # Young's modulus [Pa]
    I = 5e-6,             # Moment of inertia [m^4]
    load_type = 'udl'     # 'udl' or 'point_center'
)

BEAM_TYPE = 'simply_supported'  # 'simply_supported' or 'cantilever'
# ============================================

# Find VTK file if not in current directory
if not os.path.exists(VTK_FILE):
    for root, dirs, files in os.walk('.'):
        for f in files:
            if f.endswith('.vtk'):
                VTK_FILE = os.path.join(root, f)
                print(f"Found: {VTK_FILE}")
                break

if os.path.exists(VTK_FILE):
    vtk_data = parse_vtk_file(VTK_FILE)
    errors = calculate_errors(vtk_data, params, BEAM_TYPE)
    print_results_with_errors(vtk_data, errors, params)
    plot_results_with_analytical(vtk_data, errors)
    plot_error_distribution(errors)
    plt.show()
else:
    print("No VTK file found!")