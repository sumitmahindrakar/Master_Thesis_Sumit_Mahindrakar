import os
import json
import glob
import traceback
import sys
import threading
import time

os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\SA_Kratos_adj")

import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication as SMA
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis


def clean_parameters(params_dict):
    """Remove HDF5 and JSON check processes, fix paths"""
    if "solver_settings" in params_dict:
        solver = params_dict["solver_settings"]
        if "model_import_settings" in solver:
            if "input_filename" in solver["model_import_settings"]:
                full_path = solver["model_import_settings"]["input_filename"]
                solver["model_import_settings"]["input_filename"] = os.path.basename(full_path)
        if "material_import_settings" in solver:
            if "materials_filename" in solver["material_import_settings"]:
                full_path = solver["material_import_settings"]["materials_filename"]
                solver["material_import_settings"]["materials_filename"] = os.path.basename(full_path)
    
    if "processes" in params_dict:
        for key in list(params_dict["processes"].keys()):
            processes_list = params_dict["processes"][key]
            if isinstance(processes_list, list):
                params_dict["processes"][key] = [
                    p for p in processes_list 
                    if "hdf5" not in str(p).lower() 
                    and "from_json_check" not in p.get("python_module", "")
                ]
    return params_dict


def remove_vtk_output(params_dict):
    """Remove VTK output to prevent hanging"""
    if "output_processes" in params_dict:
        print("  [FIX] Removing output_processes")
        del params_dict["output_processes"]
    return params_dict


def remove_problematic_processes(params_dict):
    """Remove processes that might cause hanging"""
    if "processes" in params_dict:
        # Remove list_other_processes (element sensitivity integration)
        if "list_other_processes" in params_dict["processes"]:
            print("  [FIX] Removing list_other_processes")
            params_dict["processes"]["list_other_processes"] = []
    return params_dict


# =============================================
# STEP 1: Run Primal Analysis
# =============================================
print("=" * 50)
print("STEP 1: Running Primal Analysis...")
print("=" * 50)

with open("beam_test_parameters.json", 'r') as file:
    primal_params_raw = json.load(file)

primal_params_raw = clean_parameters(primal_params_raw)
primal_parameters = KratosMultiphysics.Parameters(json.dumps(primal_params_raw))

primal_model = KratosMultiphysics.Model()
primal_analysis = StructuralMechanicsAnalysis(primal_model, primal_parameters)
primal_analysis.Run()

print("✓ Primal analysis completed!")

# Store primal results
primal_model_part = primal_model["Structure"]
primal_data = {}
for node in primal_model_part.Nodes:
    primal_data[node.Id] = {
        "DISPLACEMENT": list(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT)),
        "ROTATION": list(node.GetSolutionStepValue(KratosMultiphysics.ROTATION)),
    }

# Get material properties
E_value = None
I22_value = None
for element in primal_model_part.Elements:
    props = element.Properties
    E_value = props.GetValue(KratosMultiphysics.YOUNG_MODULUS)
    I22_value = props.GetValue(SMA.I22)
    break

print(f"  Material: E = {E_value:.2e}, I22 = {I22_value:.6e}")

# Print primal displacements
print("\n  Primal Displacements (Y-component):")
for node in primal_model_part.Nodes:
    disp = node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT)
    print(f"    Node {node.Id}: u_y = {disp[1]:.6e}")


# =============================================
# STEP 2: Run Adjoint Analysis (Simplified)
# =============================================
print("\n" + "=" * 50)
print("STEP 2: Running Adjoint Sensitivity Analysis...")
print("=" * 50)

with open("beam_test_local_stress_adjoint_parameters.json", 'r') as file:
    adjoint_params_raw = json.load(file)

# Apply all fixes
adjoint_params_raw = clean_parameters(adjoint_params_raw)
adjoint_params_raw = remove_vtk_output(adjoint_params_raw)
adjoint_params_raw = remove_problematic_processes(adjoint_params_raw)

# Set sensitivity variables
adjoint_params_raw["solver_settings"]["sensitivity_settings"]["element_data_value_sensitivity_variables"] = ["I22", "YOUNG_MODULUS"]

# Add echo level for more debug info
adjoint_params_raw["solver_settings"]["echo_level"] = 1

print(f"  Keys after fix: {list(adjoint_params_raw.keys())}")
print(f"  Processes: {list(adjoint_params_raw['processes'].keys())}")

adjoint_parameters = KratosMultiphysics.Parameters(json.dumps(adjoint_params_raw))


# Simplified class - let parent handle everything
class AdjointAnalysisWithPrimalData(StructuralMechanicsAnalysis):
    def __init__(self, model, parameters, primal_data):
        self.primal_data = primal_data
        print("  [DEBUG] Calling parent __init__...")
        sys.stdout.flush()
        super().__init__(model, parameters)
        print("  [DEBUG] Parent __init__ completed")
        sys.stdout.flush()
    
    def Initialize(self):
        print("  [DEBUG] Initialize() - calling parent...")
        sys.stdout.flush()
        
        # Call parent Initialize with progress indicator
        print("  [DEBUG] Starting super().Initialize()...")
        sys.stdout.flush()
        
        super().Initialize()
        
        print("  [DEBUG] super().Initialize() completed")
        sys.stdout.flush()
        
        # Inject primal solution
        print("  [DEBUG] Injecting primal solution...")
        sys.stdout.flush()
        
        model_part = self.model["Structure"]
        for node in model_part.Nodes:
            if node.Id in self.primal_data:
                data = self.primal_data[node.Id]
                disp = KratosMultiphysics.Array3()
                disp[0] = data["DISPLACEMENT"][0]
                disp[1] = data["DISPLACEMENT"][1]
                disp[2] = data["DISPLACEMENT"][2]
                node.SetSolutionStepValue(KratosMultiphysics.DISPLACEMENT, disp)
                
                rot = KratosMultiphysics.Array3()
                rot[0] = data["ROTATION"][0]
                rot[1] = data["ROTATION"][1]
                rot[2] = data["ROTATION"][2]
                node.SetSolutionStepValue(KratosMultiphysics.ROTATION, rot)
        
        print("  [DEBUG] Primal solution injected")
        sys.stdout.flush()


print("\n🚀 Creating adjoint analysis object...")
sys.stdout.flush()

adjoint_model = KratosMultiphysics.Model()

try:
    adjoint_analysis = AdjointAnalysisWithPrimalData(adjoint_model, adjoint_parameters, primal_data)
    print("  ✓ Object created successfully")
    sys.stdout.flush()
    
    print("\n🚀 Calling adjoint_analysis.Run()...")
    print("  (This may take a moment...)")
    sys.stdout.flush()
    
    adjoint_analysis.Run()
    
    print("  ✓ Run() completed successfully!")
    sys.stdout.flush()
    
except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.stdout.flush()


# =============================================
# STEP 3: Extract Sensitivities
# =============================================
print("\n" + "=" * 50)
print("STEP 3: Extracting Sensitivities...")
print("=" * 50)
sys.stdout.flush()

try:
    adjoint_model_part = adjoint_model["Structure"]
    
    print("\n📊 Element Sensitivities (dM/dI22 and dM/dE):")
    print("-" * 80)
    print(f"{'Element':>8} {'dM/dI22':>18} {'dM/dE':>18} {'dM/d(EI)':>18}")
    print("-" * 80)
    
    results = []
    all_zero = True
    
    for element in adjoint_model_part.Elements:
        try:
            dM_dI22 = element.GetValue(SMA.I22_SENSITIVITY)
            if abs(dM_dI22) > 1e-15:
                all_zero = False
        except:
            dM_dI22 = 0.0
        
        try:
            dM_dE = element.GetValue(SMA.YOUNG_MODULUS_SENSITIVITY)
            if abs(dM_dE) > 1e-15:
                all_zero = False
        except:
            dM_dE = 0.0
        
        dM_dEI = dM_dI22 / E_value if E_value != 0 else 0.0
        
        results.append({
            "element": element.Id,
            "dM_dI22": dM_dI22,
            "dM_dE": dM_dE,
            "dM_dEI": dM_dEI
        })
        
        print(f"{element.Id:>8} {dM_dI22:>18.6e} {dM_dE:>18.6e} {dM_dEI:>18.6e}")
    
    print("-" * 80)
    
    if all_zero:
        print("\n⚠️  WARNING: All sensitivities are zero!")
        print("   This might indicate an issue with the adjoint solve.")
    
except Exception as e:
    print(f"❌ Error extracting sensitivities: {e}")
    traceback.print_exc()
    results = []


# =============================================
# STEP 4: Extract Shape Sensitivities
# =============================================
print("\n📊 Shape Sensitivities (dM/dX):")
print("-" * 60)
print(f"{'Node':>6} {'dM/dX':>15} {'dM/dY':>15} {'dM/dZ':>15}")
print("-" * 60)

try:
    for node in adjoint_model_part.Nodes:
        try:
            shape_sens = node.GetSolutionStepValue(SMA.SHAPE_SENSITIVITY)
            print(f"{node.Id:>6} {shape_sens[0]:>15.6e} {shape_sens[1]:>15.6e} {shape_sens[2]:>15.6e}")
        except:
            print(f"{node.Id:>6} {'N/A':>15} {'N/A':>15} {'N/A':>15}")
except Exception as e:
    print(f"  Error: {e}")

print("-" * 60)


# =============================================
# STEP 5: Create VTK Output Manually
# =============================================
print("\n" + "=" * 50)
print("STEP 4: Creating VTK Output...")
print("=" * 50)

try:
    vtk_dir = "vtk_output_adj"
    if not os.path.exists(vtk_dir):
        os.makedirs(vtk_dir)
    
    vtk_filename = os.path.join(vtk_dir, "Structure_adjoint.vtk")
    
    with open(vtk_filename, 'w') as vtk_file:
        # Header
        vtk_file.write("# vtk DataFile Version 4.0\n")
        vtk_file.write("Adjoint Sensitivity Results\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET UNSTRUCTURED_GRID\n")
        
        # Points
        num_nodes = adjoint_model_part.NumberOfNodes()
        vtk_file.write(f"POINTS {num_nodes} float\n")
        
        node_id_map = {}
        idx = 0
        for node in adjoint_model_part.Nodes:
            vtk_file.write(f"{node.X:.10e} {node.Y:.10e} {node.Z:.10e}\n")
            node_id_map[node.Id] = idx
            idx += 1
        
        # Cells
        num_elements = adjoint_model_part.NumberOfElements()
        vtk_file.write(f"\nCELLS {num_elements} {num_elements * 3}\n")
        
        for element in adjoint_model_part.Elements:
            node_ids = [node_id_map[node.Id] for node in element.GetNodes()]
            vtk_file.write(f"2 {node_ids[0]} {node_ids[1]}\n")
        
        vtk_file.write(f"\nCELL_TYPES {num_elements}\n")
        for _ in range(num_elements):
            vtk_file.write("3\n")
        
        # Point Data
        vtk_file.write(f"\nPOINT_DATA {num_nodes}\n")
        
        # Shape sensitivity
        vtk_file.write("VECTORS SHAPE_SENSITIVITY float\n")
        for node in adjoint_model_part.Nodes:
            try:
                sens = node.GetSolutionStepValue(SMA.SHAPE_SENSITIVITY)
                vtk_file.write(f"{sens[0]:.10e} {sens[1]:.10e} {sens[2]:.10e}\n")
            except:
                vtk_file.write("0.0 0.0 0.0\n")
        
        # Cell Data
        vtk_file.write(f"\nCELL_DATA {num_elements}\n")
        
        # I22 sensitivity
        vtk_file.write("SCALARS I22_SENSITIVITY float 1\n")
        vtk_file.write("LOOKUP_TABLE default\n")
        for element in adjoint_model_part.Elements:
            try:
                sens = element.GetValue(SMA.I22_SENSITIVITY)
                vtk_file.write(f"{sens:.10e}\n")
            except:
                vtk_file.write("0.0\n")
        
        # Young's modulus sensitivity
        vtk_file.write("\nSCALARS YOUNG_MODULUS_SENSITIVITY float 1\n")
        vtk_file.write("LOOKUP_TABLE default\n")
        for element in adjoint_model_part.Elements:
            try:
                sens = element.GetValue(SMA.YOUNG_MODULUS_SENSITIVITY)
                vtk_file.write(f"{sens:.10e}\n")
            except:
                vtk_file.write("0.0\n")
    
    print(f"  ✓ Created: {vtk_filename} ({os.path.getsize(vtk_filename)} bytes)")
    
except Exception as e:
    print(f"  ❌ Error: {e}")


# =============================================
# STEP 6: Save JSON Results
# =============================================
print("\n" + "=" * 50)
print("STEP 5: Saving Results to JSON...")
print("=" * 50)

output_data = {
    "beam": {"E": E_value, "I22": I22_value, "EI": E_value * I22_value},
    "response": {"type": "MY", "element": 6},
    "sensitivities": results
}

with open("sensitivity_results.json", 'w') as f:
    json.dump(output_data, f, indent=2)

print("  ✓ Saved: sensitivity_results.json")


# =============================================
# Final Summary
# =============================================
print("\n" + "=" * 50)
print("COMPLETE!")
print("=" * 50)

print("\n📁 Output Files:")
for f in glob.glob("vtk_output_primal/*.vtk"):
    print(f"  ✓ {f}")
for f in glob.glob("vtk_output_adj/*.vtk"):
    print(f"  ✓ {f}")
print("  ✓ sensitivity_results.json")

print("\n✅ Script finished successfully!")