import KratosMultiphysics
from KratosMultiphysics import Logger
Logger.GetDefaultOutput().SetSeverity(Logger.Severity.INFO)

import KratosMultiphysics.StructuralMechanicsApplication as SMA
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis

import json
import os

# ============================================
# CONFIGURATION
# ============================================
WORKING_DIR = r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\SA_Kratos_adj"
PRIMAL_PARAMS = "beam_test_parameters.json"
ADJOINT_PARAMS = "beam_test_local_stress_adjoint_parameters.json"

os.chdir(WORKING_DIR)

# ============================================
# STEP 1: Run Primal Analysis
# ============================================
print("=" * 60)
print("STEP 1: Running Primal Analysis")
print("=" * 60)

with open(PRIMAL_PARAMS, 'r') as f:
    primal_parameters = KratosMultiphysics.Parameters(f.read())

primal_model = KratosMultiphysics.Model()
primal_analysis = StructuralMechanicsAnalysis(primal_model, primal_parameters)
primal_analysis.Run()

# Save primal results for adjoint analysis
primal_model_part = primal_model["Structure"]
print("\n--- Primal Results ---")
for node in primal_model_part.Nodes:
    disp = node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT)
    print(f"Node {node.Id}: DISPLACEMENT = [{disp[0]:.6e}, {disp[1]:.6e}, {disp[2]:.6e}]")

print("\n✅ Primal analysis completed!")

# ============================================
# STEP 2: Run Adjoint Analysis
# ============================================
print("\n" + "=" * 60)
print("STEP 2: Running Adjoint Analysis")
print("=" * 60)

with open(ADJOINT_PARAMS, 'r') as f:
    adjoint_parameters = KratosMultiphysics.Parameters(f.read())

adjoint_model = KratosMultiphysics.Model()
adjoint_analysis = StructuralMechanicsAnalysis(adjoint_model, adjoint_parameters)
adjoint_analysis.Run()

print("\n✅ Adjoint analysis completed!")

# ============================================
# STEP 3: Extract Sensitivities
# ============================================
print("\n" + "=" * 60)
print("STEP 3: Extracting Sensitivities")
print("=" * 60)

adjoint_model_part = adjoint_model["Structure"]

print("\n--- Adjoint Nodal Results ---")
for node in adjoint_model_part.Nodes:
    node_id = node.Id
    
    try:
        adj_disp = node.GetSolutionStepValue(SMA.ADJOINT_DISPLACEMENT)
        print(f"Node {node_id}: ADJOINT_DISPLACEMENT = [{adj_disp[0]:.6e}, {adj_disp[1]:.6e}, {adj_disp[2]:.6e}]")
    except:
        pass
    
    try:
        shape_sens = node.GetSolutionStepValue(KratosMultiphysics.SHAPE_SENSITIVITY)
        print(f"Node {node_id}: SHAPE_SENSITIVITY = [{shape_sens[0]:.6e}, {shape_sens[1]:.6e}, {shape_sens[2]:.6e}]")
    except:
        pass

print("\n--- Element Sensitivities ---")
for element in adjoint_model_part.Elements:
    elem_id = element.Id
    print(f"\nElement {elem_id}:")
    
    try:
        i22_sens = element.GetValue(SMA.I22_SENSITIVITY)
        print(f"  I22_SENSITIVITY = {i22_sens:.8e}")
    except:
        print("  I22_SENSITIVITY = N/A")
    
    try:
        ym_sens = element.GetValue(KratosMultiphysics.YOUNG_MODULUS_SENSITIVITY)
        print(f"  YOUNG_MODULUS_SENSITIVITY = {ym_sens:.8e}")
    except:
        print("  YOUNG_MODULUS_SENSITIVITY = N/A")

print("\n" + "=" * 60)
print("✅ ALL COMPLETED!")
print("=" * 60)