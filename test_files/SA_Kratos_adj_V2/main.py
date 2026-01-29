"""
Main script for running Kratos beam structure analysis
with adjoint sensitivity computation.

This version works WITHOUT HDF5Application by manually
transferring primal solution to adjoint analysis.

Sensitivities computed:
- Shape sensitivity (dJ/dX, dJ/dY, dJ/dZ)
- I22 sensitivity (dJ/dI22)
- Young's modulus sensitivity (dJ/dE)
- Point load sensitivity (dJ/dP)

Outputs:
- Console output with all results
- VTK files for visualization in ParaView
"""

import os
import sys

# Set working directory to the project folder
os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\SA_Kratos_adj_V2")
print(f"Working directory: {os.getcwd()}")

import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication as StructuralMechanicsApplication

from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis


class PrimalAnalysis(StructuralMechanicsAnalysis):
    """Primal analysis that stores the solution."""
    
    def __init__(self, model, parameters):
        super().__init__(model, parameters)
        self.primal_solution = {}
    
    def Finalize(self):
        # Store the solution before finalizing
        model_part = self.model["Structure"]
        for node in model_part.Nodes:
            self.primal_solution[node.Id] = {
                "DISPLACEMENT": list(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT)),
                "ROTATION": list(node.GetSolutionStepValue(KratosMultiphysics.ROTATION))
            }
        super().Finalize()


class AdjointAnalysis(StructuralMechanicsAnalysis):
    """Adjoint analysis that loads primal solution."""
    
    def __init__(self, model, parameters, primal_solution):
        super().__init__(model, parameters)
        self.primal_solution = primal_solution
    
    def Initialize(self):
        super().Initialize()
        # Load the primal solution
        model_part = self.model["Structure"]
        for node in model_part.Nodes:
            if node.Id in self.primal_solution:
                sol = self.primal_solution[node.Id]
                disp = sol["DISPLACEMENT"]
                rot = sol["ROTATION"]
                node.SetSolutionStepValue(KratosMultiphysics.DISPLACEMENT, 
                    KratosMultiphysics.Array3([disp[0], disp[1], disp[2]]))
                node.SetSolutionStepValue(KratosMultiphysics.ROTATION, 
                    KratosMultiphysics.Array3([rot[0], rot[1], rot[2]]))


def run_primal_analysis(parameter_file_name):
    """Run primal analysis and return the model with solution."""
    with open(parameter_file_name, 'r') as f:
        parameters = KratosMultiphysics.Parameters(f.read())
    
    model = KratosMultiphysics.Model()
    analysis = PrimalAnalysis(model, parameters)
    analysis.Run()
    
    return model, analysis.primal_solution


def run_adjoint_analysis(parameter_file_name, primal_solution):
    """Run adjoint analysis using primal solution."""
    with open(parameter_file_name, 'r') as f:
        parameters = KratosMultiphysics.Parameters(f.read())
    
    model = KratosMultiphysics.Model()
    analysis = AdjointAnalysis(model, parameters, primal_solution)
    analysis.Run()
    
    return model


def print_primal_results(model_part):
    """Print primal analysis results."""
    print("\n" + "="*70)
    print("PRIMAL ANALYSIS RESULTS")
    print("="*70)
    
    print("\nNodal Displacements:")
    print("-"*70)
    print(f"{'Node':>6} {'DISP_X':>14} {'DISP_Y':>14} {'DISP_Z':>14}")
    print("-"*70)
    
    for node in model_part.Nodes:
        disp = node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT)
        print(f"{node.Id:>6} {disp[0]:>14.6e} {disp[1]:>14.6e} {disp[2]:>14.6e}")
    
    print("\nNodal Rotations:")
    print("-"*70)
    print(f"{'Node':>6} {'ROT_X':>14} {'ROT_Y':>14} {'ROT_Z':>14}")
    print("-"*70)
    
    for node in model_part.Nodes:
        rot = node.GetSolutionStepValue(KratosMultiphysics.ROTATION)
        print(f"{node.Id:>6} {rot[0]:>14.6e} {rot[1]:>14.6e} {rot[2]:>14.6e}")
    
    print("\nReaction Forces (at supports):")
    print("-"*70)
    try:
        for node in model_part.GetSubModelPart("DISPLACEMENT_support").Nodes:
            reaction = node.GetSolutionStepValue(KratosMultiphysics.REACTION)
            print(f"Node {node.Id}: [{reaction[0]:>12.4e}, {reaction[1]:>12.4e}, {reaction[2]:>12.4e}]")
    except:
        print("Reactions not available")
    
    print("\n>>> VTK output saved to: vtk_output_primal/")


def print_adjoint_results(model_part):
    """Print adjoint analysis results."""
    print("\n" + "="*70)
    print("ADJOINT SENSITIVITY RESULTS")
    print("="*70)
    
    # Adjoint displacements
    print("\nAdjoint Displacements:")
    print("-"*70)
    print(f"{'Node':>6} {'ADJ_DISP_X':>14} {'ADJ_DISP_Y':>14} {'ADJ_DISP_Z':>14}")
    print("-"*70)
    
    for node in model_part.Nodes:
        try:
            adj_disp = node.GetSolutionStepValue(StructuralMechanicsApplication.ADJOINT_DISPLACEMENT)
            print(f"{node.Id:>6} {adj_disp[0]:>14.6e} {adj_disp[1]:>14.6e} {adj_disp[2]:>14.6e}")
        except:
            pass
    
    # Adjoint rotations
    print("\nAdjoint Rotations:")
    print("-"*70)
    print(f"{'Node':>6} {'ADJ_ROT_X':>14} {'ADJ_ROT_Y':>14} {'ADJ_ROT_Z':>14}")
    print("-"*70)
    
    for node in model_part.Nodes:
        try:
            adj_rot = node.GetSolutionStepValue(StructuralMechanicsApplication.ADJOINT_ROTATION)
            print(f"{node.Id:>6} {adj_rot[0]:>14.6e} {adj_rot[1]:>14.6e} {adj_rot[2]:>14.6e}")
        except:
            pass
    
    # Shape sensitivities
    print("\n" + "="*70)
    print("SHAPE SENSITIVITIES (dJ/dX)")
    print("="*70)
    print("-"*70)
    print(f"{'Node':>6} {'dJ/dX':>14} {'dJ/dY':>14} {'dJ/dZ':>14}")
    print("-"*70)
    
    try:
        sensitivity_mp = model_part.GetSubModelPart("sensitivity_mp")
        for node in sensitivity_mp.Nodes:
            try:
                sens = node.GetSolutionStepValue(KratosMultiphysics.SHAPE_SENSITIVITY)
                print(f"{node.Id:>6} {sens[0]:>14.6e} {sens[1]:>14.6e} {sens[2]:>14.6e}")
            except:
                pass
    except:
        print("Shape sensitivities not available")
    
    # Element sensitivities - I22
    print("\n" + "="*70)
    print("ELEMENT I22 SENSITIVITIES (dJ/dI22)")
    print("="*70)
    print("-"*50)
    print(f"{'Element':>10} {'dJ/dI22':>20}")
    print("-"*50)
    
    try:
        sensitivity_mp = model_part.GetSubModelPart("sensitivity_mp")
        total_i22_sens = 0.0
        for element in sensitivity_mp.Elements:
            try:
                i22_sens = element.GetValue(StructuralMechanicsApplication.I22_SENSITIVITY)
                total_i22_sens += i22_sens
                print(f"{element.Id:>10} {i22_sens:>20.6e}")
            except:
                pass
        print("-"*50)
        print(f"{'TOTAL':>10} {total_i22_sens:>20.6e}")
    except:
        print("I22 sensitivities not available")
    
    # Element sensitivities - Young's Modulus
    print("\n" + "="*70)
    print("ELEMENT YOUNG'S MODULUS SENSITIVITIES (dJ/dE)")
    print("="*70)
    print("-"*50)
    print(f"{'Element':>10} {'dJ/dE':>20}")
    print("-"*50)
    
    try:
        sensitivity_mp = model_part.GetSubModelPart("sensitivity_mp")
        total_E_sens = 0.0
        for element in sensitivity_mp.Elements:
            try:
                E_sens = element.GetValue(StructuralMechanicsApplication.YOUNG_MODULUS_SENSITIVITY)
                total_E_sens += E_sens
                print(f"{element.Id:>10} {E_sens:>20.6e}")
            except:
                pass
        print("-"*50)
        print(f"{'TOTAL':>10} {total_E_sens:>20.6e}")
    except:
        print("Young's modulus sensitivities not available")
    
    # Condition sensitivities - Point Load
    print("\n" + "="*70)
    print("CONDITION POINT LOAD SENSITIVITIES (dJ/dP)")
    print("="*70)
    print("-"*70)
    print(f"{'Cond':>6} {'dJ/dPx':>14} {'dJ/dPy':>14} {'dJ/dPz':>14}")
    print("-"*70)
    
    found_load_sens = False
    try:
        load_mp = model_part.GetSubModelPart("PointLoad3D_load")
        for condition in load_mp.Conditions:
            try:
                P_sens = condition.GetValue(StructuralMechanicsApplication.POINT_LOAD_SENSITIVITY)
                print(f"{condition.Id:>6} {P_sens[0]:>14.6e} {P_sens[1]:>14.6e} {P_sens[2]:>14.6e}")
                found_load_sens = True
            except:
                pass
    except:
        pass
    
    # Also check in sensitivity_mp
    if not found_load_sens:
        try:
            sensitivity_mp = model_part.GetSubModelPart("sensitivity_mp")
            for condition in sensitivity_mp.Conditions:
                try:
                    P_sens = condition.GetValue(StructuralMechanicsApplication.POINT_LOAD_SENSITIVITY)
                    print(f"{condition.Id:>6} {P_sens[0]:>14.6e} {P_sens[1]:>14.6e} {P_sens[2]:>14.6e}")
                    found_load_sens = True
                except:
                    pass
        except:
            pass
    
    if not found_load_sens:
        print("Point load sensitivities not available")
    
    print("\n>>> VTK output saved to: vtk_output_adjoint/")


def print_summary(model_part):
    """Print a summary of all sensitivities."""
    print("\n" + "="*70)
    print("SENSITIVITY SUMMARY")
    print("="*70)
    
    try:
        sensitivity_mp = model_part.GetSubModelPart("sensitivity_mp")
        
        # Calculate totals
        total_shape_sens = [0.0, 0.0, 0.0]
        for node in sensitivity_mp.Nodes:
            try:
                sens = node.GetSolutionStepValue(KratosMultiphysics.SHAPE_SENSITIVITY)
                total_shape_sens[0] += sens[0]
                total_shape_sens[1] += sens[1]
                total_shape_sens[2] += sens[2]
            except:
                pass
        
        total_i22_sens = 0.0
        total_E_sens = 0.0
        for element in sensitivity_mp.Elements:
            try:
                i22_sens = element.GetValue(StructuralMechanicsApplication.I22_SENSITIVITY)
                total_i22_sens += i22_sens
            except:
                pass
            try:
                E_sens = element.GetValue(StructuralMechanicsApplication.YOUNG_MODULUS_SENSITIVITY)
                total_E_sens += E_sens
            except:
                pass
        
        print(f"\nTotal Shape Sensitivity:")
        print(f"  Sum(dJ/dX) = {total_shape_sens[0]:>14.6e}")
        print(f"  Sum(dJ/dY) = {total_shape_sens[1]:>14.6e}")
        print(f"  Sum(dJ/dZ) = {total_shape_sens[2]:>14.6e}")
        print(f"\nTotal I22 Sensitivity:")
        print(f"  Sum(dJ/dI22) = {total_i22_sens:>14.6e}")
        print(f"\nTotal Young's Modulus Sensitivity:")
        print(f"  Sum(dJ/dE) = {total_E_sens:>14.6e}")
        
    except Exception as e:
        print(f"Error computing summary: {e}")


def print_vtk_info():
    """Print information about VTK output files."""
    print("\n" + "="*70)
    print("VTK OUTPUT INFORMATION")
    print("="*70)
    print("\nVTK files can be visualized using ParaView (https://www.paraview.org)")
    print(f"\nOutput location: {os.getcwd()}")
    print("\nPrimal output folder: vtk_output_primal/")
    print("  - Contains: DISPLACEMENT, ROTATION, REACTION, POINT_LOAD")
    print("  - Element data: CROSS_AREA, I22, I33")
    print("\nAdjoint output folder: vtk_output_adjoint/")
    print("  - Contains: ADJOINT_DISPLACEMENT, ADJOINT_ROTATION, SHAPE_SENSITIVITY")
    print("  - Element data: I22_SENSITIVITY, YOUNG_MODULUS_SENSITIVITY")
    print("  - Condition data: POINT_LOAD_SENSITIVITY")
    print("\nTo visualize in ParaView:")
    print("  1. Open ParaView")
    print("  2. File -> Open -> Select the .vtk files")
    print("  3. Apply 'Tube' filter to see beam elements better")
    print("  4. Color by desired variable (e.g., DISPLACEMENT, SHAPE_SENSITIVITY)")


def main():
    """Main function to run the analysis."""
    print("="*70)
    print("KRATOS BEAM STRUCTURE ANALYSIS")
    print("Primal + Adjoint Sensitivity Analysis")
    print("="*70)
    print(f"\nWorking directory: {os.getcwd()}")
    
    # Create output directories if they don't exist
    os.makedirs("vtk_output_primal", exist_ok=True)
    os.makedirs("vtk_output_adjoint", exist_ok=True)
    
    # Step 1: Run primal analysis
    print("\n>>> Running primal analysis...")
    try:
        primal_model, primal_solution = run_primal_analysis("beam_test_parameters.json")
        print_primal_results(primal_model["Structure"])
    except Exception as e:
        print(f"Primal analysis error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 2: Run adjoint analysis
    print("\n>>> Running adjoint analysis...")
    try:
        adjoint_model = run_adjoint_analysis("beam_test_local_stress_adjoint_parameters.json", primal_solution)
        print_adjoint_results(adjoint_model["Structure"])
        print_summary(adjoint_model["Structure"])
    except Exception as e:
        print(f"Adjoint analysis error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Print VTK information
    print_vtk_info()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())