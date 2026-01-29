"""
Main script for running Kratos beam structure analysis
with adjoint sensitivity computation.

UDL (Uniformly Distributed Load) of 10 N/m in Y direction
"""

import os
import sys

# Set working directory to the project folder
os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\SA_Kratos_adj_V3_udl")
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
        model_part = self.model["Structure"]
        print("Transferring primal solution to adjoint model...")
        for node in model_part.Nodes:
            if node.Id in self.primal_solution:
                sol = self.primal_solution[node.Id]
                disp = sol["DISPLACEMENT"]
                rot = sol["ROTATION"]
                node.SetSolutionStepValue(KratosMultiphysics.DISPLACEMENT, 
                    KratosMultiphysics.Array3([disp[0], disp[1], disp[2]]))
                node.SetSolutionStepValue(KratosMultiphysics.ROTATION, 
                    KratosMultiphysics.Array3([rot[0], rot[1], rot[2]]))
        print("Primal solution transferred successfully.")


def run_primal_analysis(parameter_file_name):
    """Run primal analysis."""
    print(f"Loading parameters from: {parameter_file_name}")
    with open(parameter_file_name, 'r') as f:
        parameters = KratosMultiphysics.Parameters(f.read())
    
    model = KratosMultiphysics.Model()
    analysis = PrimalAnalysis(model, parameters)
    analysis.Run()
    
    return model, analysis.primal_solution


def run_adjoint_analysis(parameter_file_name, primal_solution):
    """Run adjoint analysis."""
    print(f"Loading parameters from: {parameter_file_name}")
    with open(parameter_file_name, 'r') as f:
        parameters = KratosMultiphysics.Parameters(f.read())
    
    # Print response function info
    resp_settings = parameters["solver_settings"]["response_function_settings"]
    print(f"Response type: {resp_settings['response_type'].GetString()}")
    if resp_settings.Has("traced_dof"):
        print(f"Traced DOF: {resp_settings['traced_dof'].GetString()}")
    if resp_settings.Has("stress_type"):
        print(f"Stress type: {resp_settings['stress_type'].GetString()}")
    
    model = KratosMultiphysics.Model()
    analysis = AdjointAnalysis(model, parameters, primal_solution)
    analysis.Run()
    
    return model


def safe_get_vector(node, variable, application=None):
    """Safely get a vector variable from a node."""
    try:
        if application:
            val = node.GetSolutionStepValue(getattr(application, variable))
        else:
            val = node.GetSolutionStepValue(getattr(KratosMultiphysics, variable))
        return [val[0], val[1], val[2]]
    except:
        return [0.0, 0.0, 0.0]


def safe_get_element_scalar(element, variable, application=None):
    """Safely get a scalar variable from an element."""
    try:
        if application:
            return element.GetValue(getattr(application, variable))
        else:
            return element.GetValue(getattr(KratosMultiphysics, variable))
    except:
        return 0.0


def write_vtk_primal(model_part, output_path):
    """Write VTK output for primal analysis."""
    os.makedirs(output_path, exist_ok=True)
    vtk_file = os.path.join(output_path, "Structure_primal.vtk")
    
    print(f"Writing primal VTK to: {vtk_file}")
    
    with open(vtk_file, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Primal Analysis Results\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        num_nodes = model_part.NumberOfNodes()
        f.write(f"\nPOINTS {num_nodes} float\n")
        
        node_map = {}
        idx = 0
        for node in model_part.Nodes:
            f.write(f"{node.X} {node.Y} {node.Z}\n")
            node_map[node.Id] = idx
            idx += 1
        
        num_elements = model_part.NumberOfElements()
        f.write(f"\nCELLS {num_elements} {num_elements * 3}\n")
        for element in model_part.Elements:
            nodes = [node.Id for node in element.GetNodes()]
            f.write(f"2 {node_map[nodes[0]]} {node_map[nodes[1]]}\n")
        
        f.write(f"\nCELL_TYPES {num_elements}\n")
        for _ in range(num_elements):
            f.write("3\n")
        
        f.write(f"\nPOINT_DATA {num_nodes}\n")
        
        f.write("\nVECTORS DISPLACEMENT float\n")
        for node in model_part.Nodes:
            d = safe_get_vector(node, "DISPLACEMENT")
            f.write(f"{d[0]} {d[1]} {d[2]}\n")
        
        f.write("\nVECTORS ROTATION float\n")
        for node in model_part.Nodes:
            r = safe_get_vector(node, "ROTATION")
            f.write(f"{r[0]} {r[1]} {r[2]}\n")
        
        f.write("\nVECTORS REACTION float\n")
        for node in model_part.Nodes:
            r = safe_get_vector(node, "REACTION")
            f.write(f"{r[0]} {r[1]} {r[2]}\n")
        
        f.write("\nSCALARS DISPLACEMENT_Y float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for node in model_part.Nodes:
            d = safe_get_vector(node, "DISPLACEMENT")
            f.write(f"{d[1]}\n")
    
    print(f"Primal VTK written: {os.path.getsize(vtk_file)} bytes")
    return vtk_file


def write_vtk_adjoint(model_part, output_path):
    """Write VTK output for adjoint analysis."""
    os.makedirs(output_path, exist_ok=True)
    vtk_file = os.path.join(output_path, "Structure_adjoint.vtk")
    
    print(f"Writing adjoint VTK to: {vtk_file}")
    
    with open(vtk_file, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Adjoint Sensitivity Results\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        num_nodes = model_part.NumberOfNodes()
        f.write(f"\nPOINTS {num_nodes} float\n")
        
        node_map = {}
        idx = 0
        for node in model_part.Nodes:
            f.write(f"{node.X} {node.Y} {node.Z}\n")
            node_map[node.Id] = idx
            idx += 1
        
        num_elements = model_part.NumberOfElements()
        f.write(f"\nCELLS {num_elements} {num_elements * 3}\n")
        for element in model_part.Elements:
            nodes = [node.Id for node in element.GetNodes()]
            f.write(f"2 {node_map[nodes[0]]} {node_map[nodes[1]]}\n")
        
        f.write(f"\nCELL_TYPES {num_elements}\n")
        for _ in range(num_elements):
            f.write("3\n")
        
        f.write(f"\nPOINT_DATA {num_nodes}\n")
        
        f.write("\nVECTORS DISPLACEMENT float\n")
        for node in model_part.Nodes:
            d = safe_get_vector(node, "DISPLACEMENT")
            f.write(f"{d[0]} {d[1]} {d[2]}\n")
        
        f.write("\nVECTORS ADJOINT_DISPLACEMENT float\n")
        for node in model_part.Nodes:
            d = safe_get_vector(node, "ADJOINT_DISPLACEMENT", StructuralMechanicsApplication)
            f.write(f"{d[0]} {d[1]} {d[2]}\n")
        
        f.write("\nVECTORS ADJOINT_ROTATION float\n")
        for node in model_part.Nodes:
            r = safe_get_vector(node, "ADJOINT_ROTATION", StructuralMechanicsApplication)
            f.write(f"{r[0]} {r[1]} {r[2]}\n")
        
        f.write("\nVECTORS SHAPE_SENSITIVITY float\n")
        for node in model_part.Nodes:
            s = safe_get_vector(node, "SHAPE_SENSITIVITY")
            f.write(f"{s[0]} {s[1]} {s[2]}\n")
        
        f.write("\nSCALARS SHAPE_SENSITIVITY_Y float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for node in model_part.Nodes:
            s = safe_get_vector(node, "SHAPE_SENSITIVITY")
            f.write(f"{s[1]}\n")
        
        f.write(f"\nCELL_DATA {num_elements}\n")
        
        f.write("\nSCALARS I22_SENSITIVITY float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for element in model_part.Elements:
            val = safe_get_element_scalar(element, "I22_SENSITIVITY", StructuralMechanicsApplication)
            f.write(f"{val}\n")
        
        f.write("\nSCALARS YOUNG_MODULUS_SENSITIVITY float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for element in model_part.Elements:
            val = safe_get_element_scalar(element, "YOUNG_MODULUS_SENSITIVITY", StructuralMechanicsApplication)
            f.write(f"{val}\n")
    
    print(f"Adjoint VTK written: {os.path.getsize(vtk_file)} bytes")
    return vtk_file


def print_results(model_part, is_adjoint=False):
    """Print analysis results."""
    
    if not is_adjoint:
        print("\n" + "="*70)
        print("PRIMAL RESULTS")
        print("="*70)
        
        print("\nDisplacements (Y-direction):")
        print("-"*40)
        for node in model_part.Nodes:
            d = node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT)
            print(f"  Node {node.Id:>2}: {d[1]:>14.6e}")
        
        print("\nReactions:")
        print("-"*40)
        for node in model_part.GetSubModelPart("DISPLACEMENT_support").Nodes:
            r = node.GetSolutionStepValue(KratosMultiphysics.REACTION)
            print(f"  Node {node.Id:>2}: [{r[0]:.4e}, {r[1]:.4e}, {r[2]:.4e}]")
    
    else:
        print("\n" + "="*70)
        print("ADJOINT RESULTS")
        print("="*70)
        
        print("\nAdjoint Displacements (Y-component):")
        print("-"*40)
        for node in model_part.Nodes:
            try:
                ad = node.GetSolutionStepValue(StructuralMechanicsApplication.ADJOINT_DISPLACEMENT)
                print(f"  Node {node.Id:>2}: {ad[1]:>14.6e}")
            except:
                print(f"  Node {node.Id:>2}: N/A")
        
        print("\nShape Sensitivities:")
        print("-"*60)
        print(f"  {'Node':>4} {'dJ/dX':>14} {'dJ/dY':>14} {'dJ/dZ':>14}")
        print("-"*60)
        try:
            sens_mp = model_part.GetSubModelPart("sensitivity_mp")
            for node in sens_mp.Nodes:
                try:
                    ss = node.GetSolutionStepValue(KratosMultiphysics.SHAPE_SENSITIVITY)
                    print(f"  {node.Id:>4} {ss[0]:>14.6e} {ss[1]:>14.6e} {ss[2]:>14.6e}")
                except:
                    print(f"  {node.Id:>4} {'N/A':>14} {'N/A':>14} {'N/A':>14}")
        except Exception as e:
            print(f"  Error: {e}")
        
        print("\nI22 Sensitivities:")
        print("-"*40)
        total_i22 = 0.0
        try:
            sens_mp = model_part.GetSubModelPart("sensitivity_mp")
            for elem in sens_mp.Elements:
                try:
                    val = elem.GetValue(StructuralMechanicsApplication.I22_SENSITIVITY)
                    total_i22 += val
                    print(f"  Element {elem.Id:>2}: {val:>14.6e}")
                except:
                    print(f"  Element {elem.Id:>2}: N/A")
            print("-"*40)
            print(f"  {'TOTAL':>10}: {total_i22:>14.6e}")
        except Exception as e:
            print(f"  Error: {e}")
        
        print("\nYoung's Modulus Sensitivities:")
        print("-"*40)
        total_E = 0.0
        try:
            sens_mp = model_part.GetSubModelPart("sensitivity_mp")
            for elem in sens_mp.Elements:
                try:
                    val = elem.GetValue(StructuralMechanicsApplication.YOUNG_MODULUS_SENSITIVITY)
                    total_E += val
                    print(f"  Element {elem.Id:>2}: {val:>14.6e}")
                except:
                    print(f"  Element {elem.Id:>2}: N/A")
            print("-"*40)
            print(f"  {'TOTAL':>10}: {total_E:>14.6e}")
        except Exception as e:
            print(f"  Error: {e}")
        
        print("\nLine Load Sensitivities:")
        print("-"*60)
        try:
            sens_mp = model_part.GetSubModelPart("sensitivity_mp")
            for cond in sens_mp.Conditions:
                try:
                    val = cond.GetValue(StructuralMechanicsApplication.LINE_LOAD_SENSITIVITY)
                    print(f"  Cond {cond.Id:>2}: [{val[0]:>10.4e}, {val[1]:>10.4e}, {val[2]:>10.4e}]")
                except:
                    pass
        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Main function."""
    print("="*70)
    print("KRATOS BEAM ANALYSIS - UDL 10 N/m in Y direction")
    print("="*70)
    
    os.makedirs("vtk_output_primal", exist_ok=True)
    os.makedirs("vtk_output_adjoint", exist_ok=True)
    
    # Primal analysis
    print("\n>>> RUNNING PRIMAL ANALYSIS...")
    print("-"*70)
    try:
        primal_model, primal_solution = run_primal_analysis("beam_test_parameters.json")
        print_results(primal_model["Structure"], is_adjoint=False)
        write_vtk_primal(primal_model["Structure"], "vtk_output_primal")
    except Exception as e:
        print(f"Primal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Adjoint analysis
    print("\n>>> RUNNING ADJOINT ANALYSIS...")
    print("-"*70)
    try:
        adjoint_model = run_adjoint_analysis("beam_test_local_stress_adjoint_parameters.json", primal_solution)
        print_results(adjoint_model["Structure"], is_adjoint=True)
        write_vtk_adjoint(adjoint_model["Structure"], "vtk_output_adjoint")
    except Exception as e:
        print(f"Adjoint error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Summary
    print("\n" + "="*70)
    print("OUTPUT FILES")
    print("="*70)
    
    files = [
        "vtk_output_primal/Structure_primal.vtk",
        "vtk_output_adjoint/Structure_adjoint.vtk"
    ]
    
    for f in files:
        if os.path.exists(f):
            print(f"  [OK] {f} ({os.path.getsize(f)} bytes)")
        else:
            print(f"  [MISSING] {f}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())