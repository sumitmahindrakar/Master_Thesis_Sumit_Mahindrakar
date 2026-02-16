import sys
import os
import KratosMultiphysics as Kratos
# from KratosMultiphysics.AdjointApplication import AdjointAnalysis
from KratosMultiphysics.StructuralMechanicsApplication import StructuralMechanicsApplication

# Try to register applications explicitly to ensure 'adjoint_static' solver is found
Kratos.RegisterAllApplications()

def RunAdjointSensitivityAnalysis():
    # --- File Configuration ---
    # Ensure these filenames match your actual file locations
    mdpa_file = "test_files/SA_Kratos_adj_V2_HDF5/Beam_structure.mdpa"
    materials_file = "test_files/SA_Kratos_adj_V2_HDF5/materials_beam.json"
    config_file = "test_files/SA_Kratos_adj_V2_HDF5/beam_test_local_stress_adjoint_parameters.json"

    # --- Validation ---
    if not os.path.exists(config_file):
        print(f"\nError: '{config_file}' not found.")
        print("Please ensure the 'property json adjoint' text provided is saved as '{config_file}' in the working directory.")
        return

    print(f"Loading configuration from: {config_file}")
    
    # --- Create ModelPart ---
    model = Kratos.Model()
    
    # --- Load Parameters ---
    with open(config_file, 'r') as file:
        project_parameters = Kratos.Parameters(file.read())
        
    # Override input paths from JSON to use local files
    project_parameters["solver_settings"]["model_import_settings"]["input_filename"] = mdpa_file
    project_parameters["solver_settings"]["material_import_settings"]["materials_filename"] = materials_file
    
    # Optional: Validate the parameters to catch errors early
    try:
        project_parameters.ValidateAndPrintSettings()
    except Exception as e:
        print(f"Validation Error: {e}")
        return

    # --- Run Analysis ---
    print("Starting Adjoint Sensitivity Analysis...")
    
    try:
        # Create the AdjointAnalysis wrapper
        # This wrapper handles the sequential execution (Primal -> Adjoint) 
        # if the settings allow, and processes the sensitivities.
        analysis = AdjointAnalysis(model, project_parameters)
        
        # Execute the analysis
        analysis.Execute()
        
        print("Adjoint Analysis Successfully Completed.")
        print("Results (Sensitivities) should be available in the 'vtk_output_adjoint' directory.")
        
    except Exception as e:
        print(f"\nAn error occurred during analysis execution:")
        print(f"{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    RunAdjointSensitivityAnalysis()