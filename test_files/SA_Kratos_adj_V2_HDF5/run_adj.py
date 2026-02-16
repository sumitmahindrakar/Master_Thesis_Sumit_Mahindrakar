import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import os
# import sys

# Set working directory to the project folder
os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\SA_Kratos_adj_V2_HDF5")
print(f"Working directory: {os.getcwd()}")

import KratosMultiphysics
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis

if __name__ == "__main__":
    with open("beam_test_local_stress_adjoint_parameters.json", 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    model = KratosMultiphysics.Model()
    simulation = StructuralMechanicsAnalysis(model, parameters)
    simulation.Run()