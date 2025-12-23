"""
Dual Analysis for Sensitivity: Unit Kink at Midspan
Fixed-Fixed Beam with prescribed rotations θ₃=-0.5, θ₄=+0.5 (total kink = 1.0)
"""

import os
import KratosMultiphysics
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis

#hing_node
left=26
right=27

class DualKinkAnalysis(StructuralMechanicsAnalysis):
    
    def __init__(self, model, project_parameters):
        super().__init__(model, project_parameters)
        self._mpc_created = False
    
    def Initialize(self):
        super().Initialize()
        
        if not self._mpc_created:
            self._mpc_created = True
            self._couple_hinge_displacements()
    
    def _couple_hinge_displacements(self):


        """Couple displacements at hinge nodes (rotations prescribed via JSON)."""
        model_part = self.model["Structure"]
        
        node_left = model_part.GetNode(left)
        node_right = model_part.GetNode(right)
        
        print("\n" + "="*60)
        print("DUAL ANALYSIS: UNIT KINK AT MIDSPAN")
        print("="*60)
        print(f"Node left (x={node_left.X}): θ = -0.5 rad (prescribed)")
        print(f"Node right (x={node_right.X}): θ = +0.5 rad (prescribed)")
        print(f"Total kink = 1.0 rad")
        print("="*60)
        
        # Couple X displacement: u_x(left) = u_x(right)
        model_part.CreateNewMasterSlaveConstraint(
            "LinearMasterSlaveConstraint", 1,
            node_left, KratosMultiphysics.DISPLACEMENT_X,
            node_right, KratosMultiphysics.DISPLACEMENT_X,
            1.0, 0.0
        )
        
        # Couple Y displacement: u_y(left) = u_y(right)
        model_part.CreateNewMasterSlaveConstraint(
            "LinearMasterSlaveConstraint", 2,
            node_left, KratosMultiphysics.DISPLACEMENT_Y,
            node_right, KratosMultiphysics.DISPLACEMENT_Y,
            1.0, 0.0
        )
        
        print("Hinge displacement coupling applied")
        print("="*60 + "\n")
    
    def Finalize(self):
        super().Finalize()
        self._print_results()
    
    def _print_results(self):
        """Print final results."""
        model_part = self.model["Structure"]
        
        print("\n" + "="*60)
        print("DUAL ANALYSIS RESULTS")
        print("="*60)
        
        # Nodal results
        print("\nNODAL RESULTS:")
        print("-"*60)
        print(f"{'Node':<6}{'X':<8}{'Disp_Y':<16}{'Rot_Z':<16}")
        print("-"*60)
        
        for node in model_part.Nodes:
            disp = node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT)
            rot = node.GetSolutionStepValue(KratosMultiphysics.ROTATION)
            print(f"{node.Id:<6}{node.X:<8.2f}{disp[1]:>+16.6e}{rot[2]:>+16.6e}")
        
        # Kink verification
        rot_left = model_part.GetNode(left).GetSolutionStepValue(KratosMultiphysics.ROTATION)[2]
        rot_right = model_part.GetNode(right).GetSolutionStepValue(KratosMultiphysics.ROTATION)[2]
        
        print("-"*60)
        print(f"\nKINK VERIFICATION:")
        print(f"  θ_left = {rot_left:+.6f} rad")
        print(f"  θ_right = {rot_right:+.6f} rad")
        print(f"  Δθ = {rot_right - rot_left:+.6f} rad (should be +1.0)")
        
        print("-"*60)
        print("\nCheck VTK file for MOMENT values (M_dual)")
        print("="*60)
if __name__ == "__main__":
    
    # Get the folder where THIS script is located
    FOLDER = os.path.dirname(os.path.abspath(__file__))
    # Paths
    # base_dir = "test_files/SA_beam_2D_udl_kink.gid"
    json_file = os.path.join(FOLDER, "ProjectParameters_dual.json")
    output_dir = os.path.join(FOLDER, "vtk_output_dual")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load parameters
    with open(json_file, 'r') as f:
        parameters = KratosMultiphysics.Parameters(f.read())
    
    # Run analysis
    print("\n" + "="*60)
    print("STARTING DUAL ANALYSIS")
    print("="*60 + "\n")
    
    model = KratosMultiphysics.Model()
    simulation = DualKinkAnalysis(model, parameters)
    simulation.Run()
    
    # Check output
    print(f"\nVTK Output: {os.path.abspath(output_dir)}")
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        print(f"Files: {files if files else 'EMPTY'}")