"""
Dual Analysis for Sensitivity: Unit Kink at Midspan
Fixed-Fixed Beam with prescribed rotations θ₃=-0.5, θ₄=+0.5 (total kink = 1.0)
"""

import os
import KratosMultiphysics
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis


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
        
        node_left = model_part.GetNode(11)
        node_right = model_part.GetNode(12)
        
        print("\n" + "="*60)
        print("DUAL ANALYSIS: UNIT KINK AT MIDSPAN")
        print("="*60)
        print(f"Node 11 (x={node_left.X}): θ = -0.5 rad (prescribed)")
        print(f"Node 12 (x={node_right.X}): θ = +0.5 rad (prescribed)")
        print(f"Total kink = 1.0 rad")
        print("="*60)
        
        # Couple X displacement: u_x(11) = u_x(12)
        model_part.CreateNewMasterSlaveConstraint(
            "LinearMasterSlaveConstraint", 1,
            node_left, KratosMultiphysics.DISPLACEMENT_X,
            node_right, KratosMultiphysics.DISPLACEMENT_X,
            1.0, 0.0
        )
        
        # Couple Y displacement: u_y(11) = u_y(12)
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
        rot_11 = model_part.GetNode(11).GetSolutionStepValue(KratosMultiphysics.ROTATION)[2]
        rot_12 = model_part.GetNode(12).GetSolutionStepValue(KratosMultiphysics.ROTATION)[2]
        
        print("-"*60)
        print(f"\nKINK VERIFICATION:")
        print(f"  θ₃ = {rot_11:+.6f} rad")
        print(f"  θ₄ = {rot_12:+.6f} rad")
        print(f"  Δθ = {rot_12 - rot_11:+.6f} rad (should be +1.0)")
        
        print("-"*60)
        print("\nCheck VTK file for MOMENT values (M_dual)")
        print("="*60)
if __name__ == "__main__":
    
    # Paths
    base_dir = "test_files/SA_beam_2D_udl_kink.gid"
    json_file = os.path.join(base_dir, "ProjectParameters_dual.json")
    output_dir = os.path.join(base_dir, "vtk_output_dual")
    
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