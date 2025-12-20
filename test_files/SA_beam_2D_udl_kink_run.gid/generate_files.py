"""
AUTO-GENERATOR FOR SENSITIVITY ANALYSIS FILES
==============================================
Reads config.py and generates:
  - Structure.mdpa
  - ProjectParameters_primal.json
  - ProjectParameters_dual.json
  - main_primal.py
  - main_dual.py
"""

import os
import json
from config import (
    BEAM_LENGTH, NUM_ELEMENTS, SENSITIVITY_LOCATION_X,
    YOUNGS_MODULUS, POISSON_RATIO, DENSITY,
    UDL_VALUE,
    OUTPUT_BASE_DIR, compute_derived
)


def generate_mdpa():
    """Generate the .mdpa file with proper node numbering for hinge."""
    
    derived = compute_derived()
    element_length = derived['element_length']
    hinge_left = derived['hinge_left_node']
    hinge_right = derived['hinge_right_node']
    
    # Build node list
    nodes = []
    node_id = 1
    
    for i in range(NUM_ELEMENTS + 1):
        x = i * element_length
        
        # Check if this is the hinge location
        if abs(x - SENSITIVITY_LOCATION_X) < 1e-10 and 0 < x < BEAM_LENGTH:
            # Create two coincident nodes for the hinge
            nodes.append((node_id, x, 0.0, 0.0))      # Left side of hinge
            node_id += 1
            nodes.append((node_id, x, 0.0, 0.0))      # Right side of hinge
            node_id += 1
        else:
            nodes.append((node_id, x, 0.0, 0.0))
            node_id += 1
    
    # Build element connectivity
    elements = []
    current_node = 1
    for elem_id in range(1, NUM_ELEMENTS + 1):
        node1 = current_node
        
        # Check if we cross the hinge
        x_end = elem_id * element_length
        if abs(x_end - SENSITIVITY_LOCATION_X) < 1e-10 and x_end < BEAM_LENGTH:
            node2 = current_node + 1  # Left hinge node
            current_node = current_node + 2  # Skip to right hinge node for next element
        else:
            node2 = current_node + 1
            current_node = current_node + 1
        
        elements.append((elem_id, node1, node2))
    
    # Identify boundary and hinge nodes
    first_node = 1
    last_node = nodes[-1][0]
    
    # Find hinge nodes (coincident nodes at sensitivity location)
    hinge_nodes = []
    for i in range(len(nodes) - 1):
        if abs(nodes[i][1] - nodes[i+1][1]) < 1e-10:  # Same x-coordinate
            hinge_nodes = [nodes[i][0], nodes[i+1][0]]
            break
    
    # Find dummy load node (for dual analysis - at hinge location)
    dummy_load_node = hinge_nodes[0] if hinge_nodes else 2
    
    mdpa_content = f"""Begin ModelPartData
End ModelPartData

Begin Properties 1
End Properties

Begin Nodes
"""
    
    for node in nodes:
        mdpa_content += f"    {node[0]}   {node[1]:.10f}   {node[2]:.10f}   {node[3]:.10f}\n"
    
    mdpa_content += """End Nodes

Begin Elements CrBeamElement2D2N
"""
    
    for elem in elements:
        mdpa_content += f"    {elem[0]}   1   {elem[1]}   {elem[2]}\n"
    
    mdpa_content += """
End Elements

Begin ElementalData LOCAL_AXIS_2
"""
    
    for elem in elements:
        mdpa_content += f"    {elem[0]} [3]( 0.0, 1.0, 0.0)\n"
    
    mdpa_content += """
End ElementalData

Begin Conditions PointMomentCondition3D1N
    1 1 """ + str(dummy_load_node) + """

End Conditions

Begin SubModelPart Parts_Beam_Beams
    Begin SubModelPartNodes
"""
    
    for node in nodes:
        mdpa_content += f"        {node[0]}\n"
    
    mdpa_content += """
    End SubModelPartNodes
    Begin SubModelPartElements
"""
    
    for elem in elements:
        mdpa_content += f"        {elem[0]}\n"
    
    mdpa_content += f"""
    End SubModelPartElements
End SubModelPart

Begin SubModelPart Support_Left
    Begin SubModelPartNodes
        {first_node}
    End SubModelPartNodes
End SubModelPart

Begin SubModelPart Support_Right
    Begin SubModelPartNodes
        {last_node}
    End SubModelPartNodes
End SubModelPart
"""
    
    if hinge_nodes:
        mdpa_content += f"""
Begin SubModelPart Hinge_Left
    Begin SubModelPartNodes
        {hinge_nodes[0]}
    End SubModelPartNodes
End SubModelPart

Begin SubModelPart Hinge_Right
    Begin SubModelPartNodes
        {hinge_nodes[1]}
    End SubModelPartNodes
End SubModelPart
"""
    
    mdpa_content += f"""
Begin SubModelPart DummyLoad
    Begin SubModelPartNodes
        {dummy_load_node}
    End SubModelPartNodes
    Begin SubModelPartConditions
        1
    End SubModelPartConditions
End SubModelPart
"""
    
    return mdpa_content, hinge_nodes, first_node, last_node, dummy_load_node


def generate_primal_json(first_node, last_node):
    """Generate ProjectParameters for primal analysis (actual loading)."""
    
    derived = compute_derived()
    
    params = {
        "problem_data": {
            "problem_name": "sensitivity_primal",
            "parallel_type": "OpenMP",
            "echo_level": 1,
            "start_time": 0.0,
            "end_time": 1.0
        },
        "solver_settings": {
            "solver_type": "Static",
            "model_part_name": "Structure",
            "domain_size": 2,
            "echo_level": 1,
            "analysis_type": "linear",
            "model_import_settings": {
                "input_type": "mdpa",
                "input_filename": "Structure"
            },
            "material_import_settings": {
                "materials_filename": "StructuralMaterials.json"
            },
            "time_stepping": {
                "time_step": 1.0
            },
            "rotation_dofs": True
        },
        "processes": {
            "constraints_process_list": [
                {
                    "python_module": "assign_vector_variable_process",
                    "kratos_module": "KratosMultiphysics",
                    "process_name": "AssignVectorVariableProcess",
                    "Parameters": {
                        "model_part_name": "Structure.Support_Left",
                        "variable_name": "DISPLACEMENT",
                        "constrained": [True, True, True],
                        "value": [0.0, 0.0, 0.0]
                    }
                },
                {
                    "python_module": "assign_vector_variable_process",
                    "kratos_module": "KratosMultiphysics",
                    "process_name": "AssignVectorVariableProcess",
                    "Parameters": {
                        "model_part_name": "Structure.Support_Left",
                        "variable_name": "ROTATION",
                        "constrained": [True, True, True],
                        "value": [0.0, 0.0, 0.0]
                    }
                },
                {
                    "python_module": "assign_vector_variable_process",
                    "kratos_module": "KratosMultiphysics",
                    "process_name": "AssignVectorVariableProcess",
                    "Parameters": {
                        "model_part_name": "Structure.Support_Right",
                        "variable_name": "DISPLACEMENT",
                        "constrained": [True, True, True],
                        "value": [0.0, 0.0, 0.0]
                    }
                },
                {
                    "python_module": "assign_vector_variable_process",
                    "kratos_module": "KratosMultiphysics",
                    "process_name": "AssignVectorVariableProcess",
                    "Parameters": {
                        "model_part_name": "Structure.Support_Right",
                        "variable_name": "ROTATION",
                        "constrained": [True, True, True],
                        "value": [0.0, 0.0, 0.0]
                    }
                }
            ],
            "loads_process_list": [
                {
                    "python_module": "assign_vector_by_direction_to_condition_process",
                    "kratos_module": "KratosMultiphysics",
                    "process_name": "AssignVectorByDirectionToConditionProcess",
                    "Parameters": {
                        "model_part_name": "Structure.Parts_Beam_Beams",
                        "variable_name": "LINE_LOAD",
                        "modulus": abs(UDL_VALUE),
                        "direction": [0.0, -1.0 if UDL_VALUE < 0 else 1.0, 0.0]
                    }
                }
            ],
            "list_other_processes": []
        },
        "output_processes": {
            "vtk_output": [
                {
                    "python_module": "vtk_output_process",
                    "kratos_module": "KratosMultiphysics",
                    "process_name": "VtkOutputProcess",
                    "Parameters": {
                        "model_part_name": "Structure",
                        "output_path": "vtk_output_primal",
                        "file_format": "ascii",
                        "output_precision": 7,
                        "nodal_solution_step_data_variables": ["DISPLACEMENT", "ROTATION"],
                        "element_data_value_variables": ["MOMENT"]
                    }
                }
            ]
        }
    }
    
    return params


def generate_dual_json(hinge_nodes, first_node, last_node, dummy_load_node):
    """Generate ProjectParameters for dual analysis (unit kink)."""
    
    params = {
        "problem_data": {
            "problem_name": "sensitivity_dual",
            "parallel_type": "OpenMP",
            "echo_level": 1,
            "start_time": 0.0,
            "end_time": 1.0
        },
        "solver_settings": {
            "solver_type": "Static",
            "model_part_name": "Structure",
            "domain_size": 2,
            "echo_level": 1,
            "analysis_type": "linear",
            "model_import_settings": {
                "input_type": "mdpa",
                "input_filename": "Structure"
            },
            "material_import_settings": {
                "materials_filename": "StructuralMaterials.json"
            },
            "time_stepping": {
                "time_step": 1.0
            },
            "rotation_dofs": True
        },
        "processes": {
            "constraints_process_list": [
                {
                    "python_module": "assign_vector_variable_process",
                    "kratos_module": "KratosMultiphysics",
                    "process_name": "AssignVectorVariableProcess",
                    "Parameters": {
                        "model_part_name": "Structure.Support_Left",
                        "variable_name": "DISPLACEMENT",
                        "constrained": [True, True, True],
                        "value": [0.0, 0.0, 0.0]
                    }
                },
                {
                    "python_module": "assign_vector_variable_process",
                    "kratos_module": "KratosMultiphysics",
                    "process_name": "AssignVectorVariableProcess",
                    "Parameters": {
                        "model_part_name": "Structure.Support_Left",
                        "variable_name": "ROTATION",
                        "constrained": [True, True, True],
                        "value": [0.0, 0.0, 0.0]
                    }
                },
                {
                    "python_module": "assign_vector_variable_process",
                    "kratos_module": "KratosMultiphysics",
                    "process_name": "AssignVectorVariableProcess",
                    "Parameters": {
                        "model_part_name": "Structure.Support_Right",
                        "variable_name": "DISPLACEMENT",
                        "constrained": [True, True, True],
                        "value": [0.0, 0.0, 0.0]
                    }
                },
                {
                    "python_module": "assign_vector_variable_process",
                    "kratos_module": "KratosMultiphysics",
                    "process_name": "AssignVectorVariableProcess",
                    "Parameters": {
                        "model_part_name": "Structure.Support_Right",
                        "variable_name": "ROTATION",
                        "constrained": [True, True, True],
                        "value": [0.0, 0.0, 0.0]
                    }
                },
                {
                    "python_module": "assign_vector_variable_process",
                    "kratos_module": "KratosMultiphysics",
                    "process_name": "AssignVectorVariableProcess",
                    "Parameters": {
                        "model_part_name": "Structure.Hinge_Left",
                        "variable_name": "ROTATION",
                        "constrained": [False, False, True],
                        "value": [0.0, 0.0, -0.5]
                    }
                },
                {
                    "python_module": "assign_vector_variable_process",
                    "kratos_module": "KratosMultiphysics",
                    "process_name": "AssignVectorVariableProcess",
                    "Parameters": {
                        "model_part_name": "Structure.Hinge_Right",
                        "variable_name": "ROTATION",
                        "constrained": [False, False, True],
                        "value": [0.0, 0.0, 0.5]
                    }
                }
            ],
            "loads_process_list": [
                {
                    "python_module": "assign_vector_by_direction_to_condition_process",
                    "kratos_module": "KratosMultiphysics",
                    "process_name": "AssignVectorByDirectionToConditionProcess",
                    "Parameters": {
                        "model_part_name": "Structure.DummyLoad",
                        "variable_name": "POINT_MOMENT",
                        "modulus": 0.0,
                        "direction": [0.0, 0.0, 1.0]
                    }
                }
            ],
            "list_other_processes": []
        },
        "output_processes": {
            "vtk_output": [
                {
                    "python_module": "vtk_output_process",
                    "kratos_module": "KratosMultiphysics",
                    "process_name": "VtkOutputProcess",
                    "Parameters": {
                        "model_part_name": "Structure",
                        "output_path": "vtk_output_dual",
                        "file_format": "ascii",
                        "output_precision": 7,
                        "nodal_solution_step_data_variables": ["DISPLACEMENT", "ROTATION"],
                        "element_data_value_variables": ["MOMENT"]
                    }
                }
            ]
        }
    }
    
    return params


def generate_materials_json():
    """Generate StructuralMaterials.json."""
    
    derived = compute_derived()
    
    materials = {
        "properties": [
            {
                "model_part_name": "Structure.Parts_Beam_Beams",
                "properties_id": 1,
                "Material": {
                    "constitutive_law": {
                        "name": "BeamConstitutiveLaw"
                    },
                    "Variables": {
                        "DENSITY": DENSITY,
                        "YOUNG_MODULUS": YOUNGS_MODULUS,
                        "POISSON_RATIO": POISSON_RATIO,
                        "CROSS_AREA": derived['area'],
                        "I33": derived['inertia']
                    },
                    "Tables": {}
                }
            }
        ]
    }
    
    return materials


def generate_main_primal():
    """Generate main script for primal analysis."""
    
    script = '''"""
Primal Analysis - Auto-generated
"""

import os
import KratosMultiphysics
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis

if __name__ == "__main__":
    
    with open("ProjectParameters_primal.json", 'r') as f:
        parameters = KratosMultiphysics.Parameters(f.read())
    
    model = KratosMultiphysics.Model()
    simulation = StructuralMechanicsAnalysis(model, parameters)
    simulation.Run()
    
    print("\\nPrimal analysis completed. Check vtk_output_primal/")
'''
    
    return script


def generate_main_dual(hinge_nodes):
    """Generate main script for dual analysis with MPC coupling."""
    
    hinge_left = hinge_nodes[0]
    hinge_right = hinge_nodes[1]
    
    script = f'''"""
Dual Analysis for Sensitivity - Auto-generated
Unit Kink at x = {SENSITIVITY_LOCATION_X}
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
        """Couple displacements at hinge nodes."""
        model_part = self.model["Structure"]
        
        node_left = model_part.GetNode({hinge_left})
        node_right = model_part.GetNode({hinge_right})
        
        print("\\n" + "="*60)
        print("DUAL ANALYSIS: UNIT KINK")
        print("="*60)
        print(f"Hinge at x = {SENSITIVITY_LOCATION_X}")
        print(f"Node {{node_left.Id}} (left): θ = -0.5 rad")
        print(f"Node {{node_right.Id}} (right): θ = +0.5 rad")
        print(f"Total kink = 1.0 rad")
        print("="*60)
        
        # Couple X displacement
        model_part.CreateNewMasterSlaveConstraint(
            "LinearMasterSlaveConstraint", 1,
            node_left, KratosMultiphysics.DISPLACEMENT_X,
            node_right, KratosMultiphysics.DISPLACEMENT_X,
            1.0, 0.0
        )
        
        # Couple Y displacement
        model_part.CreateNewMasterSlaveConstraint(
            "LinearMasterSlaveConstraint", 2,
            node_left, KratosMultiphysics.DISPLACEMENT_Y,
            node_right, KratosMultiphysics.DISPLACEMENT_Y,
            1.0, 0.0
        )
        
        print("Displacement coupling applied")
        print("="*60 + "\\n")
    
    def Finalize(self):
        super().Finalize()
        self._print_results()
    
    def _print_results(self):
        """Print final results."""
        model_part = self.model["Structure"]
        
        print("\\n" + "="*60)
        print("DUAL ANALYSIS RESULTS")
        print("="*60)
        
        print("\\nNODAL RESULTS:")
        print("-"*60)
        print(f"{{'Node':<6}}{{'X':<8}}{{'Disp_Y':<16}}{{'Rot_Z':<16}}")
        print("-"*60)
        
        for node in model_part.Nodes:
            disp = node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT)
            rot = node.GetSolutionStepValue(KratosMultiphysics.ROTATION)
            print(f"{{node.Id:<6}}{{node.X:<8.2f}}{{disp[1]:>+16.6e}}{{rot[2]:>+16.6e}}")
        
        # Kink verification
        rot_left = model_part.GetNode({hinge_left}).GetSolutionStepValue(KratosMultiphysics.ROTATION)[2]
        rot_right = model_part.GetNode({hinge_right}).GetSolutionStepValue(KratosMultiphysics.ROTATION)[2]
        
        print("-"*60)
        print(f"\\nKINK VERIFICATION:")
        print(f"  θ_left = {{rot_left:+.6f}} rad")
        print(f"  θ_right = {{rot_right:+.6f}} rad")
        print(f"  Δθ = {{rot_right - rot_left:+.6f}} rad (should be +1.0)")
        print("="*60)


if __name__ == "__main__":
    
    with open("ProjectParameters_dual.json", 'r') as f:
        parameters = KratosMultiphysics.Parameters(f.read())
    
    model = KratosMultiphysics.Model()
    simulation = DualKinkAnalysis(model, parameters)
    simulation.Run()
    
    print("\\nDual analysis completed. Check vtk_output_dual/")
'''
    
    return script


def generate_run_all():
    """Generate a master run script."""
    
    script = f'''"""
MASTER RUN SCRIPT - Auto-generated
===================================
Configuration:
  - Beam Length: {BEAM_LENGTH} m
  - Number of Elements: {NUM_ELEMENTS}
  - Sensitivity Location: x = {SENSITIVITY_LOCATION_X} m
  - UDL: {UDL_VALUE} N/m
"""

import subprocess
import sys

print("="*60)
print("SENSITIVITY ANALYSIS: dM/dEI")
print("="*60)
print(f"Sensitivity location: x = {SENSITIVITY_LOCATION_X} m")
print("="*60)

print("\\n[1/2] Running Primal Analysis...")
subprocess.run([sys.executable, "main_primal.py"], check=True)

print("\\n[2/2] Running Dual Analysis...")
subprocess.run([sys.executable, "main_dual.py"], check=True)

print("\\n" + "="*60)
print("BOTH ANALYSES COMPLETED")
print("="*60)
print("\\nTo compute sensitivity: dM/dEI = M_primal × M_dual / EI")
print("="*60)
'''
    
    return script


def main():
    """Main function to generate all files."""
    
    print("="*60)
    print("SENSITIVITY ANALYSIS FILE GENERATOR")
    print("="*60)
    print(f"Configuration:")
    print(f"  Beam Length: {BEAM_LENGTH} m")
    print(f"  Number of Elements: {NUM_ELEMENTS}")
    print(f"  Sensitivity Location: x = {SENSITIVITY_LOCATION_X} m")
    print("="*60)
    
    # Create output directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # Generate MDPA
    print("\n[1/7] Generating Structure.mdpa...")
    mdpa_content, hinge_nodes, first_node, last_node, dummy_load_node = generate_mdpa()
    with open(os.path.join(OUTPUT_BASE_DIR, "Structure.mdpa"), 'w') as f:
        f.write(mdpa_content)
    print(f"       Hinge nodes: {hinge_nodes}")
    
    # Generate primal JSON
    print("[2/7] Generating ProjectParameters_primal.json...")
    primal_json = generate_primal_json(first_node, last_node)
    with open(os.path.join(OUTPUT_BASE_DIR, "ProjectParameters_primal.json"), 'w') as f:
        json.dump(primal_json, f, indent=4)
    
    # Generate dual JSON
    print("[3/7] Generating ProjectParameters_dual.json...")
    dual_json = generate_dual_json(hinge_nodes, first_node, last_node, dummy_load_node)
    with open(os.path.join(OUTPUT_BASE_DIR, "ProjectParameters_dual.json"), 'w') as f:
        json.dump(dual_json, f, indent=4)
    
    # Generate materials JSON
    print("[4/7] Generating StructuralMaterials.json...")
    materials_json = generate_materials_json()
    with open(os.path.join(OUTPUT_BASE_DIR, "StructuralMaterials.json"), 'w') as f:
        json.dump(materials_json, f, indent=4)
    
    # Generate main scripts
    print("[5/7] Generating main_primal.py...")
    with open(os.path.join(OUTPUT_BASE_DIR, "main_primal.py"), 'w') as f:
        f.write(generate_main_primal())
    
    print("[6/7] Generating main_dual.py...")
    with open(os.path.join(OUTPUT_BASE_DIR, "main_dual.py"), 'w') as f:
        f.write(generate_main_dual(hinge_nodes))
    
    print("[7/7] Generating run_all.py...")
    with open(os.path.join(OUTPUT_BASE_DIR, "run_all.py"), 'w') as f:
        f.write(generate_run_all())
    
    print("\n" + "="*60)
    print("ALL FILES GENERATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nOutput directory: {os.path.abspath(OUTPUT_BASE_DIR)}")
    print("\nGenerated files:")
    for f in os.listdir(OUTPUT_BASE_DIR):
        print(f"  - {f}")
    print("\nTo run the analysis:")
    print(f"  cd {OUTPUT_BASE_DIR}")
    print("  python run_all.py")
    print("="*60)


if __name__ == "__main__":
    main()