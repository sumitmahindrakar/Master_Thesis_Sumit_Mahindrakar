import sys
import time
import importlib

import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication as SMA

def CreateAnalysisStageWithFlushInstance(cls, global_model, parameters):
    class AnalysisStageWithFlush(cls):

        def __init__(self, model, project_parameters, flush_frequency=10.0):
            super().__init__(model, project_parameters)
            self.flush_frequency = flush_frequency
            self.last_flush = time.time()
            sys.stdout.flush()

        def Initialize(self):
            super().Initialize()
            sys.stdout.flush()

        def FinalizeSolutionStep(self):
            super().FinalizeSolutionStep()

            if self.parallel_type == "OpenMP":
                now = time.time()
                if now - self.last_flush > self.flush_frequency:
                    sys.stdout.flush()
                    self.last_flush = now

    return AnalysisStageWithFlush(global_model, parameters)


def find_available_beam_variables():
    """
    Find available variables for beam elements in Kratos
    """
    available_vars = {}
    
    # Check KratosMultiphysics core
    core_vars = ['MOMENT', 'FORCE', 'ROTATION', 'DISPLACEMENT']
    for var_name in core_vars:
        if hasattr(KratosMultiphysics, var_name):
            available_vars[var_name] = getattr(KratosMultiphysics, var_name)
    
    # Check StructuralMechanicsApplication
    sma_vars = [
        'MOMENT', 'FORCE', 'LOCAL_MOMENT_VECTOR', 'LOCAL_FORCE_VECTOR',
        'BEAM_CURVATURE', 'CURVATURE', 'I22', 'I33', 'CROSS_AREA',
        'LOCAL_AXES_VECTOR', 'SECTION_FORCE_MOMENT'
    ]
    for var_name in sma_vars:
        if hasattr(SMA, var_name):
            available_vars['SMA_' + var_name] = getattr(SMA, var_name)
    
    return available_vars


def extract_moment_sensitivity_method1(model_part):
    """
    Method 1: Direct Curvature Approach
    
    Theory: M = EI * κ
    Therefore: dM/d(EI) = κ (curvature)
    """
    
    print("\n" + "="*70)
    print("METHOD 1: DIRECT CURVATURE APPROACH")
    print("Theory: dM/d(EI) = κ (curvature)")
    print("="*70)
    
    # First, let's find what variables are available
    print("\nSearching for available variables...")
    available_vars = find_available_beam_variables()
    print(f"Found variables: {list(available_vars.keys())}")
    
    results = []
    process_info = model_part.ProcessInfo
    
    for element in model_part.Elements:
        elem_id = element.Id
        props = element.Properties
        
        # Get material properties
        E = props.GetValue(KratosMultiphysics.YOUNG_MODULUS)
        
        # Try to get I33 (for 2D bending)
        I = None
        for var_name in ['I33', 'I22']:
            try:
                if hasattr(SMA, var_name):
                    I = props.GetValue(getattr(SMA, var_name))
                    break
            except:
                continue
        
        if I is None:
            I = 5e-6  # Default from your material file
            print(f"Warning: Could not get I for element {elem_id}, using default {I}")
        
        EI = E * I
        
        # Get nodes and element length
        nodes = element.GetNodes()
        n1 = nodes[0]
        n2 = nodes[1]
        L = ((n2.X - n1.X)**2 + (n2.Y - n1.Y)**2 + (n2.Z - n1.Z)**2)**0.5
        
        # Get nodal rotations
        rot1_z = n1.GetSolutionStepValue(KratosMultiphysics.ROTATION_Z)
        rot2_z = n2.GetSolutionStepValue(KratosMultiphysics.ROTATION_Z)
        
        # Get nodal displacements
        disp1_y = n1.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y)
        disp2_y = n2.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y)
        
        # Try to extract moment from element using different approaches
        moment_extracted = False
        M_value = 0.0
        extraction_method = "none"
        
        # Approach 1: Try MOMENT variable from core Kratos
        if not moment_extracted:
            try:
                moments = element.CalculateOnIntegrationPoints(
                    KratosMultiphysics.MOMENT,
                    process_info
                )
                if len(moments) > 0:
                    M_vec = moments[0]
                    if hasattr(M_vec, '__len__'):
                        M_value = M_vec[2] if len(M_vec) >= 3 else M_vec[0]
                    else:
                        M_value = float(M_vec)
                    moment_extracted = True
                    extraction_method = "MOMENT"
            except Exception as e:
                pass
        
        # Approach 2: Try getting element values directly
        if not moment_extracted:
            try:
                M_vec = element.GetValue(KratosMultiphysics.MOMENT)
                if hasattr(M_vec, '__len__'):
                    M_value = M_vec[2] if len(M_vec) >= 3 else M_vec[0]
                else:
                    M_value = float(M_vec)
                moment_extracted = True
                extraction_method = "GetValue_MOMENT"
            except:
                pass
        
        # Approach 3: Calculate from nodal reactions/rotations
        # For Euler-Bernoulli beam: κ = dθ/dx ≈ (θ2 - θ1) / L
        # And M = EI * κ
        if not moment_extracted:
            # Calculate curvature from rotation gradient
            kappa = (rot2_z - rot1_z) / L
            M_value = EI * kappa
            moment_extracted = True
            extraction_method = "from_rotations"
        
        # Calculate curvature: κ = M / EI
        if abs(EI) > 1e-30:
            kappa = M_value / EI
        else:
            kappa = 0.0
        
        # dM/d(EI) = κ (the key result!)
        dM_dEI = kappa
        
        results.append({
            'element_id': elem_id,
            'gauss_point': 0,
            'node1': n1.Id,
            'node2': n2.Id,
            'length': L,
            'E': E,
            'I': I,
            'EI': EI,
            'rot1_z': rot1_z,
            'rot2_z': rot2_z,
            'disp1_y': disp1_y,
            'disp2_y': disp2_y,
            'moment': M_value,
            'curvature': kappa,
            'dM_dEI': dM_dEI,
            'method': extraction_method
        })
    
    return results


def print_sensitivity_results(results):
    """Print sensitivity results in a formatted table"""
    
    print("\n" + "-"*120)
    print(f"{'Elem':<6} {'Nodes':<10} {'L':<8} {'EI':<14} {'rot1_z':<12} {'rot2_z':<12} {'Moment':<14} {'Curvature':<14} {'dM/dEI':<14} {'Method':<15}")
    print("-"*120)
    
    for s in results:
        nodes_str = f"{s['node1']}-{s['node2']}"
        print(f"{s['element_id']:<6} {nodes_str:<10} {s['length']:<8.4f} "
              f"{s['EI']:<14.4e} {s['rot1_z']:<12.6e} {s['rot2_z']:<12.6e} "
              f"{s['moment']:<14.4e} {s['curvature']:<14.6e} {s['dM_dEI']:<14.6e} {s['method']:<15}")
    
    print("-"*120)
    
    # Summary statistics
    if results:
        curvatures = [r['curvature'] for r in results]
        moments = [r['moment'] for r in results]
        print(f"\nSummary:")
        print(f"  Max |moment|    = {max(abs(m) for m in moments):.6e}")
        print(f"  Max |curvature| = Max |dM/dEI| = {max(abs(k) for k in curvatures):.6e}")
        print(f"  Min |curvature| = Min |dM/dEI| = {min(abs(k) for k in curvatures):.6e}")
        
        # Check extraction methods used
        methods = set(r['method'] for r in results)
        print(f"  Extraction methods used: {methods}")


def save_sensitivity_to_file(results, filename="sensitivity_results_method1.csv"):
    """Save results to CSV file"""
    
    with open(filename, 'w') as f:
        # Header
        f.write('Element_ID,Node1,Node2,Length,E,I,EI,rot1_z,rot2_z,disp1_y,disp2_y,Moment,Curvature,dM_dEI,Method\n')
        
        for s in results:
            f.write(f"{s['element_id']},{s['node1']},{s['node2']},{s['length']},"
                   f"{s['E']},{s['I']},{s['EI']},{s['rot1_z']},{s['rot2_z']},"
                   f"{s['disp1_y']},{s['disp2_y']},{s['moment']},{s['curvature']},"
                   f"{s['dM_dEI']},{s['method']}\n")
    
    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":

    with open("test_files/SA_beam_2D_kink_2/ProjectParameters.json", 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    analysis_stage_module_name = parameters["analysis_stage"].GetString()
    analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
    analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

    analysis_stage_module = importlib.import_module(analysis_stage_module_name)
    analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

    global_model = KratosMultiphysics.Model()
    simulation = CreateAnalysisStageWithFlushInstance(analysis_stage_class, global_model, parameters)
    
    # Run the simulation
    simulation.Run()
    
    # =========================================================
    # METHOD 1: Extract sensitivity using direct curvature
    # =========================================================
    model_part = global_model.GetModelPart("Structure")
    
    print("\n" + "#"*70)
    print("# SENSITIVITY ANALYSIS: dM/d(EI)")
    print("#"*70)
    
    # Extract sensitivities
    sensitivities = extract_moment_sensitivity_method1(model_part)
    
    # Print results
    print_sensitivity_results(sensitivities)
    
    # Save to file
    save_sensitivity_to_file(sensitivities, "test_files/SA_beam_2D_kink_2/sensitivity_method1.csv")
    
    # =========================================================
    # Additional: Print by SubModelPart (different EI regions)
    # =========================================================
    print("\n" + "="*70)
    print("SENSITIVITY BY SUBMODELPART")
    print("="*70)
    
    for sub_mp_name in ["Parts_Beam_Beams", "Parts_Beam_Beams_2"]:
        try:
            sub_mp = model_part.GetSubModelPart(sub_mp_name)
            elem_ids = [e.Id for e in sub_mp.Elements]
            print(f"\n--- {sub_mp_name} ---")
            
            sub_results = [s for s in sensitivities if s['element_id'] in elem_ids]
            
            if sub_results:
                avg_kappa = sum(abs(s['curvature']) for s in sub_results) / len(sub_results)
                avg_moment = sum(abs(s['moment']) for s in sub_results) / len(sub_results)
                print(f"  Number of elements: {len(sub_results)}")
                print(f"  EI value: {sub_results[0]['EI']:.4e}")
                print(f"  Average |moment|: {avg_moment:.6e}")
                print(f"  Average |curvature| = Average |dM/dEI|: {avg_kappa:.6e}")
        except Exception as e:
            print(f"Error processing {sub_mp_name}: {e}")