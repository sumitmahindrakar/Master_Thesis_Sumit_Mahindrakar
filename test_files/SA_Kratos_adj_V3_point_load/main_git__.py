import os
import sys
import traceback
import math

os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\SA_Kratos_adj_V3")

import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication as SMA
from KratosMultiphysics.StructuralMechanicsApplication import structural_mechanics_analysis
import KratosMultiphysics.kratos_utilities as kratos_utilities

def flush():
    """Force flush both stdout and stderr to see output before any crash."""
    sys.stdout.flush()
    sys.stderr.flush()

def check_for_nan(mp, var_name, variable):
    """Check if any node has NaN/Inf in the given variable."""
    bad_nodes = []
    for node in mp.Nodes:
        val = node.GetSolutionStepValue(variable)
        for i in range(3):
            if math.isnan(val[i]) or math.isinf(val[i]):
                bad_nodes.append(node.Id)
                break
    if bad_nodes:
        print(f"  *** FATAL: {var_name} has NaN/Inf at nodes: {bad_nodes}")
    return len(bad_nodes) > 0

def transfer_properties_to_elements(mp, label=""):
    """Transfer material properties from Properties to element data."""
    vars_list = [
        (KratosMultiphysics.YOUNG_MODULUS, "YOUNG_MODULUS"),
        (KratosMultiphysics.POISSON_RATIO, "POISSON_RATIO"),
        (KratosMultiphysics.DENSITY, "DENSITY"),
        (SMA.CROSS_AREA, "CROSS_AREA"),
        (SMA.TORSIONAL_INERTIA, "TORSIONAL_INERTIA"),
        (SMA.I22, "I22"),
        (SMA.I33, "I33"),
    ]
    
    count = 0
    for element in mp.Elements:
        for var, name in vars_list:
            if element.Properties.Has(var):
                element.SetValue(var, element.Properties[var])
        count += 1
    
    # Verify by printing first element
    if count > 0:
        first_elem = None
        for elem in mp.Elements:
            first_elem = elem
            break
        if first_elem:
            print(f"  [{label}] Element {first_elem.Id} verification:")
            for var, name in vars_list:
                if first_elem.Properties.Has(var):
                    prop_val = first_elem.Properties[var]
                    elem_val = first_elem.GetValue(var)
                    match = "OK" if abs(prop_val - elem_val) < 1e-15 else "MISMATCH!"
                    print(f"    {name}: props={prop_val}, elem_data={elem_val} [{match}]")
    
    print(f"  [{label}] Transferred properties to {count} elements")
    flush()
    return count

def print_element_lengths(mp):
    """Print element lengths to verify mesh."""
    print("  Element lengths:")
    for elem in mp.Elements:
        geom = elem.GetGeometry()
        n1 = geom[0]
        n2 = geom[1]
        dx = n2.X - n1.X
        dy = n2.Y - n1.Y
        dz = n2.Z - n1.Z
        L = math.sqrt(dx*dx + dy*dy + dz*dz)
        direction = f"[{dx/L:.3f}, {dy/L:.3f}, {dz/L:.3f}]"
        print(f"    Element {elem.Id}: nodes({n1.Id}->{n2.Id}), L={L:.6f}, dir={direction}")
    flush()

def main():
    print("=" * 60)
    print("ADJOINT BEAM SENSITIVITY - DEBUG MODE")
    print("=" * 60)
    flush()

    # ================================================================
    # STEP 1: Read parameters
    # ================================================================
    print("\n[STEP 1] Reading parameter files...")
    flush()

    with open("beam_test_parameters.json", 'r') as f:
        primal_params = KratosMultiphysics.Parameters(f.read())
    with open("beam_test_local_stress_adjoint_parameters.json", 'r') as f:
        adjoint_params = KratosMultiphysics.Parameters(f.read())

    mp_name = primal_params["solver_settings"]["model_part_name"].GetString()
    print(f"  Model part name: {mp_name}")
    
    # Print key adjoint settings
    resp = adjoint_params["solver_settings"]["response_function_settings"]
    print(f"  Traced element: {resp['traced_element_id'].GetInt()}")
    print(f"  Stress type: {resp['stress_type'].GetString()}")
    print(f"  Step size: {resp['step_size'].GetDouble()}")
    print(f"  Adapt step size: {resp['adapt_step_size'].GetBool()}")
    flush()

    # ================================================================
    # STEP 2: Primal analysis
    # ================================================================
    print("\n[STEP 2] PRIMAL ANALYSIS")
    print("-" * 40)

    print("  [2a] Creating primal model and initializing...")
    flush()
    model_primal = KratosMultiphysics.Model()
    primal = structural_mechanics_analysis.StructuralMechanicsAnalysis(
        model_primal, primal_params)
    primal.Initialize()
    print("  [2a] Primal initialized OK")
    flush()

    # Print mesh info
    mp = model_primal.GetModelPart(mp_name)
    print(f"\n  Mesh: {mp.NumberOfNodes()} nodes, {mp.NumberOfElements()} elements, {mp.NumberOfConditions()} conditions")
    print_element_lengths(mp)

    # Transfer properties
    print("\n  [2b] Transferring properties to primal elements...")
    flush()
    transfer_properties_to_elements(mp, "PRIMAL")

    # Run primal
    print("\n  [2c] Running primal solution...")
    flush()
    primal.RunSolutionLoop()
    print("  [2c] Primal solution completed OK")
    flush()

    # Check primal solution
    print("\n  [2d] Checking primal solution for NaN/Inf...")
    flush()
    has_nan = False
    has_nan |= check_for_nan(mp, "DISPLACEMENT", KratosMultiphysics.DISPLACEMENT)
    has_nan |= check_for_nan(mp, "ROTATION", KratosMultiphysics.ROTATION)

    if has_nan:
        print("  *** PRIMAL SOLUTION HAS NaN! ADJOINT WILL FAIL! ***")
    else:
        print("  Primal solution is valid (no NaN/Inf)")

    # Print primal displacements
    print("\n  Primal displacements:")
    for node in mp.Nodes:
        d = node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT)
        r = node.GetSolutionStepValue(KratosMultiphysics.ROTATION)
        print(f"    Node {node.Id:2d}: DISP=[{d[0]:+.6e}, {d[1]:+.6e}, {d[2]:+.6e}]  "
              f"ROT=[{r[0]:+.6e}, {r[1]:+.6e}, {r[2]:+.6e}]")
    flush()

    # Finalize primal
    print("\n  [2e] Finalizing primal...")
    flush()
    primal.Finalize()
    print("  [2e] Primal finalized OK")
    flush()

    # Check HDF5 files
    print("\n  [2f] Checking HDF5 output files...")
    for fname in [f"{mp_name}.h5", f"{mp_name}-1.0000.h5"]:
        if os.path.exists(fname):
            print(f"    {fname}: {os.path.getsize(fname)} bytes")
        else:
            print(f"    *** {fname}: NOT FOUND! ***")
    flush()

    # ================================================================
    # STEP 3: Adjoint initialization
    # ================================================================
    print("\n[STEP 3] ADJOINT ANALYSIS")
    print("-" * 40)

    print("  [3a] Creating adjoint model and initializing...")
    flush()
    model_adjoint = KratosMultiphysics.Model()
    adjoint = structural_mechanics_analysis.StructuralMechanicsAnalysis(
        model_adjoint, adjoint_params)
    adjoint.Initialize()
    print("  [3a] Adjoint initialized OK")
    flush()

    # Transfer properties to adjoint elements
    print("\n  [3b] Transferring properties to ADJOINT elements...")
    flush()
    adj_mp = model_adjoint.GetModelPart(mp_name)
    transfer_properties_to_elements(adj_mp, "ADJOINT")

    # Verify traced element exists and has correct data
    traced_id = resp['traced_element_id'].GetInt()
    print(f"\n  [3c] Verifying traced element {traced_id}...")
    flush()
    found = False
    for elem in adj_mp.Elements:
        if elem.Id == traced_id:
            found = True
            geom = elem.GetGeometry()
            n1, n2 = geom[0], geom[1]
            dx = n2.X - n1.X
            dy = n2.Y - n1.Y
            dz = n2.Z - n1.Z
            L = math.sqrt(dx*dx + dy*dy + dz*dz)
            print(f"    Found! Nodes {n1.Id} -> {n2.Id}")
            print(f"    Length: {L:.6f}")
            print(f"    Direction: [{dx/L:.4f}, {dy/L:.4f}, {dz/L:.4f}]")
            print(f"    I22 (elem data): {elem.GetValue(SMA.I22)}")
            print(f"    I33 (elem data): {elem.GetValue(SMA.I33)}")
            print(f"    E   (elem data): {elem.GetValue(KratosMultiphysics.YOUNG_MODULUS)}")
            print(f"    A   (elem data): {elem.GetValue(SMA.CROSS_AREA)}")
            print(f"    I22 (props):     {elem.Properties[SMA.I22]}")
            break
    if not found:
        print(f"    *** TRACED ELEMENT {traced_id} NOT FOUND! ***")
    flush()

    # ================================================================
    # STEP 4: Adjoint solve - STEP BY STEP
    # ================================================================
    print("\n[STEP 4] ADJOINT SOLVE (step by step)")
    print("-" * 40)

    print("  [4a] AdvanceInTime...")
    flush()
    adjoint.time = adjoint._GetSolver().AdvanceInTime(adjoint.time)
    print(f"    Time advanced to: {adjoint.time}")
    flush()

    print("\n  [4b] InitializeSolutionStep...")
    print("    (This reads HDF5 primal data and sets up response function)")
    flush()
    adjoint.InitializeSolutionStep()
    print("    InitializeSolutionStep completed OK")
    flush()

    # Verify primal data was loaded into adjoint model
    print("\n  [4b-check] Verifying primal data in adjoint model...")
    flush()
    for node in adj_mp.Nodes:
        d = node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT)
        if abs(d[0]) > 1e-15 or abs(d[1]) > 1e-15 or abs(d[2]) > 1e-15:
            print(f"    Node {node.Id}: DISP=[{d[0]:+.6e}, {d[1]:+.6e}, {d[2]:+.6e}]")
    has_nan_adj = check_for_nan(adj_mp, "DISPLACEMENT(adjoint model)", KratosMultiphysics.DISPLACEMENT)
    if has_nan_adj:
        print("    *** HDF5 loaded NaN into adjoint! ***")
    else:
        print("    Primal data loaded correctly into adjoint model")
    flush()

    print("\n  [4c] Predict...")
    flush()
    adjoint._GetSolver().Predict()
    print("    Predict completed OK")
    flush()

    print("\n  [4d] SolveSolutionStep...")
    print("    (This assembles pseudo-load, solves K^T * lambda = -dJ/du)")
    print("    (If crash happens here, it's the adjoint solve itself)")
    flush()
    converged = adjoint._GetSolver().SolveSolutionStep()
    print(f"    SolveSolutionStep completed OK, converged={converged}")
    flush()

    # Check adjoint solution
    print("\n  [4d-check] Checking adjoint solution...")
    flush()
    has_nan_adj_sol = check_for_nan(adj_mp, "ADJOINT_DISPLACEMENT", SMA.ADJOINT_DISPLACEMENT)
    has_nan_adj_rot = check_for_nan(adj_mp, "ADJOINT_ROTATION", SMA.ADJOINT_ROTATION)
    
    print("    Adjoint displacements:")
    for node in adj_mp.Nodes:
        ad = node.GetSolutionStepValue(SMA.ADJOINT_DISPLACEMENT)
        ar = node.GetSolutionStepValue(SMA.ADJOINT_ROTATION)
        print(f"    Node {node.Id:2d}: ADJ_DISP=[{ad[0]:+.6e}, {ad[1]:+.6e}, {ad[2]:+.6e}]  "
              f"ADJ_ROT=[{ar[0]:+.6e}, {ar[1]:+.6e}, {ar[2]:+.6e}]")
    flush()

    print("\n  [4e] FinalizeSolutionStep...")
    print("    (This computes sensitivities via semi-analytic method)")
    print("    (If crash happens here, it's in the sensitivity computation)")
    flush()
    adjoint.FinalizeSolutionStep()
    print("    FinalizeSolutionStep completed OK")
    flush()

    print("\n  [4f] OutputSolutionStep...")
    flush()
    adjoint.OutputSolutionStep()
    print("    OutputSolutionStep completed OK")
    flush()

    # ================================================================
    # STEP 5: Read and print results
    # ================================================================
    print("\n[STEP 5] RESULTS")
    print("=" * 60)

    print("\n  === I22_SENSITIVITY (dMY/dI22) ===")
    for elem in adj_mp.Elements:
        val = elem.GetValue(SMA.I22_SENSITIVITY)
        print(f"    Element {elem.Id:2d}: {val:+.10e}")
    flush()

    print("\n  === SHAPE_SENSITIVITY ===")
    for node in adj_mp.Nodes:
        ss = node.GetSolutionStepValue(KratosMultiphysics.SHAPE_SENSITIVITY)
        if abs(ss[0]) > 1e-15 or abs(ss[1]) > 1e-15 or abs(ss[2]) > 1e-15:
            print(f"    Node {node.Id:2d}: [{ss[0]:+.6e}, {ss[1]:+.6e}, {ss[2]:+.6e}]")
    flush()

    # ================================================================
    # STEP 6: Finalize
    # ================================================================
    print("\n[STEP 6] Finalizing adjoint...")
    flush()
    adjoint.Finalize()
    print("  Adjoint finalized OK")
    flush()

    # Check VTK output
    vtk_dir = "vtk_output_adjoint"
    if os.path.exists(vtk_dir):
        files = os.listdir(vtk_dir)
        print(f"\n  VTK output directory: {len(files)} files")
        for f in files:
            print(f"    {f}: {os.path.getsize(os.path.join(vtk_dir, f))} bytes")
    else:
        print(f"\n  *** VTK output directory '{vtk_dir}' NOT FOUND! ***")
    flush()

    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    flush()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        print("\n*** SystemExit caught ***")
        flush()
    except Exception as e:
        print(f"\n*** PYTHON EXCEPTION: {e} ***")
        traceback.print_exc()
        flush()
    finally:
        print("\n--- Script finished ---")
        flush()