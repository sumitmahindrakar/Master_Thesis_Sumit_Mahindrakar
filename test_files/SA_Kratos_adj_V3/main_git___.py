import os, sys, traceback, math

os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\SA_Kratos_adj_V3")

import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication as SMA
from KratosMultiphysics.StructuralMechanicsApplication import structural_mechanics_analysis

def flush():
    sys.stdout.flush()
    sys.stderr.flush()

def transfer_props(mp):
    for elem in mp.Elements:
        for var in [KratosMultiphysics.YOUNG_MODULUS, KratosMultiphysics.POISSON_RATIO,
                    KratosMultiphysics.DENSITY, SMA.CROSS_AREA, SMA.TORSIONAL_INERTIA,
                    SMA.I22, SMA.I33]:
            if elem.Properties.Has(var):
                elem.SetValue(var, elem.Properties[var])

def run_adjoint_test(adj_params, test_name):
    """Run one adjoint test with maximum granularity."""
    print(f"\n{'='*60}")
    print(f"ADJOINT TEST: {test_name}")
    print(f"{'='*60}")
    flush()

    resp = adj_params["solver_settings"]["response_function_settings"]
    traced_id = resp["traced_element_id"].GetInt()
    stress_type = resp["stress_type"].GetString()
    stress_loc = resp["stress_location"].GetInt()
    step_size = resp["step_size"].GetDouble()
    adapt = resp["adapt_step_size"].GetBool()
    print(f"  Settings: elem={traced_id}, stress={stress_type}, "
          f"loc={stress_loc}, h={step_size}, adapt={adapt}")
    flush()

    # Initialize
    model_a = KratosMultiphysics.Model()
    adj = structural_mechanics_analysis.StructuralMechanicsAnalysis(model_a, adj_params)
    
    print("  [1] Initialize...", end=" "); flush()
    adj.Initialize()
    print("OK"); flush()
    
    adj_mp = model_a.GetModelPart("Structure")
    transfer_props(adj_mp)
    pi = adj_mp.ProcessInfo

    # Advance time
    print("  [2] AdvanceInTime...", end=" "); flush()
    adj.time = adj._GetSolver().AdvanceInTime(adj.time)
    print(f"OK (t={adj.time})"); flush()

    # InitializeSolutionStep (loads HDF5 primal data)
    print("  [3] InitializeSolutionStep...", end=" "); flush()
    adj.InitializeSolutionStep()
    print("OK"); flush()

    # ========== STRESS COMPUTATION TESTS ==========
    print(f"\n  --- Stress tests on ALL elements ---")
    flush()

    for elem in adj_mp.Elements:
        eid = elem.Id
        print(f"  [Stress] Element {eid:2d}: ", end=""); flush()
        try:
            moments = elem.CalculateOnIntegrationPoints(
                KratosMultiphysics.MOMENT, pi)
            forces = elem.CalculateOnIntegrationPoints(
                KratosMultiphysics.FORCE, pi)
            n_gp = len(moments)
            my_vals = [m[1] for m in moments]  # MY component
            print(f"{n_gp} GPs, MY={my_vals}")
            flush()
        except Exception as e:
            print(f"CRASHED: {e}")
            flush()

    # ========== MANUAL FD TEST ON TRACED ELEMENT ==========
    print(f"\n  --- Manual FD test on element {traced_id} ---")
    flush()

    traced_elem = adj_mp.Elements[traced_id]
    geom = traced_elem.GetGeometry()
    nodes = [geom[i] for i in range(geom.PointsNumber())]

    # Get original stress
    print("  [FD] Computing original stress...", end=" "); flush()
    orig_moments = traced_elem.CalculateOnIntegrationPoints(
        KratosMultiphysics.MOMENT, pi)
    orig_MY = orig_moments[stress_loc][1]  # MY at stress_location
    print(f"MY_orig = {orig_MY:+.10e}"); flush()

    # Perturb each DOF and recompute
    dof_names = ["DISP_X", "DISP_Y", "DISP_Z", "ROT_X", "ROT_Y", "ROT_Z"]
    h = step_size

    for node_idx, node in enumerate(nodes):
        disp_orig = [
            node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X),
            node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y),
            node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Z),
        ]
        rot_orig = [
            node.GetSolutionStepValue(KratosMultiphysics.ROTATION_X),
            node.GetSolutionStepValue(KratosMultiphysics.ROTATION_Y),
            node.GetSolutionStepValue(KratosMultiphysics.ROTATION_Z),
        ]
        all_orig = disp_orig + rot_orig
        all_vars = [
            KratosMultiphysics.DISPLACEMENT_X,
            KratosMultiphysics.DISPLACEMENT_Y,
            KratosMultiphysics.DISPLACEMENT_Z,
            KratosMultiphysics.ROTATION_X,
            KratosMultiphysics.ROTATION_Y,
            KratosMultiphysics.ROTATION_Z,
        ]

        for dof_idx in range(6):
            var = all_vars[dof_idx]
            orig_val = all_orig[dof_idx]
            h_adapted = h * max(1.0, abs(orig_val)) if adapt else h

            # Perturb
            print(f"  [FD] Node {node.Id}, {dof_names[dof_idx]}: "
                  f"val={orig_val:+.6e}, h={h_adapted:.6e}...", end=" ")
            flush()

            node.SetSolutionStepValue(var, orig_val + h_adapted)

            # Recompute stress
            pert_moments = traced_elem.CalculateOnIntegrationPoints(
                KratosMultiphysics.MOMENT, pi)
            pert_MY = pert_moments[stress_loc][1]

            # Restore
            node.SetSolutionStepValue(var, orig_val)

            # FD gradient
            dMY = (pert_MY - orig_MY) / h_adapted
            is_nan = math.isnan(dMY) or math.isinf(dMY)
            status = "NaN!" if is_nan else "OK"
            print(f"dMY/du = {dMY:+.6e} [{status}]")
            flush()

    # ========== ATTEMPT ADJOINT SOLVE ==========
    print(f"\n  --- Adjoint Solve ---")
    flush()

    print("  [4] Predict...", end=" "); flush()
    adj._GetSolver().Predict()
    print("OK"); flush()

    print("  [5] SolveSolutionStep...", end=" "); flush()
    converged = adj._GetSolver().SolveSolutionStep()
    print(f"OK (converged={converged})"); flush()

    # Check adjoint solution for NaN
    print("  [5-check] Checking adjoint solution...")
    flush()
    for node in adj_mp.Nodes:
        ad = node.GetSolutionStepValue(SMA.ADJOINT_DISPLACEMENT)
        ar = node.GetSolutionStepValue(SMA.ADJOINT_ROTATION)
        for i in range(3):
            if math.isnan(ad[i]) or math.isinf(ad[i]):
                print(f"    *** NaN/Inf in ADJOINT_DISPLACEMENT at node {node.Id}!")
            if math.isnan(ar[i]) or math.isinf(ar[i]):
                print(f"    *** NaN/Inf in ADJOINT_ROTATION at node {node.Id}!")
    print("  [5-check] Done"); flush()

    print("  [6] FinalizeSolutionStep...", end=" "); flush()
    adj.FinalizeSolutionStep()
    print("OK"); flush()

    print("  [7] OutputSolutionStep...", end=" "); flush()
    adj.OutputSolutionStep()
    print("OK"); flush()

    # Results
    print(f"\n  --- Results ---")
    for elem in adj_mp.Elements:
        val = elem.GetValue(SMA.I22_SENSITIVITY)
        print(f"    Element {elem.Id:2d}: I22_SENS = {val:+.10e}")
    flush()

    adj.Finalize()
    print(f"\n  TEST '{test_name}' COMPLETED SUCCESSFULLY!")
    flush()
    return True


def main():
    print("STEP 0: Running primal analysis...")
    flush()

    with open("beam_test_parameters.json", 'r') as f:
        primal_params = KratosMultiphysics.Parameters(f.read())

    model_p = KratosMultiphysics.Model()
    primal = structural_mechanics_analysis.StructuralMechanicsAnalysis(
        model_p, primal_params)
    primal.Initialize()
    transfer_props(model_p.GetModelPart("Structure"))
    primal.RunSolutionLoop()
    primal.Finalize()
    print("STEP 0: Primal DONE.\n")
    flush()

    # ============================================================
    # TEST 1: Original settings
    # ============================================================
    with open("beam_test_local_stress_adjoint_parameters.json", 'r') as f:
        adj_params_1 = KratosMultiphysics.Parameters(f.read())
    run_adjoint_test(adj_params_1, "ORIGINAL (elem=18, MY, loc=1)")

    # ============================================================
    # TEST 2: Different traced element (vertical, element 1)
    # Only runs if TEST 1 crashes — script will terminate at crash
    # ============================================================
    with open("beam_test_local_stress_adjoint_parameters.json", 'r') as f:
        adj_params_2 = KratosMultiphysics.Parameters(f.read())
    adj_params_2["solver_settings"]["response_function_settings"]["traced_element_id"].SetInt(1)
    run_adjoint_test(adj_params_2, "VERTICAL ELEMENT (elem=1, MY, loc=1)")

    # ============================================================
    # TEST 3: stress_location=0 on element 18
    # ============================================================
    with open("beam_test_local_stress_adjoint_parameters.json", 'r') as f:
        adj_params_3 = KratosMultiphysics.Parameters(f.read())
    adj_params_3["solver_settings"]["response_function_settings"]["stress_location"].SetInt(0)
    run_adjoint_test(adj_params_3, "LOCATION 0 (elem=18, MY, loc=0)")

    # ============================================================
    # TEST 4: Larger step size
    # ============================================================
    with open("beam_test_local_stress_adjoint_parameters.json", 'r') as f:
        adj_params_4 = KratosMultiphysics.Parameters(f.read())
    adj_params_4["solver_settings"]["response_function_settings"]["step_size"].SetDouble(1e-4)
    adj_params_4["solver_settings"]["response_function_settings"]["adapt_step_size"].SetBool(False)
    run_adjoint_test(adj_params_4, "LARGE STEP (elem=18, MY, h=1e-4, no adapt)")

    # ============================================================
    # TEST 5: stress_type FX
    # ============================================================
    with open("beam_test_local_stress_adjoint_parameters.json", 'r') as f:
        adj_params_5 = KratosMultiphysics.Parameters(f.read())
    adj_params_5["solver_settings"]["response_function_settings"]["stress_type"].SetString("FX")
    run_adjoint_test(adj_params_5, "FORCE X (elem=18, FX, loc=1)")

    # ============================================================
    # TEST 6: stress_type MZ
    # ============================================================
    with open("beam_test_local_stress_adjoint_parameters.json", 'r') as f:
        adj_params_6 = KratosMultiphysics.Parameters(f.read())
    adj_params_6["solver_settings"]["response_function_settings"]["stress_type"].SetString("MZ")
    run_adjoint_test(adj_params_6, "MOMENT Z (elem=18, MZ, loc=1)")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        print("\n*** SystemExit ***")
    except Exception as e:
        print(f"\n*** Exception: {e} ***")
        traceback.print_exc()
    finally:
        print("\n--- Script finished ---")
        flush()