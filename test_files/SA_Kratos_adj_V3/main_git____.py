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

def matrix_has_nan(mat):
    for i in range(mat.Size1()):
        for j in range(mat.Size2()):
            if math.isnan(mat[i,j]) or math.isinf(mat[i,j]):
                return True
    return False

def vector_has_nan(vec):
    for i in range(vec.Size()):
        if math.isnan(vec[i]) or math.isinf(vec[i]):
            return True
    return False

def main():
    print("="*60)
    print("DEEP DIAGNOSTIC: Isolating the adjoint crash")
    print("="*60)
    flush()

    # ==============================================================
    # PART A: PRIMAL - verify moments ARE non-zero in primal model
    # ==============================================================
    print("\n[PART A] PRIMAL - checking moments after solve")
    flush()

    with open("beam_test_parameters.json", 'r') as f:
        primal_params = KratosMultiphysics.Parameters(f.read())

    # Remove eigenvalue process (can interfere)
    other_procs = primal_params["processes"]["list_other_processes"]
    clean_procs = KratosMultiphysics.Parameters("[]")
    for i in range(other_procs.size()):
        proc = other_procs[i]
        if "eigenvalues" not in proc["python_module"].GetString():
            clean_procs.Append(proc)
    primal_params["processes"]["list_other_processes"] = clean_procs

    model_p = KratosMultiphysics.Model()
    primal = structural_mechanics_analysis.StructuralMechanicsAnalysis(model_p, primal_params)
    primal.Initialize()
    mp = model_p.GetModelPart("Structure")
    transfer_props(mp)
    primal.RunSolutionLoop()

    print("\n  PRIMAL moments (should be NON-ZERO):")
    pi = mp.ProcessInfo
    for elem in mp.Elements:
        try:
            moments = elem.CalculateOnIntegrationPoints(KratosMultiphysics.MOMENT, pi)
            forces = elem.CalculateOnIntegrationPoints(KratosMultiphysics.FORCE, pi)
            m0 = moments[0]
            f0 = forces[0]
            print(f"    Elem {elem.Id:2d}: M=[{m0[0]:+.4e},{m0[1]:+.4e},{m0[2]:+.4e}]  "
                  f"F=[{f0[0]:+.4e},{f0[1]:+.4e},{f0[2]:+.4e}]")
        except Exception as e:
            print(f"    Elem {elem.Id:2d}: CRASH - {e}")
    flush()

    # Also check element info/type in primal
    print("\n  PRIMAL element info:")
    for elem in mp.Elements:
        print(f"    Elem {elem.Id}: {elem.Info()}")
        break
    flush()

    primal.Finalize()
    print("  Primal finalized OK")
    flush()

    # ==============================================================
    # PART B: ADJOINT - detailed component testing
    # ==============================================================
    print("\n" + "="*60)
    print("[PART B] ADJOINT ANALYSIS - component testing")
    print("="*60)
    flush()

    with open("beam_test_local_stress_adjoint_parameters.json", 'r') as f:
        adj_params = KratosMultiphysics.Parameters(f.read())

    model_a = KratosMultiphysics.Model()
    adj = structural_mechanics_analysis.StructuralMechanicsAnalysis(model_a, adj_params)
    adj.Initialize()
    adj_mp = model_a.GetModelPart("Structure")
    transfer_props(adj_mp)

    # Advance time and load HDF5
    adj.time = adj._GetSolver().AdvanceInTime(adj.time)
    adj.InitializeSolutionStep()
    pi_adj = adj_mp.ProcessInfo
    print("  Adjoint initialized and HDF5 loaded.")
    flush()

    # ==============================================================
    # PART C: Check adjoint element type
    # ==============================================================
    print("\n[PART C] ADJOINT element type")
    for elem in adj_mp.Elements:
        print(f"    Elem {elem.Id}: {elem.Info()}")
        break
    flush()

    # ==============================================================
    # PART D: CalculateLeftHandSide for each element
    # ==============================================================
    print("\n[PART D] CalculateLeftHandSide (K^T) for each adjoint element")
    flush()
    for elem in adj_mp.Elements:
        eid = elem.Id
        print(f"  Elem {eid:2d}: ", end=""); flush()
        try:
            lhs = KratosMultiphysics.Matrix()
            elem.CalculateLeftHandSide(lhs, pi_adj)
            nan = matrix_has_nan(lhs)
            diag_sum = sum(lhs[i,i] for i in range(min(6, lhs.Size1())))
            print(f"size={lhs.Size1()}x{lhs.Size2()}, diag_sum={diag_sum:.4e}, "
                  f"NaN={'YES!' if nan else 'no'}")
        except Exception as e:
            print(f"CRASH: {e}")
        flush()

    # ==============================================================
    # PART E: CalculateRightHandSide for each element
    # ==============================================================
    print("\n[PART E] CalculateRightHandSide (pseudo-load) for each adjoint element")
    flush()
    for elem in adj_mp.Elements:
        eid = elem.Id
        print(f"  Elem {eid:2d}: ", end=""); flush()
        try:
            rhs = KratosMultiphysics.Vector()
            elem.CalculateRightHandSide(rhs, pi_adj)
            nan = vector_has_nan(rhs)
            norm = math.sqrt(sum(rhs[i]**2 for i in range(rhs.Size())))
            print(f"size={rhs.Size()}, |rhs|={norm:.4e}, "
                  f"NaN={'YES!' if nan else 'no'}")
        except Exception as e:
            print(f"CRASH: {e}")
        flush()

    # ==============================================================
    # PART F: CalculateLocalSystem for each element
    # ==============================================================
    print("\n[PART F] CalculateLocalSystem (K^T and f together) for each element")
    flush()
    for elem in adj_mp.Elements:
        eid = elem.Id
        print(f"  Elem {eid:2d}: ", end=""); flush()
        try:
            lhs = KratosMultiphysics.Matrix()
            rhs = KratosMultiphysics.Vector()
            elem.CalculateLocalSystem(lhs, rhs, pi_adj)
            lhs_nan = matrix_has_nan(lhs)
            rhs_nan = vector_has_nan(rhs)
            rhs_norm = math.sqrt(sum(rhs[i]**2 for i in range(rhs.Size())))
            print(f"K:{lhs.Size1()}x{lhs.Size2()}, |f|={rhs_norm:.4e}, "
                  f"K_NaN={'YES!' if lhs_nan else 'no'}, "
                  f"f_NaN={'YES!' if rhs_nan else 'no'}")
        except Exception as e:
            print(f"CRASH: {e}")
        flush()

    # ==============================================================
    # PART G: Full stiffness matrix for element 18 (traced)
    # ==============================================================
    print("\n[PART G] Full K matrix for traced element 18")
    flush()
    try:
        elem18 = adj_mp.Elements[18]
        lhs = KratosMultiphysics.Matrix()
        elem18.CalculateLeftHandSide(lhs, pi_adj)
        n = lhs.Size1()
        print(f"  Size: {n}x{n}")
        print(f"  Diagonal:")
        for i in range(n):
            print(f"    K[{i},{i}] = {lhs[i,i]:+.6e}")
        # Check symmetry
        max_asym = 0.0
        for i in range(n):
            for j in range(i+1, n):
                asym = abs(lhs[i,j] - lhs[j,i])
                if asym > max_asym:
                    max_asym = asym
        print(f"  Max asymmetry: {max_asym:.6e}")
        # Frobenius norm
        fro = math.sqrt(sum(lhs[i,j]**2 for i in range(n) for j in range(n)))
        print(f"  Frobenius norm: {fro:.6e}")
    except Exception as e:
        print(f"  CRASH: {e}")
    flush()

    # ==============================================================
    # PART H: EquationIdVector for each element
    # ==============================================================
    print("\n[PART H] EquationIdVector for each element")
    flush()
    for elem in adj_mp.Elements:
        eid = elem.Id
        print(f"  Elem {eid:2d}: ", end=""); flush()
        try:
            eq_ids = elem.EquationIdVector(pi_adj)
            print(f"IDs = {list(eq_ids)}")
        except Exception as e:
            print(f"CRASH: {e}")
        flush()

    # ==============================================================
    # PART I: Check GetDofList for each element
    # ==============================================================
    print("\n[PART I] GetDofList for each element")
    flush()
    for elem in adj_mp.Elements:
        eid = elem.Id
        print(f"  Elem {eid:2d}: ", end=""); flush()
        try:
            dof_list = elem.GetDofList(pi_adj)
            dof_vars = [str(d.GetVariable().Name()) for d in dof_list]
            print(f"{len(dof_list)} DOFs: {dof_vars[:6]}...")
        except Exception as e:
            print(f"CRASH: {e}")
        flush()

    # ==============================================================
    # PART J: Predict
    # ==============================================================
    print("\n[PART J] Predict")
    flush()
    print("  Calling...", end=" "); flush()
    adj._GetSolver().Predict()
    print("OK"); flush()

    # ==============================================================
    # PART K: Try building the system manually
    # ==============================================================
    print("\n[PART K] Attempting to build the global system")
    flush()

    try:
        strategy = adj._GetSolver()._GetSolutionStrategy()
        print(f"  Strategy type: {type(strategy)}")
        flush()
    except Exception as e:
        print(f"  Could not get strategy: {e}")
    flush()

    try:
        builder = adj._GetSolver()._GetBuilderAndSolver()
        print(f"  Builder type: {type(builder)}")
        flush()
    except Exception as e:
        print(f"  Could not get builder: {e}")
    flush()

    try:
        scheme = adj._GetSolver()._GetScheme()
        print(f"  Scheme type: {type(scheme)}")
        flush()
    except Exception as e:
        print(f"  Could not get scheme: {e}")
    flush()

    # ==============================================================
    # PART L: SolveSolutionStep (the crash point)
    # ==============================================================
    print("\n[PART L] SolveSolutionStep")
    print("  If this crashes, look at the LAST printed part above for clue.")
    print("  Calling...", end=" "); flush()
    converged = adj._GetSolver().SolveSolutionStep()
    print(f"OK (converged={converged})"); flush()

    # ==============================================================
    # PART M: Results
    # ==============================================================
    print("\n[PART M] RESULTS")
    for elem in adj_mp.Elements:
        val = elem.GetValue(SMA.I22_SENSITIVITY)
        print(f"  Elem {elem.Id:2d}: I22_SENS = {val:+.10e}")
    flush()

    adj.FinalizeSolutionStep()
    adj.OutputSolutionStep()
    adj.Finalize()
    print("\nALL STEPS COMPLETED!")

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(f"\n*** SystemExit: {e} ***")
    except Exception as e:
        print(f"\n*** Exception: {e} ***")
        traceback.print_exc()
    finally:
        print("\n--- Script finished ---")
        flush()