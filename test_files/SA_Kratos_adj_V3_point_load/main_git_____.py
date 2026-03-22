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

def mat_nan(m):
    for i in range(m.Size1()):
        for j in range(m.Size2()):
            if math.isnan(m[i,j]) or math.isinf(m[i,j]):
                return True
    return False

def vec_nan(v):
    for i in range(v.Size()):
        if math.isnan(v[i]) or math.isinf(v[i]):
            return True
    return False

def vec_norm(v):
    return math.sqrt(sum(v[i]**2 for i in range(v.Size())))

def main():
    print("="*60)
    print("DIAGNOSTIC: Scheme-level and Condition-level testing")
    print("="*60)
    flush()

    # ==========================================
    # PRIMAL
    # ==========================================
    print("\n[PRIMAL] Running...")
    flush()

    with open("beam_test_parameters.json", 'r') as f:
        pp = KratosMultiphysics.Parameters(f.read())

    # Remove eigenvalue process
    other = pp["processes"]["list_other_processes"]
    clean = KratosMultiphysics.Parameters("[]")
    for i in range(other.size()):
        if "eigenvalues" not in other[i]["python_module"].GetString():
            clean.Append(other[i])
    pp["processes"]["list_other_processes"] = clean

    mp = KratosMultiphysics.Model()
    pa = structural_mechanics_analysis.StructuralMechanicsAnalysis(mp, pp)
    pa.Initialize()
    transfer_props(mp.GetModelPart("Structure"))
    pa.RunSolutionLoop()
    pa.Finalize()
    print("[PRIMAL] Done.")
    flush()

    # ==========================================
    # ADJOINT SETUP
    # ==========================================
    print("\n[ADJOINT] Setting up...")
    flush()

    with open("beam_test_local_stress_adjoint_parameters.json", 'r') as f:
        ap = KratosMultiphysics.Parameters(f.read())

    ma = KratosMultiphysics.Model()
    adj = structural_mechanics_analysis.StructuralMechanicsAnalysis(ma, ap)
    adj.Initialize()
    adj_mp = ma.GetModelPart("Structure")
    transfer_props(adj_mp)

    # Advance time + load HDF5
    adj.time = adj._GetSolver().AdvanceInTime(adj.time)
    adj.InitializeSolutionStep()
    pi = adj_mp.ProcessInfo
    print("[ADJOINT] Initialized, HDF5 loaded.")
    flush()

    # Get scheme, builder, strategy
    scheme = adj._GetSolver()._GetScheme()
    builder = adj._GetSolver()._GetBuilderAndSolver()
    strategy = adj._GetSolver()._GetSolutionStrategy()

    print(f"  Scheme: {type(scheme).__name__}")
    print(f"  Builder: {type(builder).__name__}")
    print(f"  Strategy: {type(strategy).__name__}")
    flush()

    # ==========================================
    # TEST 1: Condition-level operations
    # ==========================================
    print("\n" + "="*60)
    print("[TEST 1] CONDITION-LEVEL OPERATIONS")
    print("="*60)
    flush()

    for cond in adj_mp.Conditions:
        cid = cond.Id
        geom = cond.GetGeometry()
        n1, n2 = geom[0].Id, geom[1].Id
        print(f"\n  Condition {cid} (nodes {n1}->{n2}):")
        flush()

        # CalculateLeftHandSide
        print(f"    CalculateLeftHandSide...", end=" "); flush()
        try:
            lhs = KratosMultiphysics.Matrix()
            cond.CalculateLeftHandSide(lhs, pi)
            nan = mat_nan(lhs)
            fro = math.sqrt(sum(lhs[i,j]**2 for i in range(lhs.Size1()) for j in range(lhs.Size2())))
            print(f"OK size={lhs.Size1()}x{lhs.Size2()}, |K|={fro:.4e}, NaN={'YES!' if nan else 'no'}")
        except Exception as e:
            print(f"EXCEPTION: {e}")
        flush()

        # CalculateRightHandSide
        print(f"    CalculateRightHandSide...", end=" "); flush()
        try:
            rhs = KratosMultiphysics.Vector()
            cond.CalculateRightHandSide(rhs, pi)
            nan = vec_nan(rhs)
            norm = vec_norm(rhs)
            print(f"OK size={rhs.Size()}, |f|={norm:.4e}, NaN={'YES!' if nan else 'no'}")
        except Exception as e:
            print(f"EXCEPTION: {e}")
        flush()

        # CalculateLocalSystem
        print(f"    CalculateLocalSystem...", end=" "); flush()
        try:
            lhs2 = KratosMultiphysics.Matrix()
            rhs2 = KratosMultiphysics.Vector()
            cond.CalculateLocalSystem(lhs2, rhs2, pi)
            print(f"OK K:{lhs2.Size1()}x{lhs2.Size2()}, |f|={vec_norm(rhs2):.4e}")
        except Exception as e:
            print(f"EXCEPTION: {e}")
        flush()

        # EquationIdVector
        print(f"    EquationIdVector...", end=" "); flush()
        try:
            eq = cond.EquationIdVector(pi)
            print(f"OK IDs={list(eq)}")
        except Exception as e:
            print(f"EXCEPTION: {e}")
        flush()

        # GetDofList
        print(f"    GetDofList...", end=" "); flush()
        try:
            dofs = cond.GetDofList(pi)
            print(f"OK {len(dofs)} DOFs")
        except Exception as e:
            print(f"EXCEPTION: {e}")
        flush()

    # ==========================================
    # TEST 2: Scheme CalculateSystemContributions for ELEMENTS
    # ==========================================
    print("\n" + "="*60)
    print("[TEST 2] SCHEME CalculateSystemContributions - ELEMENTS")
    print("  (This is what the builder calls during assembly)")
    print("="*60)
    flush()

    for elem in adj_mp.Elements:
        eid = elem.Id
        print(f"  Scheme+Element {eid:2d}...", end=" "); flush()
        try:
            lhs = KratosMultiphysics.Matrix()
            rhs = KratosMultiphysics.Vector()
            eq_ids = KratosMultiphysics.Vector()  # might need different type

            # Try calling scheme.CalculateSystemContributions
            scheme.CalculateSystemContributions(elem, lhs, rhs, eq_ids, pi)

            nan_lhs = mat_nan(lhs)
            nan_rhs = vec_nan(rhs)
            rhs_n = vec_norm(rhs) if rhs.Size() > 0 else 0
            traced = " *** TRACED ***" if eid == 18 else ""
            print(f"OK K:{lhs.Size1()}x{lhs.Size2()}, |f|={rhs_n:.4e}, "
                  f"K_NaN={'YES!' if nan_lhs else 'no'}, "
                  f"f_NaN={'YES!' if nan_rhs else 'no'}{traced}")
        except TypeError as e:
            print(f"TYPE ERROR: {e}")
            # Try with different argument types
            try:
                print(f"    Retrying with list for eq_ids...", end=" "); flush()
                eq_ids_list = []
                scheme.CalculateSystemContributions(elem, lhs, rhs, eq_ids_list, pi)
                print(f"OK (list worked)")
            except Exception as e2:
                print(f"ALSO FAILED: {e2}")
        except Exception as e:
            print(f"EXCEPTION: {e}")
        flush()

    # ==========================================
    # TEST 3: Scheme CalculateSystemContributions for CONDITIONS
    # ==========================================
    print("\n" + "="*60)
    print("[TEST 3] SCHEME CalculateSystemContributions - CONDITIONS")
    print("="*60)
    flush()

    for cond in adj_mp.Conditions:
        cid = cond.Id
        print(f"  Scheme+Condition {cid}...", end=" "); flush()
        try:
            lhs = KratosMultiphysics.Matrix()
            rhs = KratosMultiphysics.Vector()
            eq_ids = KratosMultiphysics.Vector()
            scheme.CalculateSystemContributions(cond, lhs, rhs, eq_ids, pi)
            nan_lhs = mat_nan(lhs)
            nan_rhs = vec_nan(rhs)
            rhs_n = vec_norm(rhs) if rhs.Size() > 0 else 0
            print(f"OK K:{lhs.Size1()}x{lhs.Size2()}, |f|={rhs_n:.4e}, "
                  f"K_NaN={'YES!' if nan_lhs else 'no'}, "
                  f"f_NaN={'YES!' if nan_rhs else 'no'}")
        except Exception as e:
            print(f"EXCEPTION: {e}")
        flush()

    # ==========================================
    # TEST 4: Scheme InitializeNonLinIteration
    # ==========================================
    print("\n" + "="*60)
    print("[TEST 4] Scheme InitializeNonLinIteration")
    print("  (Called by strategy before Build)")
    print("="*60)
    flush()

    try:
        print("  Getting system matrix...", end=" "); flush()
        A = strategy.GetSystemMatrix()
        print(f"OK size={A.Size1()}x{A.Size2()}")
        flush()
    except Exception as e:
        print(f"EXCEPTION: {e}")
        A = None
    flush()

    # ==========================================
    # TEST 5: Check strategy
    # ==========================================
    print("\n" + "="*60)
    print("[TEST 5] Strategy Check")
    print("="*60)
    flush()

    try:
        print("  strategy.Check()...", end=" "); flush()
        strategy.Check()
        print("OK")
    except Exception as e:
        print(f"EXCEPTION: {e}")
    flush()

    # ==========================================
    # TEST 6: Predict
    # ==========================================
    print("\n" + "="*60)
    print("[TEST 6] Predict")
    print("="*60)
    flush()

    print("  Predict...", end=" "); flush()
    adj._GetSolver().Predict()
    print("OK")
    flush()

    # ==========================================
    # TEST 7: SolveSolutionStep (THE CRASH POINT)
    # ==========================================
    print("\n" + "="*60)
    print("[TEST 7] SolveSolutionStep")
    print("  Previous tests all passed.")
    print("  If crash here, it's in builder.BuildAndSolve or linear solver.")
    print("="*60)
    flush()

    print("  Calling SolveSolutionStep...", end=" "); flush()
    converged = adj._GetSolver().SolveSolutionStep()
    print(f"OK converged={converged}")
    flush()

    # ==========================================
    # RESULTS
    # ==========================================
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for elem in adj_mp.Elements:
        val = elem.GetValue(SMA.I22_SENSITIVITY)
        print(f"  Element {elem.Id:2d}: I22_SENS = {val:+.10e}")

    adj.FinalizeSolutionStep()
    adj.OutputSolutionStep()
    adj.Finalize()
    print("\nALL COMPLETED!")

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