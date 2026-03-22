import os, sys, math, subprocess, tempfile

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

def main():
    # ==========================================
    # PRIMAL
    # ==========================================
    print("="*60)
    print("PRIMAL SOLVE")
    print("="*60); flush()

    with open("beam_test_parameters.json", 'r') as f:
        pp = KratosMultiphysics.Parameters(f.read())
    other = pp["processes"]["list_other_processes"]
    clean = KratosMultiphysics.Parameters("[]")
    for i in range(other.size()):
        if "eigenvalues" not in other[i]["python_module"].GetString():
            clean.Append(other[i])
    pp["processes"]["list_other_processes"] = clean

    model_p = KratosMultiphysics.Model()
    pa = structural_mechanics_analysis.StructuralMechanicsAnalysis(model_p, pp)
    pa.Initialize()
    transfer_props(model_p.GetModelPart("Structure"))
    pa.RunSolutionLoop()

    # Save primal rotations for comparison
    primal_mp = model_p.GetModelPart("Structure")
    primal_rot = {}
    primal_disp = {}
    for node in primal_mp.Nodes:
        r = node.GetSolutionStepValue(KratosMultiphysics.ROTATION)
        d = node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT)
        primal_rot[node.Id] = [r[0], r[1], r[2]]
        primal_disp[node.Id] = [d[0], d[1], d[2]]

    # Primal moments for comparison
    pi_p = primal_mp.ProcessInfo
    primal_moments = {}
    for elem in primal_mp.Elements:
        m = elem.CalculateOnIntegrationPoints(KratosMultiphysics.MOMENT, pi_p)
        primal_moments[elem.Id] = [[m[g][i] for i in range(3)] for g in range(len(m))]

    pa.Finalize()
    print("Primal done.\n"); flush()

    # ==========================================
    # ADJOINT SETUP
    # ==========================================
    print("="*60)
    print("ADJOINT SETUP + DIAGNOSTICS")
    print("="*60); flush()

    with open("beam_test_local_stress_adjoint_parameters.json", 'r') as f:
        ap = KratosMultiphysics.Parameters(f.read())

    model_a = KratosMultiphysics.Model()
    adj = structural_mechanics_analysis.StructuralMechanicsAnalysis(model_a, ap)
    adj.Initialize()
    adj_mp = model_a.GetModelPart("Structure")
    transfer_props(adj_mp)

    adj.time = adj._GetSolver().AdvanceInTime(adj.time)
    adj.InitializeSolutionStep()
    pi = adj_mp.ProcessInfo

    # ==========================================
    # CHECK 1: ROTATION loaded from HDF5?
    # ==========================================
    print("\n[CHECK 1] ROTATION values (adjoint vs primal):")
    rot_ok = True
    for node in adj_mp.Nodes:
        r = node.GetSolutionStepValue(KratosMultiphysics.ROTATION)
        p = primal_rot[node.Id]
        match = all(abs(r[i] - p[i]) < 1e-10 for i in range(3))
        if not match:
            rot_ok = False
        if abs(p[1]) > 1e-10:  # Only print nodes with non-zero primal rotation
            status = "OK" if match else "*** MISMATCH ***"
            print(f"  Node {node.Id:2d}: adj_ROT_Y={r[1]:+.6e}  primal_ROT_Y={p[1]:+.6e}  [{status}]")
    print(f"  ROTATION loaded correctly: {rot_ok}")
    flush()

    # ==========================================
    # CHECK 2: Which variable does the element read?
    # ==========================================
    print("\n[CHECK 2] Variable read test on element 18:")
    elem18 = adj_mp.Elements[18]
    geom = elem18.GetGeometry()
    n_first = geom[0]  # Node 18
    n_second = geom[1]  # Node 14

    print(f"  Element 18: node {n_first.Id} -> node {n_second.Id}")

    # Current state - moments should be non-zero if element reads DISPLACEMENT
    print("\n  [2a] Current state (DISP from HDF5, ADJ_DISP = 0):")
    m0 = elem18.CalculateOnIntegrationPoints(KratosMultiphysics.MOMENT, pi)
    for g, m in enumerate(m0):
        print(f"    GP {g}: MY={m[1]:+.10e}")
    flush()

    # Perturb DISPLACEMENT_Y by large amount
    print("\n  [2b] After perturbing DISPLACEMENT_Y of node {n_first.Id} by +10.0:")
    orig_dy = n_first.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y)
    n_first.SetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y, orig_dy + 10.0)
    m1 = elem18.CalculateOnIntegrationPoints(KratosMultiphysics.MOMENT, pi)
    for g, m in enumerate(m1):
        print(f"    GP {g}: MY={m[1]:+.10e}")
    n_first.SetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y, orig_dy)
    flush()

    # Perturb ADJOINT_DISPLACEMENT_Y by large amount
    print(f"\n  [2c] After perturbing ADJOINT_DISPLACEMENT_Y of node {n_first.Id} by +10.0:")
    try:
        orig_adj_dy = n_first.GetSolutionStepValue(SMA.ADJOINT_DISPLACEMENT_Y)
        n_first.SetSolutionStepValue(SMA.ADJOINT_DISPLACEMENT_Y, orig_adj_dy + 10.0)
        m2 = elem18.CalculateOnIntegrationPoints(KratosMultiphysics.MOMENT, pi)
        for g, m in enumerate(m2):
            print(f"    GP {g}: MY={m[1]:+.10e}")
        n_first.SetSolutionStepValue(SMA.ADJOINT_DISPLACEMENT_Y, orig_adj_dy)
    except Exception as e:
        print(f"    ERROR: {e}")
    flush()

    # Perturb DISPLACEMENT_Z (bending direction for horizontal beam)
    print(f"\n  [2d] After perturbing DISPLACEMENT_Z of node {n_first.Id} by +10.0:")
    orig_dz = n_first.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Z)
    n_first.SetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Z, orig_dz + 10.0)
    m3 = elem18.CalculateOnIntegrationPoints(KratosMultiphysics.MOMENT, pi)
    for g, m in enumerate(m3):
        print(f"    GP {g}: MY={m[1]:+.10e}")
    n_first.SetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Z, orig_dz)
    flush()

    # Perturb ADJOINT_DISPLACEMENT_Z
    print(f"\n  [2e] After perturbing ADJOINT_DISPLACEMENT_Z of node {n_first.Id} by +10.0:")
    try:
        orig_adj_dz = n_first.GetSolutionStepValue(SMA.ADJOINT_DISPLACEMENT_Z)
        n_first.SetSolutionStepValue(SMA.ADJOINT_DISPLACEMENT_Z, orig_adj_dz + 10.0)
        m4 = elem18.CalculateOnIntegrationPoints(KratosMultiphysics.MOMENT, pi)
        for g, m in enumerate(m4):
            print(f"    GP {g}: MY={m[1]:+.10e}")
        n_first.SetSolutionStepValue(SMA.ADJOINT_DISPLACEMENT_Z, orig_adj_dz)
    except Exception as e:
        print(f"    ERROR: {e}")
    flush()

    # Perturb ROTATION_Y
    print(f"\n  [2f] After perturbing ROTATION_Y of node {n_first.Id} by +10.0:")
    orig_ry = n_first.GetSolutionStepValue(KratosMultiphysics.ROTATION_Y)
    n_first.SetSolutionStepValue(KratosMultiphysics.ROTATION_Y, orig_ry + 10.0)
    m5 = elem18.CalculateOnIntegrationPoints(KratosMultiphysics.MOMENT, pi)
    for g, m in enumerate(m5):
        print(f"    GP {g}: MY={m[1]:+.10e}")
    n_first.SetSolutionStepValue(KratosMultiphysics.ROTATION_Y, orig_ry)
    flush()

    # Perturb ADJOINT_ROTATION_Y
    print(f"\n  [2g] After perturbing ADJOINT_ROTATION_Y of node {n_first.Id} by +10.0:")
    try:
        orig_adj_ry = n_first.GetSolutionStepValue(SMA.ADJOINT_ROTATION_Y)
        n_first.SetSolutionStepValue(SMA.ADJOINT_ROTATION_Y, orig_adj_ry + 10.0)
        m6 = elem18.CalculateOnIntegrationPoints(KratosMultiphysics.MOMENT, pi)
        for g, m in enumerate(m6):
            print(f"    GP {g}: MY={m[1]:+.10e}")
        n_first.SetSolutionStepValue(SMA.ADJOINT_ROTATION_Y, orig_adj_ry)
    except Exception as e:
        print(f"    ERROR: {e}")
    flush()

    # ==========================================
    # CHECK 3: Manual moment computation
    # ==========================================
    print("\n[CHECK 3] Manual moment computation for element 18:")
    d1 = n_first.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT)
    r1 = n_first.GetSolutionStepValue(KratosMultiphysics.ROTATION)
    d2 = n_second.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT)
    r2 = n_second.GetSolutionStepValue(KratosMultiphysics.ROTATION)
    
    print(f"  Node {n_first.Id}: D=[{d1[0]:.4f}, {d1[1]:.4f}, {d1[2]:.4f}] "
          f"R=[{r1[0]:.4f}, {r1[1]:.4f}, {r1[2]:.4f}]")
    print(f"  Node {n_second.Id}: D=[{d2[0]:.4f}, {d2[1]:.4f}, {d2[2]:.4f}] "
          f"R=[{r2[0]:.4f}, {r2[1]:.4f}, {r2[2]:.4f}]")
    
    # Beam along X: bending in XZ plane -> MY
    # DOFs: [uz1, ry1, uz2, ry2]
    E_val = 1.0; I22_val = 1.0; L_val = 1.0
    EI = E_val * I22_val
    k = EI / (L_val**3)
    
    uz1, ry1, uz2, ry2 = d1[2], r1[1], d2[2], r2[1]
    
    # K * u for bending DOFs
    Fz1 = k * (12*uz1 + 6*L_val*ry1 - 12*uz2 + 6*L_val*ry2)
    MY1 = k * (6*L_val*uz1 + 4*L_val*L_val*ry1 - 6*L_val*uz2 + 2*L_val*L_val*ry2)
    Fz2 = k * (-12*uz1 - 6*L_val*ry1 + 12*uz2 - 6*L_val*ry2)
    MY2 = k * (6*L_val*uz1 + 2*L_val*L_val*ry1 - 6*L_val*uz2 + 4*L_val*L_val*ry2)
    
    print(f"\n  Manual (XZ bending): uz1={uz1:.4f}, ry1={ry1:.4f}, uz2={uz2:.4f}, ry2={ry2:.4f}")
    print(f"    MY1 = {MY1:.4f}")
    print(f"    MY2 = {MY2:.4f}")
    print(f"    Primal elem 18 MY: {primal_moments[18]}")
    flush()

    # ==========================================
    # CHECK 4: Test on vertical element (elem 1)
    # ==========================================
    print("\n[CHECK 4] Moment test on VERTICAL element 1 (along Z):")
    elem1 = adj_mp.Elements[1]
    m_e1 = elem1.CalculateOnIntegrationPoints(KratosMultiphysics.MOMENT, pi)
    print(f"  Adjoint elem 1 moments:")
    for g, m in enumerate(m_e1):
        print(f"    GP {g}: M=[{m[0]:+.6e}, {m[1]:+.6e}, {m[2]:+.6e}]")
    print(f"  Primal elem 1 moments: {primal_moments[1]}")
    flush()

    # ==========================================
    # CHECK 5: Test FORCE variable instead of MOMENT
    # ==========================================
    print("\n[CHECK 5] Using FORCE variable instead of MOMENT:")
    for eid in [1, 18]:
        elem = adj_mp.Elements[eid]
        try:
            f = elem.CalculateOnIntegrationPoints(KratosMultiphysics.FORCE, pi)
            print(f"  Element {eid} FORCE:")
            for g, fv in enumerate(f):
                print(f"    GP {g}: F=[{fv[0]:+.6e}, {fv[1]:+.6e}, {fv[2]:+.6e}]")
        except Exception as e:
            print(f"  Element {eid} FORCE: ERROR - {e}")
    flush()

    # ==========================================
    # CHECK 6: Element registered variables
    # ==========================================
    print("\n[CHECK 6] Element registered solution step variables:")
    print(f"  DISPLACEMENT registered: {adj_mp.HasNodalSolutionStepVariable(KratosMultiphysics.DISPLACEMENT)}")
    print(f"  ROTATION registered: {adj_mp.HasNodalSolutionStepVariable(KratosMultiphysics.ROTATION)}")
    print(f"  ADJOINT_DISPLACEMENT registered: {adj_mp.HasNodalSolutionStepVariable(SMA.ADJOINT_DISPLACEMENT)}")
    print(f"  ADJOINT_ROTATION registered: {adj_mp.HasNodalSolutionStepVariable(SMA.ADJOINT_ROTATION)}")
    flush()

    # ==========================================
    # SOLVE ATTEMPT with stderr redirect
    # ==========================================
    print("\n" + "="*60)
    print("SOLVE ATTEMPT (stderr redirected to crash_stderr.log)")
    print("="*60)

    # Redirect C-level stderr
    stderr_path = os.path.join(os.getcwd(), "crash_stderr.log")
    try:
        stderr_file = open(stderr_path, "w")
        old_stderr = os.dup(2)
        os.dup2(stderr_file.fileno(), 2)
        print(f"  stderr -> {stderr_path}")
    except:
        print("  WARNING: Could not redirect stderr")
        old_stderr = None
    flush()

    # Set high echo level
    try:
        strategy = adj._GetSolver()._GetSolutionStrategy()
        strategy.SetEchoLevel(3)
        builder = adj._GetSolver()._GetBuilderAndSolver()
        builder.SetEchoLevel(3)
        print("  Echo level set to 3")
    except:
        pass
    flush()

    print("  Calling Predict...", end=" "); flush()
    adj._GetSolver().Predict()
    print("OK"); flush()

    print("  Calling SolveSolutionStep...", end=" "); flush()
    converged = adj._GetSolver().SolveSolutionStep()
    print(f"OK (converged={converged})"); flush()

    # Restore stderr
    if old_stderr is not None:
        os.dup2(old_stderr, 2)
        stderr_file.close()
        with open(stderr_path, 'r') as f:
            content = f.read()
        if content.strip():
            print(f"\n  STDERR CONTENT:\n{content}")
        else:
            print("  (stderr log empty)")

    # Results
    print("\n=== RESULTS ===")
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
        import traceback; traceback.print_exc()
    finally:
        # Try to read stderr log even after crash
        try:
            stderr_path = os.path.join(os.getcwd(), "crash_stderr.log")
            if os.path.exists(stderr_path):
                with open(stderr_path, 'r') as f:
                    content = f.read()
                if content.strip():
                    print(f"\nSTDERR LOG CONTENT:\n{content}")
        except:
            pass
        print("\n--- Script finished ---")
        flush()