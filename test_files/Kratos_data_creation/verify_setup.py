"""
Verify generated files from parametric study setup.
Usage: python verify_setup.py --folder primal_test/case_primal_1
"""

import os
import sys
import json
import argparse


def check_mdpa(folder):
    """Verify MDPA has dual condition blocks and per-node SMPs."""
    mdpa_path = os.path.join(folder, "Frame_structure_refined.mdpa")
    if not os.path.exists(mdpa_path):
        print(f"  ✗ MDPA not found: {mdpa_path}")
        return False

    with open(mdpa_path, 'r') as f:
        content = f.read()

    checks = {
        "PointLoadCondition3D1N": "Begin Conditions PointLoadCondition3D1N" in content,
        "PointMomentCondition3D1N": "Begin Conditions PointMomentCondition3D1N" in content,
        "AutoForce SMPs": "AutoForce_node_" in content,
        "AutoMoment SMPs": "AutoMoment_node_" in content,
        "DISPLACEMENT_support": "DISPLACEMENT_support" in content,
        "ROTATION_support": "ROTATION_support" in content,
        "sensitivity_mp": "sensitivity_mp" in content,
    }

    # Count conditions and SMPs
    force_count = content.count("Begin Conditions PointLoadCondition3D1N")
    moment_count = content.count("Begin Conditions PointMomentCondition3D1N")
    auto_force = content.count("AutoForce_node_")
    auto_moment = content.count("AutoMoment_node_")

    # Count nodes
    in_nodes = False
    node_count = 0
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith("Begin Nodes"):
            in_nodes = True
            continue
        if line.startswith("End Nodes"):
            break
        if in_nodes and line and not line.startswith("//"):
            node_count += 1

    print(f"\n  MDPA: {mdpa_path}")
    print(f"    Nodes: {node_count}")
    print(f"    PointLoad blocks: {force_count}")
    print(f"    PointMoment blocks: {moment_count}")
    print(f"    AutoForce SMPs: {auto_force // 2}")
    print(f"    AutoMoment SMPs: {auto_moment // 2}")

    all_ok = True
    for name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"    {status} {name}")
        if not passed:
            all_ok = False

    return all_ok


def check_primal_json(folder):
    """Verify primal JSON has per-node load processes."""
    json_path = os.path.join(folder, "beam_test_parameters.json")
    if not os.path.exists(json_path):
        print(f"  ✗ Primal JSON not found: {json_path}")
        return False

    with open(json_path, 'r') as f:
        data = json.load(f)

    loads = data["processes"]["loads_process_list"]

    force_procs = [
        p for p in loads
        if p["Parameters"].get("variable_name") == "POINT_LOAD"
        and "AutoForce_node_" in p["Parameters"].get("model_part_name", "")
    ]
    moment_procs = [
        p for p in loads
        if p["Parameters"].get("variable_name") == "POINT_MOMENT"
        and "AutoMoment_node_" in p["Parameters"].get("model_part_name", "")
    ]
    gravity_procs = [
        p for p in loads
        if p["Parameters"].get("variable_name") == "VOLUME_ACCELERATION"
    ]

    print(f"\n  Primal JSON: {json_path}")
    print(f"    Force processes (AutoForce): {len(force_procs)}")
    print(f"    Moment processes (AutoMoment): {len(moment_procs)}")
    print(f"    Gravity processes: {len(gravity_procs)}")
    print(f"    Total load processes: {len(loads)}")

    # Check a sample force process
    if force_procs:
        sample = force_procs[0]["Parameters"]
        print(f"\n    Sample force process:")
        print(f"      model_part: {sample['model_part_name']}")
        print(f"      modulus: {sample['modulus']:.4f}")
        print(f"      direction: {sample['direction']}")

    # Check a sample moment process
    if moment_procs:
        sample = moment_procs[0]["Parameters"]
        print(f"\n    Sample moment process:")
        print(f"      model_part: {sample['model_part_name']}")
        print(f"      modulus: {sample['modulus']:.4f}")
        print(f"      direction: {sample['direction']}")

    # Count non-zero loads
    nonzero_force = sum(
        1 for p in force_procs
        if p["Parameters"]["modulus"] > 1e-15
    )
    nonzero_moment = sum(
        1 for p in moment_procs
        if p["Parameters"]["modulus"] > 1e-15
    )
    print(f"\n    Non-zero forces: {nonzero_force}/{len(force_procs)}")
    print(f"    Non-zero moments: {nonzero_moment}/{len(moment_procs)}")

    checks = {
        "Has force processes": len(force_procs) > 0,
        "Has moment processes": len(moment_procs) > 0,
        "Force == Moment count": len(force_procs) == len(moment_procs),
        "Gravity preserved": len(gravity_procs) == 1,
        "No old PointLoad3D_load": not any(
            "PointLoad3D_load" in p["Parameters"].get("model_part_name", "")
            for p in loads
            if p["Parameters"].get("variable_name") == "POINT_LOAD"
        ),
    }

    all_ok = True
    for name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"    {status} {name}")
        if not passed:
            all_ok = False

    return all_ok


def check_adjoint_json(folder):
    """Verify adjoint JSON matches primal loads."""
    json_path = os.path.join(
        folder, "beam_test_local_stress_adjoint_parameters.json"
    )
    if not os.path.exists(json_path):
        print(f"  ✗ Adjoint JSON not found: {json_path}")
        return False

    with open(json_path, 'r') as f:
        data = json.load(f)

    loads = data["processes"]["loads_process_list"]
    resp = data["solver_settings"]["response_function_settings"]

    force_procs = [
        p for p in loads
        if p["Parameters"].get("variable_name") == "POINT_LOAD"
        and "AutoForce_node_" in p["Parameters"].get("model_part_name", "")
    ]
    moment_procs = [
        p for p in loads
        if p["Parameters"].get("variable_name") == "POINT_MOMENT"
        and "AutoMoment_node_" in p["Parameters"].get("model_part_name", "")
    ]

    print(f"\n  Adjoint JSON: {json_path}")
    print(f"    Force processes: {len(force_procs)}")
    print(f"    Moment processes: {len(moment_procs)}")
    print(f"    traced_element_id: {resp['traced_element_id']}")
    print(f"    stress_type: {resp['stress_type']}")
    print(f"    stress_location: {resp['stress_location']}")

    checks = {
        "Has force processes": len(force_procs) > 0,
        "Has moment processes": len(moment_procs) > 0,
        "traced_element_id > 0": resp["traced_element_id"] > 0,
    }

    all_ok = True
    for name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"    {status} {name}")
        if not passed:
            all_ok = False

    return all_ok


def check_case_config(folder):
    """Verify case_config.json has per-node loads."""
    config_path = os.path.join(folder, "case_config.json")
    if not os.path.exists(config_path):
        print(f"  ✗ case_config.json not found")
        return False

    with open(config_path, 'r') as f:
        data = json.load(f)

    params = data["parameters"]
    summary = data.get("load_summary", {})

    print(f"\n  Case Config: {config_path}")
    print(f"    case_id: {data['case_id']}")
    print(f"    subdivisions: {params['subdivisions']}")
    print(f"    traced_element_id: {params['traced_element_id']}")
    print(f"    response_coords: {params['response_coords']}")

    if "node_loads" in params:
        n_loads = len(params["node_loads"])
        print(f"    node_loads entries: {n_loads}")
        print(f"    loaded_nodes: {summary.get('loaded_nodes', '?')}"
              f"/{summary.get('total_nodes', '?')}")
        print(f"    Fx range: {summary.get('Fx_range', '?')}")
        print(f"    Fz range: {summary.get('Fz_range', '?')}")
        print(f"    My range: {summary.get('My_range', '?')}")

        # Show first 3
        for i, (nid, vals) in enumerate(params["node_loads"].items()):
            if i >= 3:
                print(f"      ... ({n_loads - 3} more)")
                break
            print(f"      Node {nid}: Fx={vals['Fx']:.2f} "
                  f"Fz={vals['Fz']:.2f} My={vals['My']:.2f}")
    else:
        print(f"    ✗ No node_loads found!")
        return False

    return True


def check_materials(folder):
    """Verify materials JSON."""
    mat_path = os.path.join(folder, "materials_beam.json")
    if not os.path.exists(mat_path):
        print(f"  ✗ materials_beam.json not found")
        return False

    with open(mat_path, 'r') as f:
        data = json.load(f)

    variables = data["properties"][0]["Material"]["Variables"]
    print(f"\n  Materials: {mat_path}")
    for key, val in variables.items():
        print(f"    {key}: {val}")

    return True


def verify_case(folder):
    """Run all checks on a case folder."""
    print(f"\n{'=' * 60}")
    print(f"VERIFYING: {folder}")
    print(f"{'=' * 60}")

    if not os.path.exists(folder):
        print(f"  ✗ Folder not found!")
        return False

    results = {
        "MDPA": check_mdpa(folder),
        "Primal JSON": check_primal_json(folder),
        "Adjoint JSON": check_adjoint_json(folder),
        "Case Config": check_case_config(folder),
        "Materials": check_materials(folder),
    }

    print(f"\n{'─' * 40}")
    print(f"  SUMMARY:")
    all_ok = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"    {status}: {name}")
        if not passed:
            all_ok = False

    overall = "ALL CHECKS PASSED ✓" if all_ok else "SOME CHECKS FAILED ✗"
    print(f"\n  {overall}")
    return all_ok


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Verify parametric study setup"
#     )
#     parser.add_argument(
#         "--folder",
#         required=True,
#         help="Path to case folder (e.g. primal_test/case_primal_1)"
#     )
#     parser.add_argument(
#         "--working-dir",
#         default=None,
#         help="Working directory (prepended to folder)"
#     )
#     args = parser.parse_args()

#     if args.working_dir:
#         folder = os.path.join(args.working_dir, args.folder)
#     else:
#         folder = args.folder

#     success = verify_case(folder)
#     sys.exit(0 if success else 1)

if __name__ == "__main__":
    # Quick test — hardcoded path
    folder = r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\Kratos_data_creation\primal_test\case_primal_1"
    
    success = verify_case(folder)
    
    # Also verify adjoint
    adjoint_folder = folder.replace("primal_test", "adjoint_test").replace("case_primal", "case_adjoint")
    if os.path.exists(adjoint_folder):
        verify_case(adjoint_folder)
    
    sys.exit(0 if success else 1)