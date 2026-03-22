from mdpa_refiner import parse_mdpa, refine_mesh, write_mdpa


INPUT_FILE = "test_files/SA_Kratos_adj_V3_point_load/Beam_structure_1E.mdpa"
OUTPUT_FILE = "test_files/SA_Kratos_adj_V3_point_load/Beam_structure_refined.mdpa"
SUBDIVISIONS = 8


def main():
    print("=" * 50)
    print("MDPA Mesh Refinement Tool")
    print("=" * 50)

    print(f"\nInput file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Subdivisions: {SUBDIVISIONS}")

    print("\n[1/3] Reading input file...")
    try:
        data = parse_mdpa(INPUT_FILE)
        print(f"      Done")
    except FileNotFoundError:
        print(f"      Error: File '{INPUT_FILE}' not found!")
        return
    except Exception as e:
        print(f"      Error: {e}")
        return

    print("\n[2/3] Original mesh:")
    print(f"      Nodes: {len(data.nodes)}")
    print(f"      Elements: {len(data.elements)} ({data.element_type})")
    print(f"      Conditions: {len(data.conditions)} ({data.condition_type})")
    for name, smp in data.sub_model_parts.items():
        print(f"      SMP '{name}': {len(smp.nodes)} nodes, {len(smp.elements)} elems, {len(smp.conditions)} conds")

    print(f"\n[3/3] Refining with {SUBDIVISIONS} subdivisions...")
    refined = refine_mesh(data, SUBDIVISIONS)

    print(f"\n      Refined mesh:")
    print(f"      Nodes: {len(data.nodes)} -> {len(refined.nodes)}")
    print(f"      Elements: {len(data.elements)} -> {len(refined.elements)}")
    print(f"      Conditions: {len(data.conditions)} -> {len(refined.conditions)} ({refined.condition_type})")

    write_mdpa(refined, OUTPUT_FILE)
    print(f"\n      Output: {OUTPUT_FILE}")
    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()