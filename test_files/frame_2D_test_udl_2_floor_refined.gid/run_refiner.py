"""
Simple runner script for MDPA mesh refinement
Run this file directly in VS Code (F5 or Run button)
"""

from mdpa_refiner import parse_mdpa, refine_mesh, write_mdpa

# ============== CONFIGURATION ==============
# Change these parameters as needed

INPUT_FILE = "test_files/frame_2D_test_udl_2_floor_refined.gid/frame_2D_test_udl_2_floor.mdpa"      # Input file name
OUTPUT_FILE = "test_files/frame_2D_test_udl_2_floor_refined.gid/frame_2D_test_udl_2_floor_refined.mdpa"  # Output file name
SUBDIVISIONS = 10                 # Number of subdivisions per element

# ===========================================


def main():
    print("=" * 50)
    print("MDPA Mesh Refinement Tool")
    print("=" * 50)
    
    print(f"\nInput file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Subdivisions: {SUBDIVISIONS}")
    
    # Parse input file
    print("\n[1/3] Reading input file...")
    try:
        data = parse_mdpa(INPUT_FILE)
        print(f"      ✓ Successfully parsed {INPUT_FILE}")
    except FileNotFoundError:
        print(f"      ✗ Error: File '{INPUT_FILE}' not found!")
        print(f"        Make sure the file is in the same folder as this script.")
        return
    except Exception as e:
        print(f"      ✗ Error parsing file: {e}")
        return
    
    # Print original mesh info
    print("\n[2/3] Original mesh statistics:")
    print(f"      - Nodes: {len(data.nodes)}")
    print(f"      - Elements: {len(data.elements)}")
    print(f"      - Element type: {data.element_type}")
    print(f"      - Conditions: {len(data.conditions)}")
    print(f"      - SubModelParts: {len(data.sub_model_parts)}")
    for name in data.sub_model_parts:
        smp = data.sub_model_parts[name]
        print(f"        • {name}: {len(smp.nodes)} nodes, {len(smp.elements)} elements, {len(smp.conditions)} conditions")
    
    # Refine mesh
    print(f"\n[3/3] Refining mesh with {SUBDIVISIONS} subdivisions...")
    refined = refine_mesh(data, SUBDIVISIONS)
    
    # Print refined mesh info
    print("\n      Refined mesh statistics:")
    print(f"      - Nodes: {len(data.nodes)} → {len(refined.nodes)}")
    print(f"      - Elements: {len(data.elements)} → {len(refined.elements)}")
    print(f"      - Conditions: {len(data.conditions)} → {len(refined.conditions)}")
    
    # Write output
    write_mdpa(refined, OUTPUT_FILE)
    print(f"\n      ✓ Output written to: {OUTPUT_FILE}")
    
    print("\n" + "=" * 50)
    print("Refinement completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()