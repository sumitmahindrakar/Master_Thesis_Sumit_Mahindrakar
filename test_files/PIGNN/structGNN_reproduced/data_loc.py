"""
Script to find .pt data files in your project
"""
import os

print("=" * 60)
print("SEARCHING FOR .PT DATA FILES")
print("=" * 60)

# Get the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"\nSearching from: {script_dir}")

# Go up a few directories to search more broadly
search_roots = [
    script_dir,
    os.path.dirname(script_dir),  # Parent
    os.path.dirname(os.path.dirname(script_dir)),  # Grandparent
]

pt_files_found = []

for root_dir in search_roots:
    print(f"\nSearching in: {root_dir}")
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip some directories to speed up search
        dirnames[:] = [d for d in dirnames if d not in ['.git', '__pycache__', 'venv', 'env', '.venv']]
        
        for filename in filenames:
            if filename.endswith('.pt'):
                full_path = os.path.join(dirpath, filename)
                pt_files_found.append(full_path)
                
                # Stop if we found too many
                if len(pt_files_found) > 50:
                    break
        
        if len(pt_files_found) > 50:
            break
    
    if len(pt_files_found) > 50:
        print("  (Stopped after finding 50+ files)")
        break

print("\n" + "=" * 60)
print(f"FOUND {len(pt_files_found)} .PT FILES:")
print("=" * 60)

if pt_files_found:
    # Group by directory pattern
    for pt_file in pt_files_found[:20]:  # Show first 20
        print(f"  {pt_file}")
    
    if len(pt_files_found) > 20:
        print(f"  ... and {len(pt_files_found) - 20} more")
    
    # Check if any look like structure data
    print("\n" + "-" * 60)
    print("Files that look like structure graph data:")
    print("-" * 60)
    
    structure_files = [f for f in pt_files_found if 'structure' in f.lower()]
    if structure_files:
        for f in structure_files[:10]:
            print(f"  {f}")
    else:
        print("  No files with 'structure' in the name found.")
        
    graph_files = [f for f in pt_files_found if 'graph' in f.lower()]
    if graph_files:
        print("\n  Files with 'graph' in the name:")
        for f in graph_files[:10]:
            print(f"  {f}")
else:
    print("  No .pt files found!")
    print("\n  You may need to:")
    print("  1. Download the data from the original repository")
    print("  2. Generate the data using the data generation scripts")

print("\n" + "=" * 60)