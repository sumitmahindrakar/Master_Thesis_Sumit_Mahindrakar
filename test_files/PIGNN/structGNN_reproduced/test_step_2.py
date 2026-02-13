"""
TEST STEP 2: Verify __init__.py files exist in GNN and Utils folders
These files are required for Python to recognize folders as importable packages.
"""
import os
import sys

print("=" * 60)
print("TEST STEP 2: Python Package Files (__init__.py) Check")
print("=" * 60)

# Get the directory where THIS script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print(f"\nProject directory: {script_dir}")

# Required __init__.py files
required_init_files = [
    os.path.join('GNN', '__init__.py'),
    os.path.join('Utils', '__init__.py')
]

print("\n" + "-" * 60)
print("Checking __init__.py files:")
print("-" * 60)

all_passed = True
missing_files = []

for init_file in required_init_files:
    full_path = os.path.join(script_dir, init_file)
    if os.path.exists(full_path) and os.path.isfile(full_path):
        # Check file size
        size = os.path.getsize(full_path)
        print(f"  ✓ {init_file} exists ({size} bytes)")
    else:
        print(f"  ✗ {init_file} MISSING!")
        all_passed = False
        missing_files.append(init_file)

# Also verify the folders still exist
print("\n" + "-" * 60)
print("Verifying parent folders:")
print("-" * 60)

for folder in ['GNN', 'Utils']:
    folder_path = os.path.join(script_dir, folder)
    if os.path.exists(folder_path):
        print(f"  ✓ {folder}/ exists")
    else:
        print(f"  ✗ {folder}/ MISSING!")
        all_passed = False

# Test if Python can recognize them as packages
print("\n" + "-" * 60)
print("Testing Python package recognition:")
print("-" * 60)

# Add script directory to Python path
sys.path.insert(0, script_dir)

try:
    import GNN
    print(f"  ✓ 'import GNN' works")
    print(f"      GNN package location: {GNN.__file__}")
except ImportError as e:
    print(f"  ✗ 'import GNN' failed: {e}")
    all_passed = False

try:
    import Utils
    print(f"  ✓ 'import Utils' works")
    print(f"      Utils package location: {Utils.__file__}")
except ImportError as e:
    print(f"  ✗ 'import Utils' failed: {e}")
    all_passed = False

# Summary
print("\n" + "=" * 60)
if all_passed:
    print("STEP 2 PASSED! ✓")
    print("=" * 60)
    print("\nYour package structure is correct:")
    print(f"""
    {os.path.basename(script_dir)}/
    ├── GNN/
    │   └── __init__.py  ✓
    ├── Utils/
    │   └── __init__.py  ✓
    ├── Data/
    └── Results/
    """)
    print("Python can now import from GNN and Utils folders.")
    print("\nYou can proceed to Step 3.")
else:
    print("STEP 2 FAILED! ✗")
    print("=" * 60)
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        print("\nTo create these files:")
        print("\nOption 1 - PowerShell:")
        for f in missing_files:
            folder = os.path.dirname(f)
            print(f'  New-Item -Path "{f}" -ItemType File')
        print("\nOption 2 - Create empty text files and rename them to __init__.py")
    print("\nThen run this test again.")
print("=" * 60)