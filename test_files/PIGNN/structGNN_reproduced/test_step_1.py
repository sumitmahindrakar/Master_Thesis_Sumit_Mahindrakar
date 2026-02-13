"""
TEST STEP 1: Verify folder structure is created correctly
This script checks for folders relative to where THIS script is located.
"""
import os
import sys

print("=" * 60)
print("TEST STEP 1: Folder Structure Check")
print("=" * 60)

# Get the directory where THIS script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

print(f"\nScript location: {script_dir}")
print(f"Current working directory: {os.getcwd()}")
print(f"\nNote: Folders will be checked in the SCRIPT location.")

# Change to script directory for checking
os.chdir(script_dir)
print(f"Changed to: {os.getcwd()}")

# Required folders
required_folders = ['Data', 'GNN', 'Utils', 'Results']

print("\n" + "-" * 60)
print("Checking folders:")
print("-" * 60)

all_passed = True
missing_folders = []

for folder in required_folders:
    folder_path = os.path.join(script_dir, folder)
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        print(f"  ✓ {folder}/ exists at {folder_path}")
    else:
        print(f"  ✗ {folder}/ MISSING!")
        print(f"      Expected at: {folder_path}")
        all_passed = False
        missing_folders.append(folder)

print("\n" + "=" * 60)
if all_passed:
    print("STEP 1 PASSED! ✓")
    print("=" * 60)
    print("\nYour folder structure is correct:")
    print(f"""
    {script_dir}
    ├── Data/
    ├── GNN/
    ├── Utils/
    ├── Results/
    └── test_step1.py
    """)
    print("You can proceed to Step 2.")
else:
    print("STEP 1 FAILED! ✗")
    print("=" * 60)
    print(f"\nMissing folders: {missing_folders}")
    print(f"\nPlease create these folders inside:")
    print(f"  {script_dir}")
    print("\nOption 1 - Use PowerShell:")
    print(f'  cd "{script_dir}"')
    print(f'  mkdir {", ".join(missing_folders)}')
    print("\nOption 2 - Use File Explorer:")
    print(f"  Navigate to: {script_dir}")
    print(f"  Create folders: {', '.join(missing_folders)}")
    print("\nThen run this test again.")
print("=" * 60)