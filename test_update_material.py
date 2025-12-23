"""
Test script to verify material update works
"""

import json
import os
import sys

# Add scripts directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'scripts'))

from utils.config_loader import load_config

def test_update():
    """Test updating material file."""
    
    print("=" * 60)
    print("TESTING MATERIAL UPDATE")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    print(f"\n1. Config values:")
    print(f"   E = {config.material.E}")
    print(f"   I = {config.material.I}")
    
    # Get material path
    material_path = config.paths.input_materials
    print(f"\n2. Material file path:")
    print(f"   {material_path}")
    print(f"   Exists: {os.path.exists(material_path)}")
    
    # Read current values
    print(f"\n3. Current values in file:")
    with open(material_path, 'r') as f:
        data = json.load(f)
    
    variables = data['properties'][0]['Material']['Variables']
    print(f"   YOUNG_MODULUS = {variables['YOUNG_MODULUS']}")
    print(f"   I33 = {variables.get('I33')}")
    
    # Update if config has values
    if config.material.E is not None:
        print(f"\n4. Updating E to {config.material.E}...")
        variables['YOUNG_MODULUS'] = config.material.E
        
        with open(material_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        print("   Done!")
        
        # Verify
        print(f"\n5. Verifying update:")
        with open(material_path, 'r') as f:
            data_check = json.load(f)
        
        new_E = data_check['properties'][0]['Material']['Variables']['YOUNG_MODULUS']
        print(f"   YOUNG_MODULUS = {new_E}")
        
        if new_E == config.material.E:
            print("\n   ✅ SUCCESS: E was updated correctly!")
        else:
            print("\n   ❌ FAILED: E was not updated!")
    else:
        print("\n4. config.material.E is None - nothing to update")
        print("   Set E in config.yaml and try again")


if __name__ == "__main__":
    test_update()