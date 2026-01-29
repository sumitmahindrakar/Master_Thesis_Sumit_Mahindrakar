import KratosMultiphysics
print("Kratos Version:", KratosMultiphysics.KratosGlobals.Kernel.Version())

# Check StructuralMechanicsApplication
import KratosMultiphysics.StructuralMechanicsApplication as SMA
print("\nStructuralMechanicsApplication loaded successfully!")

# List available modules
import os
sma_path = os.path.dirname(SMA.__file__)
print(f"\nSMA Location: {sma_path}")

print("\nAvailable Python files:")
for f in os.listdir(sma_path):
    if f.endswith('.py') and 'adjoint' in f.lower():
        print(f"  - {f}")