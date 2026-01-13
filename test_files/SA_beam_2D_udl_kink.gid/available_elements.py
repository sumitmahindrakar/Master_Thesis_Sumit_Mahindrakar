# Save as: find_beam_elements.py
# Run: python find_beam_elements.py

import KratosMultiphysics
from KratosMultiphysics import StructuralMechanicsApplication

model = KratosMultiphysics.Model()
mp = model.CreateModelPart("Test")
mp.AddNodalSolutionStepVariable(KratosMultiphysics.DISPLACEMENT)
mp.AddNodalSolutionStepVariable(KratosMultiphysics.ROTATION)
mp.CreateNewNode(1, 0.0, 0.0, 0.0)
mp.CreateNewNode(2, 1.0, 0.0, 0.0)
props = mp.CreateNewProperties(1)

candidates = [
    "CrBeamElement2D2N",
    "CrBeamElementLinear2D2N",
    "CrLinearBeamElement2D2N",
    "LinearBeamElement2D2N",
    "BeamElement2D2N",
]

print("Available 2D beam elements:")
for name in candidates:
    try:
        mp.CreateNewElement(name, 1, [1, 2], props)
        print(f"  ✓ {name}")
        mp.RemoveElement(1)
    except:
        pass