\nNodal Displacements and Rotations:")
for node in mp.Nodes:
    ux = node.GetSolutionStepValue(KM.DISPLACEMENT_X)
    uy = node.GetSolutionStepValue(KM.DISPLACEMENT_Y)
    rot = node.GetSolutionStepValue(KM.ROTATION)
    print(f"