import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
from KratosMultiphysics import python_linear_solver_factory as linear_solver_factory

# --------------------------------------------------------
# Model + ModelPart
# --------------------------------------------------------
model = KM.Model()
mp = model.CreateModelPart("beam")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
mp.ProcessInfo[KM.TIME] = 0.0
mp.ProcessInfo[KM.STEP] = 0

# Add nodal solution step variables
mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
mp.AddNodalSolutionStepVariable(KM.ROTATION_Z)
mp.AddNodalSolutionStepVariable(KM.REACTION)

# --------------------------------------------------------
# Geometry (2 nodes, 1 element)
# --------------------------------------------------------
L = 2.0
mp.CreateNewNode(1, 0.0, 0.0, 0.0)
mp.CreateNewNode(2, L, 0.0, 0.0)

# --------------------------------------------------------
# Add DOFs manually (required in Kratos 10.3)
# --------------------------------------------------------
for node in mp.Nodes:
    node.AddDof(KM.DISPLACEMENT_X)  # scalar
    node.AddDof(KM.DISPLACEMENT_Y)  # scalar
    node.AddDof(KM.ROTATION_Z)        # scalar rotation about Z

# --------------------------------------------------------
# Material properties
# --------------------------------------------------------
props = mp.CreateNewProperties(1)
props.SetValue(KM.YOUNG_MODULUS, 210e9)
props.SetValue(SMA.CROSS_AREA, 0.01)
props.SetValue(KM.DENSITY, 7850)
props.SetValue(SMA.I33, 8.3e-6)  # 2D beam bending about Z-axis

# --------------------------------------------------------
# Element
# --------------------------------------------------------
mp.CreateNewElement("CrBeamElement2D2N", 1, [1, 2], props)

# --------------------------------------------------------
# Boundary Conditions
# --------------------------------------------------------
# Node 1: pinned (UX, UY, rotation)
mp.GetNode(1).Fix(KM.DISPLACEMENT_X)
mp.GetNode(1).Fix(KM.DISPLACEMENT_Y)
# mp.GetNode(1).Fix(KM.ROTATION_Z)

# Node 2: roller (UY, rotation free in X)
mp.GetNode(2).Fix(KM.DISPLACEMENT_Y)
# mp.GetNode(2).Fix(KM.ROTATION_Z)

# --------------------------------------------------------
# Apply UDL as line load
# --------------------------------------------------------
q = -5000.0  # N/m downward
for elem in mp.Elements:
    elem.SetValue(SMA.LINE_LOAD, KM.Array3([0.0, q, 0.0]))

# --------------------------------------------------------
# Solver setup
# --------------------------------------------------------
scheme = SMA.StructuralMechanicsStaticScheme(KM.Parameters("""{}"""))
linear_solver = linear_solver_factory.CreateFastestAvailableDirectLinearSolver()
convergence_criteria = SMA.ResidualDisplacementAndOtherDoFCriteria(1e-6, 1e-9)
builder_and_solver = KM.ResidualBasedBlockBuilderAndSolver(linear_solver)

max_iterations = 10
compute_reactions = True
reform_dofs_at_each_step = False
move_mesh_flag = False

strategy = KM.ResidualBasedNewtonRaphsonStrategy(
    mp,
    scheme,
    convergence_criteria,
    builder_and_solver,
    max_iterations,
    compute_reactions,
    reform_dofs_at_each_step,
    move_mesh_flag
)

# MUST initialize the strategy
strategy.Initialize()

# --------------------------------------------------------
# Solve
# --------------------------------------------------------
strategy.Solve()

# --------------------------------------------------------
# Post-process results
# --------------------------------------------------------
print("\nNodal Displacements and Rotations:")
for node in mp.Nodes:
    ux = node.GetSolutionStepValue(KM.DISPLACEMENT_X)
    uy = node.GetSolutionStepValue(KM.DISPLACEMENT_Y)
    rot = node.GetSolutionStepValue(KM.ROTATION_Z)
    print(f"Node {node.Id}: UX={ux:.6e}, UY={uy:.6e}, ROT={rot:.6e}")

print("######################### end")
# print(list(KM.KratosComponents("DoubleVariable").keys())[:100])
