import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
from KratosMultiphysics import python_linear_solver_factory as linear_solver_factory

# --------------------------------------------------------
# Model + ModelPart
# --------------------------------------------------------
model = KM.Model()
mp = model.CreateModelPart("beam")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2

mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
mp.AddNodalSolutionStepVariable(KM.ROTATION)

# --------------------------------------------------------
# Add DOFs (VERY IMPORTANT)
# --------------------------------------------------------
# KM.VariableUtils().AddDofsList([
#     KM.DISPLACEMENT_X,
#     KM.DISPLACEMENT_Y,
#     KM.ROTATION
# ], mp)


# dofs_list = ["DISPLACEMENT_X", "DISPLACEMENT_Y", "ROTATION"]
# KM.VariableUtils().AddDofsList(dofs_list, mp)

# Use AddDof directly for each node
for node in mp.Nodes:
    node.AddDof(KM.DISPLACEMENT_X)
    node.AddDof(KM.DISPLACEMENT_Y)
    node.AddDof(KM.ROTATION)

# --------------------------------------------------------
# Geometry (2 nodes, 1 element)
# --------------------------------------------------------
L = 2.0
mp.CreateNewNode(1, 0.0, 0.0, 0.0)
mp.CreateNewNode(2, L, 0.0, 0.0)

# --------------------------------------------------------
# Material properties
# --------------------------------------------------------
props = mp.CreateNewProperties(1)
props.SetValue(KM.YOUNG_MODULUS, 210e9)
props.SetValue(SMA.CROSS_AREA, 0.01)
# props.SetValue(SMA.SECOND_MOMENT_OF_INERTIA, 8.3e-6)
props.SetValue(KM.DENSITY, 7850)
props.SetValue(SMA.I33, 8.3e-6)

# --------------------------------------------------------
# Element
# --------------------------------------------------------
mp.CreateNewElement(
    "CrBeamElement2D2N",
    1,
    [1, 2],
    props
)

# --------------------------------------------------------
# Boundary Conditions
# --------------------------------------------------------

# Left support: pinned (fix X, Y)
# mp.GetNode(1).Fix(KM.DISPLACEMENT_X)
# mp.GetNode(1).Fix(KM.DISPLACEMENT_Y)
# mp.GetNode(1).Fix(SMA.ROTATION)

# Right support: roller (fix Y)
# mp.GetNode(2).Fix(KM.DISPLACEMENT_Y)
# mp.GetNode(2).Fix(SMA.ROTATION)

# --------------------------------------------------------
# Apply UDL as line load
# --------------------------------------------------------
q = -5000.0  # N/m downward

for elem in mp.Elements:
    elem.SetValue(SMA.LINE_LOAD, KM.Array3([0.0, q, 0.0]))

# --------------------------------------------------------
# Solver Settings
# --------------------------------------------------------
# solver_settings = KM.Parameters("""
# {
#     "solver_type": "Static",
#     "model_part_name": "beam",
#     "domain_size": 2,
#     "echo_level": 1
# }
# """)

# from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_solver import CreateSolver
# solver = CreateSolver(mp, solver_settings)

# solver.Initialize()

# # --------------------------------------------------------
# # Solve
# # --------------------------------------------------------
# solver.Solve()

scheme = SMA.StructuralMechanicsStaticScheme(KM.Parameters("""{}"""))
linear_solver = linear_solver_factory.CreateFastestAvailableDirectLinearSolver()
convergence_criteria = SMA.ResidualDisplacementAndOtherDoFCriteria(1e-6, 1e-9)
builder_and_solver = KM.ResidualBasedBlockBuilderAndSolver(linear_solver)

max_iterations = 10
compute_reactions = False
reform_dofs_at_each_step = False
move_mesh_flag = False

strategy = KM.ResidualBasedNewtonRaphsonStrategy(model["beam"], scheme, convergence_criteria, builder_and_solver, max_iterations, compute_reactions, reform_dofs_at_each_step, move_mesh_flag)
strategy.Solve()

# --------------------------------------------------------
# Post-process results
# --------------------------------------------------------
print("\nNodal Displacements:")
for node in mp.Nodes:
    ux = node.GetSolutionStepValue(KM.DISPLACEMENT_X)
    uy = node.GetSolutionStepValue(KM.DISPLACEMENT_Y)
    rot = node.GetSolutionStepValue(KM.ROTATION)
    print(f"Node {node.Id}: UX={ux:.6e}, UY={uy:.6e}, ROT={rot:.6e}")
