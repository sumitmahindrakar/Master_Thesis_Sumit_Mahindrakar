import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication as SMA

# ============================================
# Try different import options based on version
# ============================================

# Option 1: Newer versions (9.x+)
try:
    from KratosMultiphysics.StructuralMechanicsApplication.adjoint_structural_mechanics_analysis import AdjointStructuralMechanicsAnalysis
    print("✅ Option 1 works: adjoint_structural_mechanics_analysis")
except ImportError as e:
    print(f"❌ Option 1 failed: {e}")

# Option 2: Using response function factory
try:
    from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis
    from KratosMultiphysics.response_functions.response_function_interface import ResponseFunctionInterface
    print("✅ Option 2 works: response_function_interface")
except ImportError as e:
    print(f"❌ Option 2 failed: {e}")

# Option 3: Older versions - direct solver approach
try:
    from KratosMultiphysics.StructuralMechanicsApplication import python_solvers_wrapper_structural
    print("✅ Option 3 works: python_solvers_wrapper_structural")
except ImportError as e:
    print(f"❌ Option 3 failed: {e}")

# Option 4: Check for adjoint solver directly
try:
    from KratosMultiphysics.StructuralMechanicsApplication import adjoint_structural_mechanics_solver
    print("✅ Option 4 works: adjoint_structural_mechanics_solver")
except ImportError as e:
    print(f"❌ Option 4 failed: {e}")