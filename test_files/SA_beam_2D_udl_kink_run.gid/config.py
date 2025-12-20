"""
MASTER CONFIGURATION FILE
=========================
Change parameters HERE ONLY - everything else is auto-generated
"""

# ============================================================
# BEAM GEOMETRY
# ============================================================
BEAM_LENGTH = 2.0           # Total beam length [m]
NUM_ELEMENTS = 20            # Number of elements (creates hinge at element boundary)

# ============================================================
# SENSITIVITY ANALYSIS LOCATION
# ============================================================
# Location where you want dM/dEI (must be at an element boundary)
# For NUM_ELEMENTS=2: valid positions are 0.0, 1.0, 2.0
# For NUM_ELEMENTS=4: valid positions are 0.0, 0.5, 1.0, 1.5, 2.0
SENSITIVITY_LOCATION_X = 1.0  # x-coordinate for moment sensitivity

# ============================================================
# MATERIAL PROPERTIES
# ============================================================
YOUNGS_MODULUS = 210e9      # E [Pa]
POISSON_RATIO = 0.3         # ν [-]
DENSITY = 7850.0            # ρ [kg/m³]

# ============================================================
# CROSS-SECTION (Rectangular)
# ============================================================
SECTION_HEIGHT = 0        # h [m]
SECTION_WIDTH = 0        # b [m]

# ============================================================
# LOADING
# ============================================================
UDL_VALUE = -10000.0         # Uniform distributed load [N/m] (negative = downward)

# ============================================================
# OUTPUT DIRECTORY
# ============================================================
OUTPUT_BASE_DIR = "sensitivity_analysis"

# ============================================================
# AUTO-COMPUTED VALUES (DO NOT MODIFY)
# ============================================================
def compute_derived():
    """Compute derived quantities from primary parameters."""
    element_length = BEAM_LENGTH / NUM_ELEMENTS
    
    # Find hinge node numbers based on sensitivity location
    node_at_x = int(SENSITIVITY_LOCATION_X / element_length) + 1
    
    # Hinge is between two coincident nodes
    hinge_left_node = 2 * node_at_x      # Left side of hinge
    hinge_right_node = 2 * node_at_x + 1  # Right side of hinge
    
    # Total nodes = (NUM_ELEMENTS + 1) + NUM_INTERNAL_HINGES
    # For single hinge at sensitivity location:
    num_nodes = NUM_ELEMENTS + 1 + 1  # +1 for the duplicated hinge node
    
    # Section properties
    # area = SECTION_WIDTH * SECTION_HEIGHT
    # inertia = SECTION_WIDTH * SECTION_HEIGHT**3 / 12
    area = 0.00287
    inertia = 5e-6
    
    return {
        'element_length': element_length,
        'hinge_left_node': hinge_left_node,
        'hinge_right_node': hinge_right_node,
        'num_nodes': num_nodes,
        'area': area,
        'inertia': inertia,
        'node_at_x': node_at_x
    }