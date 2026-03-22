"""
VTK Result Plotter for Kratos Beam Structures
==============================================

Plots:
  - Deformed/undeformed structure
  - Applied forces (POINT_LOAD arrows)
  - Displacement, Rotation, Reactions
  - Internal forces (MOMENT, FORCE)
  - Material properties

Usage:
  Edit CASE_FOLDER and PLOT_WHAT below, then run.
"""

import os
import sys
import glob
import numpy as np

try:
    import pyvista as pv
    print("✓ pyvista imported")
except ImportError:
    print("✗ pyvista not installed. Run: pip install pyvista")
    sys.exit(1)


# ============================================================
# CONFIGURATION — EDIT HERE
# ============================================================

CASE_FOLDER = r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\Kratos_data_creation\primal_Fx\case_primal_1"

VTK_SUBFOLDER = "vtk_output_primal"

# What to plot — set True/False
PLOT_WHAT = {
    "structure":        True,    # beam elements
    "point_loads":      True,    # applied force arrows
    "displacement":     True,    # deformed shape
    "reactions":        False,   # reaction forces
    "reaction_moments": False,   # reaction moments
    "rotation":         False,   # rotation field
    "internal_moment":  False,   # MOMENT gauss point
    "internal_force":   False,   # FORCE gauss point
    "young_modulus":    False,   # E per element
    "cross_area":       False,   # A per element
    "I22":              False,   # I22 per element
    "point_moment":     False,   # applied moments
}

# Visual settings
DEFORMATION_SCALE = 10.0       # amplify displacements
FORCE_ARROW_SCALE = 0.02       # arrow length per unit force
MOMENT_ARROW_SCALE = 0.05      # arrow length per unit moment
WINDOW_SIZE = [1600, 900]
BACKGROUND_COLOR = "white"
STRUCTURE_COLOR = "black"
DEFORMED_COLOR = "blue"
SUPPORT_COLOR = "red"
FORCE_COLOR = "red"
MOMENT_COLOR = "green"
REACTION_COLOR = "orange"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_vtk_file(case_folder, subfolder):
    """Find the VTK file in the output folder."""
    vtk_dir = os.path.join(case_folder, subfolder)
    if not os.path.exists(vtk_dir):
        print(f"✗ VTK folder not found: {vtk_dir}")
        return None

    patterns = ["*.vtk", "*.vtu", "*.vtp"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(vtk_dir, "**", pat),
                               recursive=True))

    if not files:
        print(f"✗ No VTK files found in: {vtk_dir}")
        return None

    # Pick the latest file (highest step number)
    files.sort()
    print(f"  Found {len(files)} VTK file(s)")
    print(f"  Using: {os.path.basename(files[-1])}")
    return files[-1]


def load_vtk(vtk_path):
    """Load VTK file and print available data."""
    mesh = pv.read(vtk_path)

    print(f"\n  Mesh info:")
    print(f"    Points: {mesh.n_points}")
    print(f"    Cells:  {mesh.n_cells}")

    if mesh.point_data:
        print(f"    Point data:")
        for name in mesh.point_data:
            arr = mesh.point_data[name]
            shape = arr.shape if hasattr(arr, 'shape') else 'scalar'
            print(f"      {name}: {shape}")

    if mesh.cell_data:
        print(f"    Cell data:")
        for name in mesh.cell_data:
            arr = mesh.cell_data[name]
            shape = arr.shape if hasattr(arr, 'shape') else 'scalar'
            print(f"      {name}: {shape}")

    return mesh


def get_array_safe(mesh, name, data_type="point"):
    """Safely get array from mesh."""
    if data_type == "point" and name in mesh.point_data:
        return mesh.point_data[name]
    elif data_type == "cell" and name in mesh.cell_data:
        return mesh.cell_data[name]
    return None


def make_arrows(points, vectors, scale=1.0):
    """Create arrow glyphs from points and vectors."""
    magnitudes = np.linalg.norm(vectors, axis=1)
    mask = magnitudes > 1e-10

    if not np.any(mask):
        return None

    pts = points[mask]
    vecs = vectors[mask] * scale

    cloud = pv.PolyData(pts)
    cloud["vectors"] = vecs
    cloud.set_active_vectors("vectors")

    arrows = cloud.glyph(
        orient="vectors",
        scale=False,
        factor=1.0
    )
    return arrows


def get_support_nodes(mesh):
    """Identify support nodes (zero displacement, non-zero reaction)."""
    disp = get_array_safe(mesh, "DISPLACEMENT", "point")
    react = get_array_safe(mesh, "REACTION", "point")

    if disp is None:
        return np.array([])

    disp_mag = np.linalg.norm(disp, axis=1)
    support_mask = disp_mag < 1e-12

    if react is not None:
        react_mag = np.linalg.norm(react, axis=1)
        support_mask = support_mask & (react_mag > 1e-10)

    return np.where(support_mask)[0]


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_structure(plotter, mesh):
    """Plot undeformed structure."""
    plotter.add_mesh(
        mesh, color=STRUCTURE_COLOR,
        line_width=3, style="wireframe",
        label="Undeformed"
    )

    # Support nodes
    support_idx = get_support_nodes(mesh)
    if len(support_idx) > 0:
        support_pts = mesh.points[support_idx]
        support_cloud = pv.PolyData(support_pts)
        plotter.add_mesh(
            support_cloud, color=SUPPORT_COLOR,
            point_size=15, render_points_as_spheres=True,
            label="Supports"
        )


def plot_displacement(plotter, mesh, scale=10.0):
    """Plot deformed structure."""
    disp = get_array_safe(mesh, "DISPLACEMENT", "point")
    if disp is None:
        print("    ⚠ DISPLACEMENT not found")
        return

    deformed = mesh.copy()
    deformed.points = mesh.points + disp * scale

    disp_mag = np.linalg.norm(disp, axis=1)

    plotter.add_mesh(
        deformed, scalars=disp_mag,
        line_width=4, style="wireframe",
        cmap="jet", scalar_bar_args={"title": "Displacement [m]"},
        label=f"Deformed (×{scale})"
    )


# def plot_point_loads(plotter, mesh, scale=0.02):
#     """Plot applied point loads as arrows."""
#     loads = get_array_safe(mesh, "POINT_LOAD", "point")
#     if loads is None:
#         print("    ⚠ POINT_LOAD not found")
#         return

#     arrows = make_arrows(mesh.points, loads, scale)
#     if arrows is None:
#         print("    ⚠ No non-zero point loads")
#         return

#     load_mags = np.linalg.norm(loads, axis=1)
#     n_loaded = np.sum(load_mags > 1e-10)
#     print(f"    Point loads: {n_loaded} nodes loaded")
#     print(f"    Max force: {np.max(load_mags):.2f}")

#     plotter.add_mesh(
#         arrows, color=FORCE_COLOR,
#         label=f"Point Loads ({n_loaded} nodes)"
#     )

def plot_point_loads(plotter, mesh, scale=0.02):
    """Plot applied point loads with scaled arrows and labels."""
    loads = get_array_safe(mesh, "POINT_LOAD", "point")
    if loads is None:
        print("    ⚠ POINT_LOAD not found")
        return

    magnitudes = np.linalg.norm(loads, axis=1)
    mask = magnitudes > 1e-10
    n_loaded = np.sum(mask)

    if n_loaded == 0:
        print("    ⚠ No non-zero point loads")
        return

    pts = mesh.points[mask]
    vecs = loads[mask]
    mags = magnitudes[mask]

    print(f"    Point loads: {n_loaded} nodes loaded")
    print(f"    Min force: {np.min(mags):.2f}")
    print(f"    Max force: {np.max(mags):.2f}")
    print(f"    Mean force: {np.mean(mags):.2f}")

    # ── Method 1: Scaled arrows (length ∝ magnitude) ──
    cloud = pv.PolyData(pts)
    cloud["vectors"] = vecs * scale
    cloud["magnitude"] = mags
    cloud.set_active_vectors("vectors")

    arrows = cloud.glyph(
        orient="vectors",
        scale="magnitude",
        factor=scale
    )
    arrows["Force Magnitude [N]"] = np.linalg.norm(
        loads[mask], axis=1
    ).repeat(arrows.n_cells // n_loaded) \
        if arrows.n_cells > 0 else np.array([])

    plotter.add_mesh(
        arrows,
        scalars="Force Magnitude [N]" if "Force Magnitude [N]" in arrows.array_names else None,
        cmap="coolwarm",
        scalar_bar_args={
            "title": "Force [N]",
            "n_labels": 5,
            "position_x": 0.85,
        },
        label=f"Forces ({n_loaded} nodes)"
    )

    # ── Method 2: Color-coded dots at load points ──
    cloud_dots = pv.PolyData(pts)
    cloud_dots["Force [N]"] = mags

    plotter.add_mesh(
        cloud_dots,
        scalars="Force [N]",
        cmap="coolwarm",
        point_size=12,
        render_points_as_spheres=True,
        scalar_bar_args={
            "title": "Force [N]",
            "n_labels": 5,
        },
    )

    # ── Method 3: Value labels at top N nodes ──
    n_labels = min(10, n_loaded)  # show top 10 largest
    top_idx = np.argsort(mags)[-n_labels:]

    for i in top_idx:
        fx = vecs[i, 0]
        fz = vecs[i, 2]
        label_text = f"Fx={fx:.1f}\nFz={fz:.1f}"

        # Offset label slightly from node
        label_pos = pts[i] + np.array([0.3, 0.0, 0.3])

        plotter.add_point_labels(
            pv.PolyData(label_pos.reshape(1, 3)),
            [label_text],
            font_size=8,
            text_color="black",
            point_size=0,
            shape=None,
            show_points=False,
            always_visible=True
        )


# def plot_point_moments(plotter, mesh, scale=0.05):
#     """Plot applied moments as arrows."""
#     moments = get_array_safe(mesh, "POINT_MOMENT", "point")
#     if moments is None:
#         print("    ⚠ POINT_MOMENT not found")
#         return

#     arrows = make_arrows(mesh.points, moments, scale)
#     if arrows is None:
#         print("    ⚠ No non-zero point moments")
#         return

#     mom_mags = np.linalg.norm(moments, axis=1)
#     n_loaded = np.sum(mom_mags > 1e-10)

#     plotter.add_mesh(
#         arrows, color=MOMENT_COLOR,
#         label=f"Point Moments ({n_loaded} nodes)"
#     )

def plot_point_moments(plotter, mesh, scale=0.05):
    """Plot applied moments with scaled arrows and labels."""
    moments = get_array_safe(mesh, "POINT_MOMENT", "point")
    if moments is None:
        print("    ⚠ POINT_MOMENT not found")
        return

    magnitudes = np.linalg.norm(moments, axis=1)
    mask = magnitudes > 1e-10
    n_loaded = np.sum(mask)

    if n_loaded == 0:
        print("    ⚠ No non-zero point moments")
        return

    pts = mesh.points[mask]
    vecs = moments[mask]
    mags = magnitudes[mask]

    print(f"    Point moments: {n_loaded} nodes")
    print(f"    Min moment: {np.min(mags):.2f}")
    print(f"    Max moment: {np.max(mags):.2f}")

    cloud = pv.PolyData(pts)
    cloud["vectors"] = vecs * scale
    cloud["magnitude"] = mags
    cloud.set_active_vectors("vectors")

    arrows = cloud.glyph(
        orient="vectors",
        scale="magnitude",
        factor=scale
    )

    plotter.add_mesh(
        arrows, color=MOMENT_COLOR,
        label=f"Moments ({n_loaded} nodes)"
    )

    # Color dots
    cloud_dots = pv.PolyData(pts)
    cloud_dots["Moment [Nm]"] = mags
    plotter.add_mesh(
        cloud_dots,
        scalars="Moment [Nm]",
        cmap="Greens",
        point_size=12,
        render_points_as_spheres=True,
    )

    # Labels for top values
    n_labels = min(10, n_loaded)
    top_idx = np.argsort(mags)[-n_labels:]
    for i in top_idx:
        my = vecs[i, 1]
        label_pos = pts[i] + np.array([-0.3, 0.0, 0.3])
        plotter.add_point_labels(
            pv.PolyData(label_pos.reshape(1, 3)),
            [f"My={my:.1f}"],
            font_size=8,
            text_color="darkgreen",
            point_size=0,
            shape=None,
            show_points=False,
            always_visible=True
        )


def plot_reactions(plotter, mesh, scale=0.02):
    """Plot reaction forces at supports."""
    reactions = get_array_safe(mesh, "REACTION", "point")
    if reactions is None:
        print("    ⚠ REACTION not found")
        return

    support_idx = get_support_nodes(mesh)
    if len(support_idx) == 0:
        print("    ⚠ No support nodes found")
        return

    pts = mesh.points[support_idx]
    vecs = reactions[support_idx]

    arrows = make_arrows(pts, vecs, scale)
    if arrows is not None:
        plotter.add_mesh(
            arrows, color=REACTION_COLOR,
            label="Reactions"
        )

    for i, idx in enumerate(support_idx):
        r = reactions[idx]
        print(f"    Support node {idx}: "
              f"Rx={r[0]:.2f} Ry={r[1]:.2f} Rz={r[2]:.2f}")


def plot_reaction_moments(plotter, mesh, scale=0.05):
    """Plot reaction moments at supports."""
    react_mom = get_array_safe(mesh, "REACTION_MOMENT", "point")
    if react_mom is None:
        print("    ⚠ REACTION_MOMENT not found")
        return

    support_idx = get_support_nodes(mesh)
    if len(support_idx) == 0:
        return

    pts = mesh.points[support_idx]
    vecs = react_mom[support_idx]

    arrows = make_arrows(pts, vecs, scale)
    if arrows is not None:
        plotter.add_mesh(
            arrows, color="purple",
            label="Reaction Moments"
        )


def plot_rotation(plotter, mesh, scale=10.0):
    """Plot rotation field on deformed structure."""
    rot = get_array_safe(mesh, "ROTATION", "point")
    disp = get_array_safe(mesh, "DISPLACEMENT", "point")
    if rot is None:
        print("    ⚠ ROTATION not found")
        return

    deformed = mesh.copy()
    if disp is not None:
        deformed.points = mesh.points + disp * scale

    rot_mag = np.linalg.norm(rot, axis=1)

    plotter.add_mesh(
        deformed, scalars=rot_mag,
        line_width=4, style="wireframe",
        cmap="coolwarm",
        scalar_bar_args={"title": "Rotation [rad]"},
        label=f"Rotation (deformed ×{scale})"
    )


def plot_cell_scalar(plotter, mesh, name, title, cmap="viridis"):
    """Plot cell (element) scalar data."""
    data = get_array_safe(mesh, name, "cell")
    if data is None:
        print(f"    ⚠ {name} not found in cell data")
        return

    if len(data.shape) > 1:
        data = np.linalg.norm(data, axis=1)

    plotter.add_mesh(
        mesh, scalars=data,
        line_width=5, style="wireframe",
        cmap=cmap,
        scalar_bar_args={"title": title},
        label=title
    )


def plot_internal_forces(plotter, mesh, var_name, title, scale=10.0):
    """Plot internal forces/moments from gauss points."""
    data = get_array_safe(mesh, var_name, "cell")
    if data is None:
        print(f"    ⚠ {var_name} not found")
        return

    if len(data.shape) > 1:
        mag = np.linalg.norm(data, axis=1)
    else:
        mag = np.abs(data)

    disp = get_array_safe(mesh, "DISPLACEMENT", "point")
    deformed = mesh.copy()
    if disp is not None:
        deformed.points = mesh.points + disp * scale

    plotter.add_mesh(
        deformed, scalars=mag,
        line_width=5, style="wireframe",
        cmap="jet",
        scalar_bar_args={"title": title},
        label=title
    )


# ============================================================
# MAIN PLOTTER
# ============================================================

def plot_case(case_folder, vtk_subfolder, plot_config):
    """Main plotting function."""
    print(f"\n{'=' * 60}")
    print(f"PLOTTING: {case_folder}")
    print(f"{'=' * 60}")

    # Load case config
    config_path = os.path.join(case_folder, "case_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            case_config = json.load(f)
        print(f"  Case ID: {case_config.get('case_id', '?')}")
        summary = case_config.get("load_summary", {})
        if summary:
            print(f"  Loaded nodes: {summary.get('loaded_nodes', '?')}"
                  f"/{summary.get('total_nodes', '?')}")
            print(f"  Fx range: {summary.get('Fx_range', '?')}")
            print(f"  Fz range: {summary.get('Fz_range', '?')}")

    # Find and load VTK
    vtk_path = find_vtk_file(case_folder, vtk_subfolder)
    if vtk_path is None:
        return

    mesh = load_vtk(vtk_path)

    # ── Count active plots ──
    active_plots = [k for k, v in plot_config.items() if v]
    n_plots = len(active_plots)

    if n_plots == 0:
        print("  No plots selected!")
        return

    # ── Single window with subplots ──
    if n_plots <= 2:
        rows, cols = 1, n_plots
    elif n_plots <= 4:
        rows, cols = 2, 2
    elif n_plots <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 4

    plotter = pv.Plotter(
        shape=(rows, cols),
        window_size=WINDOW_SIZE,
        title=f"Case: {os.path.basename(case_folder)}"
    )

    plot_idx = 0

    for plot_name in active_plots:
        row = plot_idx // cols
        col = plot_idx % cols
        if row >= rows:
            break

        plotter.subplot(row, col)
        plotter.set_background(BACKGROUND_COLOR)

        print(f"\n  [{plot_name}]")

        if plot_name == "structure":
            plot_structure(plotter, mesh)
            plotter.add_text("Structure", font_size=10)

        elif plot_name == "displacement":
            plot_structure(plotter, mesh)
            plot_displacement(plotter, mesh, DEFORMATION_SCALE)
            plotter.add_text(
                f"Displacement (×{DEFORMATION_SCALE})", font_size=10
            )

        elif plot_name == "point_loads":
            plot_structure(plotter, mesh)
            plot_point_loads(plotter, mesh, FORCE_ARROW_SCALE)
            plotter.add_text("Point Loads", font_size=10)

        elif plot_name == "point_moment":
            plot_structure(plotter, mesh)
            plot_point_moments(plotter, mesh, MOMENT_ARROW_SCALE)
            plotter.add_text("Point Moments", font_size=10)

        elif plot_name == "reactions":
            plot_structure(plotter, mesh)
            plot_reactions(plotter, mesh, FORCE_ARROW_SCALE)
            plotter.add_text("Reactions", font_size=10)

        elif plot_name == "reaction_moments":
            plot_structure(plotter, mesh)
            plot_reaction_moments(plotter, mesh, MOMENT_ARROW_SCALE)
            plotter.add_text("Reaction Moments", font_size=10)

        elif plot_name == "rotation":
            plot_rotation(plotter, mesh, DEFORMATION_SCALE)
            plotter.add_text("Rotation", font_size=10)

        elif plot_name == "internal_moment":
            plot_internal_forces(
                plotter, mesh, "MOMENT",
                "Internal Moment [Nm]", DEFORMATION_SCALE
            )
            plotter.add_text("Internal Moment", font_size=10)

        elif plot_name == "internal_force":
            plot_internal_forces(
                plotter, mesh, "FORCE",
                "Internal Force [N]", DEFORMATION_SCALE
            )
            plotter.add_text("Internal Force", font_size=10)

        elif plot_name == "young_modulus":
            plot_cell_scalar(
                plotter, mesh, "YOUNG_MODULUS",
                "Young's Modulus [Pa]"
            )
            plotter.add_text("Young's Modulus", font_size=10)

        elif plot_name == "cross_area":
            plot_cell_scalar(
                plotter, mesh, "CROSS_AREA",
                "Cross Area [m²]"
            )
            plotter.add_text("Cross Area", font_size=10)

        elif plot_name == "I22":
            plot_cell_scalar(
                plotter, mesh, "I22",
                "I22 [m⁴]"
            )
            plotter.add_text("I22", font_size=10)

        plotter.add_axes()
        plotter.camera_position = 'xz'

        plot_idx += 1

    plotter.link_views()
    print(f"\n  Showing {n_plots} plots...")
    plotter.show()


# ============================================================
# MAIN
# ============================================================

try:
    import json
except:
    pass

if __name__ == "__main__":
    plot_case(CASE_FOLDER, VTK_SUBFOLDER, PLOT_WHAT)