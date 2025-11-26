import sys
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5 import QtWidgets
import numpy as np

# Load VTK file
vtk_file = "test_files/frame_3D_test_udl.gid/vtk_output/Structure_0_1.vtk"
mesh = pv.read(vtk_file)
fields = list(mesh.point_data.keys())
print("Available fields:", fields)

class VTKViewer(QtWidgets.QMainWindow):
    def __init__(self, mesh, fields):
        super().__init__()
        self.mesh = mesh
        self.fields = fields

        # Descriptive field titles
        self.field_titles = {
            "DISPLACEMENT": "Nodal Displacement (m)",
            "REACTION": "Reaction Force at Nodes (N)",
            "ROTATION": "Nodal Rotation (rad)",
            "REACTION_MOMENT": "Reaction Moment at Nodes (N·m)"
        }

        # Qt central widget
        self.frame = QtWidgets.QFrame()
        self.setCentralWidget(self.frame)
        layout = QtWidgets.QVBoxLayout()
        self.frame.setLayout(layout)

        # PyVista interactor
        self.plotter = QtInteractor(self.frame)
        layout.addWidget(self.plotter.interactor)

        # White background
        self.plotter.set_background("white")

        # Controls layout
        control_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(control_layout)

        # Dropdown for field selection
        self.dropdown = QtWidgets.QComboBox()
        self.dropdown.addItems(fields)
        self.dropdown.currentIndexChanged.connect(self.update_field)
        control_layout.addWidget(QtWidgets.QLabel("Select Field:"))
        control_layout.addWidget(self.dropdown)

        # Plane view buttons
        self.xy_button = QtWidgets.QPushButton("XY View")
        self.xy_button.clicked.connect(lambda: self.plotter.view_xy())
        self.xz_button = QtWidgets.QPushButton("XZ View")
        self.xz_button.clicked.connect(lambda: self.plotter.view_xz())
        self.yz_button = QtWidgets.QPushButton("YZ View")
        self.yz_button.clicked.connect(lambda: self.plotter.view_yz())
        control_layout.addWidget(self.xy_button)
        control_layout.addWidget(self.xz_button)
        control_layout.addWidget(self.yz_button)

        # Grid toggle checkbox
        self.grid_checkbox = QtWidgets.QCheckBox("Show Grid")
        self.grid_checkbox.setChecked(True)
        self.grid_checkbox.stateChanged.connect(self.toggle_grid)
        control_layout.addWidget(self.grid_checkbox)

        # Initial field
        self.current_field = fields[0]

        # Add initial mesh (no scalar bar)
        self.mesh_actor = self.plotter.add_mesh(
            self.mesh,
            scalars=self.mesh.point_data[self.current_field],
            show_edges=True,
            line_width=3,
            cmap='viridis',
            show_scalar_bar=False
        )

        # Add vertical scalar bar
        self.scalar_bar_actor = self.plotter.add_scalar_bar(
            title=self.current_field,
            vertical=True,
            position_x=0.85,
            position_y=0.1,
            height=0.8,
            width=0.05
        )

        # Plot title
        self.plot_title = self.plotter.add_text(
            self.field_titles.get(self.current_field, self.current_field),
            position='upper_edge',
            font_size=18,
            color='black'
        )

        # Grid and axes
        self.grid_on = True
        self.grid_actor = self.plotter.show_grid(color='gray')
        self.plotter.show_axes()
        self.plotter.view_xy()

        # Min/Max markers
        self.min_actor = []
        self.max_actor = []
        self.min_label = None
        self.max_label = None
        self.update_min_max_markers()

        # Window title matches plot title
        self.setWindowTitle(self.field_titles.get(self.current_field, self.current_field))

    def update_field(self, index):
        """Update plot when a new field is selected."""
        self.current_field = self.fields[index]

        # Update window title and plot title
        self.setWindowTitle(self.field_titles.get(self.current_field, self.current_field))
        if self.plot_title:
            self.plotter.remove_actor(self.plot_title)
        self.plot_title = self.plotter.add_text(
            self.field_titles.get(self.current_field, self.current_field),
            position='upper_edge',
            font_size=18,
            color='black'
        )

        # Remove previous scalar bar if exists
        if self.scalar_bar_actor:
            self.plotter.remove_actor(self.scalar_bar_actor)
            self.scalar_bar_actor = None

        # Remove old mesh actor
        if self.mesh_actor:
            self.plotter.remove_actor(self.mesh_actor)

        # Add mesh again with current field
        self.mesh_actor = self.plotter.add_mesh(
            self.mesh,
            scalars=self.mesh.point_data[self.current_field],
            show_edges=True,
            line_width=3,
            cmap='viridis',
            show_scalar_bar=False
        )

        # Add vertical scalar bar
        self.scalar_bar_actor = self.plotter.add_scalar_bar(
            title=self.current_field,
            vertical=True,
            position_x=0.85,
            position_y=0.1,
            height=0.8,
            width=0.05
        )

        # Update min/max markers
        self.update_min_max_markers()

        self.plotter.render()

    def update_min_max_markers(self):
        """Highlight all points with min and max values."""
        # Remove previous min/max actors
        for a in self.min_actor + self.max_actor:
            self.plotter.remove_actor(a)
        self.min_actor = []
        self.max_actor = []

        if self.min_label:
            self.plotter.remove_actor(self.min_label)
        if self.max_label:
            self.plotter.remove_actor(self.max_label)
        self.min_label = None
        self.max_label = None

        # Get scalar or vector magnitude
        data = self.mesh.point_data[self.current_field]
        if data.ndim > 1 and data.shape[1] == 3:  # vector
            magnitude = np.linalg.norm(data, axis=1)
        else:
            magnitude = data

        min_val = magnitude.min()
        max_val = magnitude.max()

        # Find all indices for min/max
        min_indices = np.where(magnitude == min_val)[0]
        max_indices = np.where(magnitude == max_val)[0]

        # Add spheres for all min points
        for idx in min_indices:
            point = self.mesh.points[idx]
            actor = self.plotter.add_mesh(pv.Sphere(radius=0.05, center=point), color='blue')
            self.min_actor.append(actor)

        # Add spheres for all max points
        for idx in max_indices:
            point = self.mesh.points[idx]
            actor = self.plotter.add_mesh(pv.Sphere(radius=0.05, center=point), color='red')
            self.max_actor.append(actor)

        # Add corner labels
        self.min_label = self.plotter.add_text(f"Min: {min_val:.3e}", position=(10, 10), font_size=14, color='blue')
        self.max_label = self.plotter.add_text(f"Max: {max_val:.3e}", position=(10, 40), font_size=14, color='red')

    def toggle_grid(self):
        """Show or hide grid."""
        self.grid_on = self.grid_checkbox.isChecked()
        if self.grid_actor:
            self.plotter.remove_actor(self.grid_actor)
            self.grid_actor = None
        if self.grid_on:
            self.grid_actor = self.plotter.show_grid(color='gray')
        self.plotter.render()


# --------------------------
# Run the application
# --------------------------
app = QtWidgets.QApplication(sys.argv)
viewer = VTKViewer(mesh, fields)
viewer.show()
sys.exit(app.exec_())
