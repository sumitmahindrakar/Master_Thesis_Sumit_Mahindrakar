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

        # Qt central widget
        self.frame = QtWidgets.QFrame()
        self.setCentralWidget(self.frame)
        layout = QtWidgets.QVBoxLayout()
        self.frame.setLayout(layout)

        # PyVista interactor
        self.plotter = QtInteractor(self.frame)
        layout.addWidget(self.plotter.interactor)

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

        # Initial plot
        self.current_field = fields[0]
        self.mesh_actor = self.plotter.add_mesh(
            self.mesh,
            scalars=self.current_field,
            show_edges=True,
            line_width=3,
            cmap='viridis'
        )

        # Markers for min/max
        self.min_actor = None
        self.max_actor = None
        self.min_label = None
        self.max_label = None

        # Grid and axes
        self.plotter.show_grid(color='white')
        self.plotter.show_axes()
        self.plotter.view_xy()
        self.setWindowTitle("Interactive VTK Viewer")

        # Show initial min/max
        self.update_min_max_markers()

    def update_field(self, index):
        """Update mesh coloring and min/max markers"""
        self.current_field = self.fields[index]

        # Remove previous mesh
        if self.mesh_actor:
            self.plotter.remove_actor(self.mesh_actor)

        # Add new mesh with current field
        self.mesh_actor = self.plotter.add_mesh(
            self.mesh,
            scalars=self.current_field,
            show_edges=True,
            line_width=3,
            cmap='viridis'
        )

        # Update min/max markers
        self.update_min_max_markers()
        self.plotter.render()

    def update_min_max_markers(self):
        """Highlight all points with min and max values"""
        # Remove previous markers
        if self.min_actor:
            self.plotter.remove_actor(self.min_actor)
        if self.max_actor:
            self.plotter.remove_actor(self.max_actor)
        if self.min_label:
            self.plotter.remove_actor(self.min_label)
        if self.max_label:
            self.plotter.remove_actor(self.max_label)

        # Get scalar or vector magnitude
        data = self.mesh.point_data[self.current_field]
        if data.ndim > 1 and data.shape[1] == 3:  # vector
            magnitude = np.linalg.norm(data, axis=1)
        else:
            magnitude = data

        min_val = magnitude.min()
        max_val = magnitude.max()

        # Find all points with min/max
        min_indices = np.where(magnitude == min_val)[0]
        max_indices = np.where(magnitude == max_val)[0]

        # Add spheres at all min points
        self.min_actor = []
        for idx in min_indices:
            point = self.mesh.points[idx]
            actor = self.plotter.add_mesh(pv.Sphere(radius=0.05, center=point), color='blue')
            self.min_actor.append(actor)

        # Add spheres at all max points
        self.max_actor = []
        for idx in max_indices:
            point = self.mesh.points[idx]
            actor = self.plotter.add_mesh(pv.Sphere(radius=0.05, center=point), color='red')
            self.max_actor.append(actor)

        # Add corner labels (single value)
        self.min_label = self.plotter.add_text(f"Min: {min_val:.3e}", position=(10, 10), font_size=14, color='blue')
        self.max_label = self.plotter.add_text(f"Max: {max_val:.3e}", position=(10, 40), font_size=14, color='red')


# --------------------------
# Run the application
# --------------------------
app = QtWidgets.QApplication(sys.argv)
viewer = VTKViewer(mesh, fields)
viewer.show()
sys.exit(app.exec_())
