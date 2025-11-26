import sys
import os
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5 import QtWidgets
import numpy as np

# Load VTK file
vtk_file = "test_files/beam_2D_test_udl.gid/vtk_output/Structure_0_1.vtk"
mesh = pv.read(vtk_file)
fields = list(mesh.point_data.keys())
print("Available fields:", fields)


class VTKViewer(QtWidgets.QMainWindow):
    def __init__(self, mesh, fields):
        super().__init__()
        self.mesh = mesh
        self.original_mesh = mesh.copy()  # Store original undeformed mesh
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

        # ----- DEFORMATION CONTROLS -----
        deform_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(deform_layout)

        # Show undeformed mesh checkbox
        self.show_undeformed_checkbox = QtWidgets.QCheckBox("Show Undeformed")
        self.show_undeformed_checkbox.setChecked(False)
        self.show_undeformed_checkbox.stateChanged.connect(self.toggle_undeformed)
        deform_layout.addWidget(self.show_undeformed_checkbox)

        # Show deformed mesh checkbox
        self.show_deformed_checkbox = QtWidgets.QCheckBox("Show Deformed")
        self.show_deformed_checkbox.setChecked(True)
        self.show_deformed_checkbox.stateChanged.connect(self.toggle_deformed)
        deform_layout.addWidget(self.show_deformed_checkbox)

        # Deformation scale factor
        deform_layout.addWidget(QtWidgets.QLabel("Scale Factor:"))
        self.scale_spinbox = QtWidgets.QDoubleSpinBox()
        self.scale_spinbox.setRange(0.1, 10000.0)
        self.scale_spinbox.setValue(1.0)
        self.scale_spinbox.setSingleStep(1.0)
        self.scale_spinbox.setDecimals(1)
        self.scale_spinbox.valueChanged.connect(self.update_deformation_scale)
        deform_layout.addWidget(self.scale_spinbox)

        # Auto-scale button
        self.auto_scale_button = QtWidgets.QPushButton("Auto Scale")
        self.auto_scale_button.clicked.connect(self.auto_scale_deformation)
        deform_layout.addWidget(self.auto_scale_button)

        # Undeformed style options
        deform_layout.addWidget(QtWidgets.QLabel("Undeformed Style:"))
        self.undeformed_style_combo = QtWidgets.QComboBox()
        self.undeformed_style_combo.addItems(["Wireframe", "Surface", "Points"])
        self.undeformed_style_combo.currentIndexChanged.connect(self.update_undeformed_style)
        deform_layout.addWidget(self.undeformed_style_combo)

        # Undeformed opacity slider
        deform_layout.addWidget(QtWidgets.QLabel("Opacity:"))
        self.opacity_slider = QtWidgets.QSlider(1)  # Horizontal
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.setMaximumWidth(100)
        self.opacity_slider.valueChanged.connect(self.update_undeformed_opacity)
        deform_layout.addWidget(self.opacity_slider)

        deform_layout.addStretch()

        # ----- SAVE OPTIONS -----
        save_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(save_layout)

        # Save button with dropdown menu
        self.save_button = QtWidgets.QPushButton("💾 Save Plot")
        self.save_button.clicked.connect(self.save_screenshot)
        save_layout.addWidget(self.save_button)

        # Resolution dropdown for high-quality exports
        self.resolution_combo = QtWidgets.QComboBox()
        self.resolution_combo.addItems(["1x (Screen)", "2x (HD)", "4x (4K)", "8x (Print)"])
        self.resolution_combo.setCurrentIndex(1)  # Default to 2x
        save_layout.addWidget(QtWidgets.QLabel("Quality:"))
        save_layout.addWidget(self.resolution_combo)

        # Save all fields button
        self.save_all_button = QtWidgets.QPushButton("Save All Fields")
        self.save_all_button.clicked.connect(self.save_all_fields)
        save_layout.addWidget(self.save_all_button)

        save_layout.addStretch()

        # Initial field
        self.current_field = fields[0]

        # Initialize actor references
        self.mesh_actor = None
        self.undeformed_actor = None
        self.scalar_bar_actor = None
        self.plot_title = None
        self.min_actor = []
        self.max_actor = []
        self.min_label = None
        self.max_label = None

        # Deformation settings
        self.show_undeformed = False
        self.show_deformed = True
        self.deformation_scale = 1.0
        self.undeformed_opacity = 0.5
        self.undeformed_style = "wireframe"

        # Create deformed mesh
        self.deformed_mesh = self._create_deformed_mesh()

        # Grid and axes
        self.grid_on = True
        self.grid_actor = self.plotter.show_grid(color='gray')
        self.plotter.show_axes()

        # Add initial mesh with scalar bar
        self._add_mesh_with_scalars()

        # Plot title
        self.plot_title = self.plotter.add_text(
            self.field_titles.get(self.current_field, self.current_field),
            position='upper_edge',
            font_size=18,
            color='black'
        )

        self.plotter.view_xy()

        # Min/Max markers
        self.update_min_max_markers()

        # Window title matches plot title
        self.setWindowTitle(self.field_titles.get(self.current_field, self.current_field))

        # Create menu bar with save options
        self._create_menu_bar()

    def _create_deformed_mesh(self):
        """Create deformed mesh by applying displacement field."""
        deformed = self.original_mesh.copy()
        
        # Check if DISPLACEMENT field exists
        if "DISPLACEMENT" in self.mesh.point_data:
            displacement = self.mesh.point_data["DISPLACEMENT"]
            
            # Ensure displacement is the right shape
            if displacement.ndim == 1:
                # If 1D, assume it's magnitude only - can't deform
                print("Warning: Displacement is scalar, cannot create deformed shape")
                return deformed
            
            # Apply scaled displacement
            deformed.points = self.original_mesh.points + displacement * self.deformation_scale
            
            # Copy all point data to deformed mesh
            for key in self.mesh.point_data.keys():
                deformed.point_data[key] = self.mesh.point_data[key]
        else:
            print("Warning: No DISPLACEMENT field found")
        
        return deformed

    def _create_menu_bar(self):
        """Create menu bar with file options."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        # Save screenshot action
        save_action = QtWidgets.QAction('Save Screenshot (PNG)', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_screenshot)
        file_menu.addAction(save_action)
        
        # Save as PDF action
        save_pdf_action = QtWidgets.QAction('Save as PDF', self)
        save_pdf_action.setShortcut('Ctrl+P')
        save_pdf_action.triggered.connect(self.save_as_pdf)
        file_menu.addAction(save_pdf_action)
        
        # Save as SVG action
        save_svg_action = QtWidgets.QAction('Save as SVG (Vector)', self)
        save_svg_action.triggered.connect(self.save_as_svg)
        file_menu.addAction(save_svg_action)
        
        file_menu.addSeparator()
        
        # Save all fields
        save_all_action = QtWidgets.QAction('Save All Fields', self)
        save_all_action.setShortcut('Ctrl+Shift+S')
        save_all_action.triggered.connect(self.save_all_fields)
        file_menu.addAction(save_all_action)
        
        file_menu.addSeparator()
        
        # Export menu
        export_menu = file_menu.addMenu('Export 3D Model')
        
        # Export as VTK
        export_vtk_action = QtWidgets.QAction('Export as VTK', self)
        export_vtk_action.triggered.connect(lambda: self.export_3d_model('vtk'))
        export_menu.addAction(export_vtk_action)
        
        # Export as STL
        export_stl_action = QtWidgets.QAction('Export as STL', self)
        export_stl_action.triggered.connect(lambda: self.export_3d_model('stl'))
        export_menu.addAction(export_stl_action)
        
        # Export as OBJ
        export_obj_action = QtWidgets.QAction('Export as OBJ', self)
        export_obj_action.triggered.connect(lambda: self.export_3d_model('obj'))
        export_menu.addAction(export_obj_action)
        
        # Export deformed mesh
        file_menu.addSeparator()
        export_deformed_action = QtWidgets.QAction('Export Deformed Mesh (VTK)', self)
        export_deformed_action.triggered.connect(self.export_deformed_mesh)
        file_menu.addAction(export_deformed_action)

        file_menu.addSeparator()
        
        # Exit action
        exit_action = QtWidgets.QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu('View')
        
        xy_action = QtWidgets.QAction('XY View', self)
        xy_action.setShortcut('1')
        xy_action.triggered.connect(self.plotter.view_xy)
        view_menu.addAction(xy_action)
        
        xz_action = QtWidgets.QAction('XZ View', self)
        xz_action.setShortcut('2')
        xz_action.triggered.connect(self.plotter.view_xz)
        view_menu.addAction(xz_action)
        
        yz_action = QtWidgets.QAction('YZ View', self)
        yz_action.setShortcut('3')
        yz_action.triggered.connect(self.plotter.view_yz)
        view_menu.addAction(yz_action)
        
        view_menu.addSeparator()
        
        isometric_action = QtWidgets.QAction('Isometric View', self)
        isometric_action.setShortcut('4')
        isometric_action.triggered.connect(self.plotter.view_isometric)
        view_menu.addAction(isometric_action)
        
        reset_action = QtWidgets.QAction('Reset Camera', self)
        reset_action.setShortcut('R')
        reset_action.triggered.connect(self.plotter.reset_camera)
        view_menu.addAction(reset_action)

        # Deformation menu
        deform_menu = menubar.addMenu('Deformation')
        
        toggle_undeformed_action = QtWidgets.QAction('Toggle Undeformed', self)
        toggle_undeformed_action.setShortcut('U')
        toggle_undeformed_action.triggered.connect(self._toggle_undeformed_from_menu)
        deform_menu.addAction(toggle_undeformed_action)
        
        toggle_deformed_action = QtWidgets.QAction('Toggle Deformed', self)
        toggle_deformed_action.setShortcut('D')
        toggle_deformed_action.triggered.connect(self._toggle_deformed_from_menu)
        deform_menu.addAction(toggle_deformed_action)
        
        deform_menu.addSeparator()
        
        auto_scale_action = QtWidgets.QAction('Auto Scale Deformation', self)
        auto_scale_action.setShortcut('A')
        auto_scale_action.triggered.connect(self.auto_scale_deformation)
        deform_menu.addAction(auto_scale_action)
        
        reset_scale_action = QtWidgets.QAction('Reset Scale to 1.0', self)
        reset_scale_action.triggered.connect(lambda: self.scale_spinbox.setValue(1.0))
        deform_menu.addAction(reset_scale_action)

    def _toggle_undeformed_from_menu(self):
        """Toggle undeformed mesh visibility from menu."""
        self.show_undeformed_checkbox.setChecked(not self.show_undeformed_checkbox.isChecked())

    def _toggle_deformed_from_menu(self):
        """Toggle deformed mesh visibility from menu."""
        self.show_deformed_checkbox.setChecked(not self.show_deformed_checkbox.isChecked())

    def toggle_undeformed(self):
        """Show or hide undeformed mesh."""
        self.show_undeformed = self.show_undeformed_checkbox.isChecked()
        self._update_mesh_display()

    def toggle_deformed(self):
        """Show or hide deformed mesh."""
        self.show_deformed = self.show_deformed_checkbox.isChecked()
        self._update_mesh_display()

    def update_deformation_scale(self, value):
        """Update the deformation scale factor."""
        self.deformation_scale = value
        self.deformed_mesh = self._create_deformed_mesh()
        self._update_mesh_display()
        self.update_min_max_markers()

    def auto_scale_deformation(self):
        """Automatically calculate a good scale factor for visualization."""
        if "DISPLACEMENT" not in self.mesh.point_data:
            QtWidgets.QMessageBox.warning(
                self,
                "No Displacement Data",
                "Cannot auto-scale: No DISPLACEMENT field found in the mesh."
            )
            return
        
        displacement = self.mesh.point_data["DISPLACEMENT"]
        
        if displacement.ndim == 1:
            QtWidgets.QMessageBox.warning(
                self,
                "Scalar Displacement",
                "Cannot auto-scale: Displacement is scalar, not vector."
            )
            return
        
        # Calculate max displacement magnitude
        max_disp = np.max(np.linalg.norm(displacement, axis=1))
        
        if max_disp == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Zero Displacement",
                "Cannot auto-scale: Maximum displacement is zero."
            )
            return
        
        # Calculate model size (bounding box diagonal)
        bounds = self.original_mesh.bounds
        model_size = np.sqrt(
            (bounds[1] - bounds[0])**2 + 
            (bounds[3] - bounds[2])**2 + 
            (bounds[5] - bounds[4])**2
        )
        
        # Target: deformation should be about 10% of model size
        target_deformation = model_size * 0.1
        scale_factor = target_deformation / max_disp
        
        # Round to nice number
        scale_factor = round(scale_factor, 1)
        
        # Update spinbox (this will trigger update_deformation_scale)
        self.scale_spinbox.setValue(scale_factor)
        
        # Show info
        QtWidgets.QMessageBox.information(
            self,
            "Auto Scale Applied",
            f"Scale factor set to {scale_factor:.1f}\n\n"
            f"Max displacement: {max_disp:.2e}\n"
            f"Model size: {model_size:.2f}\n"
            f"Scaled max displacement: {max_disp * scale_factor:.2e}"
        )

    def update_undeformed_style(self, index):
        """Update the rendering style of undeformed mesh."""
        styles = ["wireframe", "surface", "points"]
        self.undeformed_style = styles[index]
        if self.show_undeformed:
            self._update_mesh_display()

    def update_undeformed_opacity(self, value):
        """Update the opacity of undeformed mesh."""
        self.undeformed_opacity = value / 100.0
        if self.show_undeformed:
            self._update_mesh_display()

    def _update_mesh_display(self):
        """Update the display of deformed and undeformed meshes."""
        # Remove existing mesh actors
        self._remove_mesh_and_scalar_bar()
        
        # Add undeformed mesh if enabled
        if self.show_undeformed:
            self._add_undeformed_mesh()
        
        # Add deformed mesh if enabled
        if self.show_deformed:
            self._add_mesh_with_scalars()
        
        self.plotter.render()

    def _add_undeformed_mesh(self):
        """Add undeformed mesh with specified style."""
        style_map = {
            "wireframe": "wireframe",
            "surface": "surface",
            "points": "points"
        }
        
        render_style = style_map.get(self.undeformed_style, "wireframe")
        
        self.undeformed_actor = self.plotter.add_mesh(
            self.original_mesh,
            style=render_style,
            color='gray',
            opacity=self.undeformed_opacity,
            show_edges=True if render_style != "points" else False,
            edge_color='darkgray',
            line_width=1,
            point_size=5 if render_style == "points" else 1,
            show_scalar_bar=False,
            name="undeformed_mesh"
        )

    def export_deformed_mesh(self):
        """Export the deformed mesh to VTK file."""
        default_name = f"deformed_mesh_scale_{self.deformation_scale:.1f}.vtk"
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Deformed Mesh",
            default_name,
            "VTK File (*.vtk);;All Files (*)"
        )
        
        if filename:
            try:
                self.deformed_mesh.save(filename)
                QtWidgets.QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Deformed mesh exported successfully!\n\n"
                    f"File: {filename}\n"
                    f"Scale factor: {self.deformation_scale}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to export deformed mesh:\n{str(e)}"
                )

    def _get_resolution_factor(self):
        """Get resolution multiplier based on dropdown selection."""
        resolution_map = {
            0: 1,   # 1x (Screen)
            1: 2,   # 2x (HD)
            2: 4,   # 4x (4K)
            3: 8    # 8x (Print)
        }
        return resolution_map.get(self.resolution_combo.currentIndex(), 2)

    def _get_default_filename(self, extension='png'):
        """Generate default filename based on current field."""
        safe_field_name = self.current_field.replace(' ', '_').replace('/', '_')
        return f"{safe_field_name}_plot.{extension}"

    def save_screenshot(self):
        """Save current view as PNG image."""
        default_name = self._get_default_filename('png')
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Screenshot",
            default_name,
            "PNG Image (*.png);;JPEG Image (*.jpg);;TIFF Image (*.tiff);;All Files (*)"
        )
        
        if filename:
            factor = self._get_resolution_factor()
            
            try:
                window_size = self.plotter.window_size
                high_res_size = (window_size[0] * factor, window_size[1] * factor)
                
                self.plotter.screenshot(
                    filename,
                    transparent_background=False,
                    window_size=high_res_size
                )
                
                QtWidgets.QMessageBox.information(
                    self,
                    "Screenshot Saved",
                    f"Screenshot saved successfully!\n\nFile: {filename}\n"
                    f"Resolution: {high_res_size[0]}x{high_res_size[1]}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to save screenshot:\n{str(e)}"
                )

    def save_as_pdf(self):
        """Save current view as PDF (vector graphics)."""
        default_name = self._get_default_filename('pdf')
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save as PDF",
            default_name,
            "PDF Document (*.pdf);;All Files (*)"
        )
        
        if filename:
            try:
                self.plotter.save_graphic(filename)
                QtWidgets.QMessageBox.information(
                    self,
                    "PDF Saved",
                    f"PDF saved successfully!\n\nFile: {filename}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to save PDF:\n{str(e)}"
                )

    def save_as_svg(self):
        """Save current view as SVG (vector graphics)."""
        default_name = self._get_default_filename('svg')
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save as SVG",
            default_name,
            "SVG Image (*.svg);;All Files (*)"
        )
        
        if filename:
            try:
                self.plotter.save_graphic(filename)
                QtWidgets.QMessageBox.information(
                    self,
                    "SVG Saved",
                    f"SVG saved successfully!\n\nFile: {filename}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to save SVG:\n{str(e)}"
                )

    def save_all_fields(self):
        """Save screenshots of all fields to a selected directory."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Save All Fields"
        )
        
        if not directory:
            return
        
        factor = self._get_resolution_factor()
        window_size = self.plotter.window_size
        high_res_size = (window_size[0] * factor, window_size[1] * factor)
        
        progress = QtWidgets.QProgressDialog(
            "Saving field plots...",
            "Cancel",
            0,
            len(self.fields),
            self
        )
        progress.setWindowModality(2)
        progress.show()
        
        saved_files = []
        
        try:
            for i, field in enumerate(self.fields):
                if progress.wasCanceled():
                    break
                
                progress.setValue(i)
                progress.setLabelText(f"Saving {field}...")
                QtWidgets.QApplication.processEvents()
                
                self.dropdown.setCurrentIndex(i)
                QtWidgets.QApplication.processEvents()
                
                safe_field_name = field.replace(' ', '_').replace('/', '_')
                filename = os.path.join(directory, f"{safe_field_name}.png")
                
                self.plotter.screenshot(
                    filename,
                    transparent_background=False,
                    window_size=high_res_size
                )
                saved_files.append(filename)
            
            progress.setValue(len(self.fields))
            
            QtWidgets.QMessageBox.information(
                self,
                "All Fields Saved",
                f"Successfully saved {len(saved_files)} plots!\n\n"
                f"Directory: {directory}\n"
                f"Resolution: {high_res_size[0]}x{high_res_size[1]}"
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                f"Failed to save all fields:\n{str(e)}"
            )

    def export_3d_model(self, format_type):
        """Export the mesh as a 3D model file."""
        extensions = {
            'vtk': ('VTK File (*.vtk)', '.vtk'),
            'stl': ('STL File (*.stl)', '.stl'),
            'obj': ('OBJ File (*.obj)', '.obj')
        }
        
        filter_str, ext = extensions.get(format_type, ('All Files (*)', ''))
        default_name = f"exported_mesh{ext}"
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            f"Export as {format_type.upper()}",
            default_name,
            f"{filter_str};;All Files (*)"
        )
        
        if filename:
            try:
                self.mesh.save(filename)
                QtWidgets.QMessageBox.information(
                    self,
                    "Export Successful",
                    f"3D model exported successfully!\n\nFile: {filename}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to export model:\n{str(e)}"
                )

    def _get_scalar_values(self, field_name):
        """Get scalar values for a field."""
        # Use deformed mesh for scalar values
        mesh_to_use = self.deformed_mesh if self.show_deformed else self.original_mesh
        data = mesh_to_use.point_data[field_name]
        if data.ndim > 1 and data.shape[1] == 3:
            return np.linalg.norm(data, axis=1)
        else:
            return data.ravel()

    def _add_mesh_with_scalars(self):
        """Add mesh with proper scalar mapping and separate scalar bar."""
        scalars = self._get_scalar_values(self.current_field)
        
        # Use deformed mesh for display
        display_mesh = self.deformed_mesh if self.show_deformed else self.original_mesh

        self.mesh_actor = self.plotter.add_mesh(
            display_mesh,
            scalars=scalars,
            show_edges=True,
            line_width=3,
            cmap='viridis',
            show_scalar_bar=False,
            name="main_mesh"
        )

        self.scalar_bar_actor = self.plotter.add_scalar_bar(
            title=self.current_field,
            vertical=True,
            position_x=0.85,
            position_y=0.1,
            height=0.8,
            width=0.05,
            title_font_size=16,
            label_font_size=14,
            color='black'
        )

    def _remove_mesh_and_scalar_bar(self):
        """Remove both mesh actor and scalar bar actor."""
        if self.mesh_actor:
            self.plotter.remove_actor(self.mesh_actor)
            self.mesh_actor = None
        
        if self.undeformed_actor:
            self.plotter.remove_actor(self.undeformed_actor)
            self.undeformed_actor = None
        
        if self.scalar_bar_actor:
            self.plotter.remove_actor(self.scalar_bar_actor)
            self.scalar_bar_actor = None
        
        if hasattr(self.plotter, '_scalar_bars'):
            self.plotter._scalar_bars.clear()

    def update_field(self, index):
        """Update plot when a new field is selected."""
        self.current_field = self.fields[index]

        self.setWindowTitle(self.field_titles.get(self.current_field, self.current_field))

        if self.plot_title:
            self.plotter.remove_actor(self.plot_title)
        self.plot_title = self.plotter.add_text(
            self.field_titles.get(self.current_field, self.current_field),
            position='upper_edge',
            font_size=18,
            color='black'
        )

        self._update_mesh_display()
        self.update_min_max_markers()

        self.plotter.render()

    def update_min_max_markers(self):
        """Highlight all points with min and max values."""
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

        magnitude = self._get_scalar_values(self.current_field)

        min_val = magnitude.min()
        max_val = magnitude.max()

        min_indices = np.where(magnitude == min_val)[0]
        max_indices = np.where(magnitude == max_val)[0]

        # Use deformed mesh for marker positions
        display_mesh = self.deformed_mesh if self.show_deformed else self.original_mesh

        for idx in min_indices:
            point = display_mesh.points[idx]
            actor = self.plotter.add_mesh(
                pv.Sphere(radius=0.05, center=point),
                color='blue',
                show_scalar_bar=False
            )
            self.min_actor.append(actor)

        for idx in max_indices:
            point = display_mesh.points[idx]
            actor = self.plotter.add_mesh(
                pv.Sphere(radius=0.05, center=point),
                color='red',
                show_scalar_bar=False
            )
            self.max_actor.append(actor)

        self.min_label = self.plotter.add_text(
            f"Min: {min_val:.3e}",
            position=(10, 10),
            font_size=14,
            color='blue'
        )
        self.max_label = self.plotter.add_text(
            f"Max: {max_val:.3e}",
            position=(10, 40),
            font_size=14,
            color='red'
        )

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