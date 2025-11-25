(f"\n=== VTK File: {filename} ===")
#     print(mesh)  # general info
    
#     # Points
#     print(f"\nTotal points: {mesh.n_points}")
#     print(f"First {min(num_preview, mesh.n_points)} points:\n", mesh.points[:num_preview])
    
#     # Cells
#     print(f"\nTotal cells: {mesh.n_cells}")
#     print(f"Cell types: {mesh.cells}")
    
#     # Point data arrays
#     if mesh.point_data:
#         print("\nPoint Data Arrays:")
#         for name in mesh.point_data.keys():
#             print(f"  - {name} (first {num_preview} values): {mesh.point_data[name][:num_preview]}")
#     else:
#         print("\nNo point data arrays found.")
    
#     # Cell data arrays
#     if mesh.cell_data:
#         print("\nCell Data Arrays:")
#         for name in mesh.cell_data.keys():
#             print(f"  - {name} (first {num_preview} values): {mesh.cell_data[name][:num_preview]}")
#     else:
#         print("\nNo cell data arrays found.")
    
#     # Optional: visualize
#     mesh.plot()

# # Example usage
# print_vtk_summar