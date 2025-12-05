g, ax = plt.subplots(figsize=(10, 8), dpi=FIGURE_DPI)
    # setup_axes(ax, f'Axial Force Diagram (Scale: {AXIAL_SCALE}x)', points)
    # plot_structure(ax, points, cells, color='gray', linewidth=1.5)
    # plot_axial_diagram(ax, points, cells, cell_force, scale=AXIAL_SCALE,
    #                   show_values=True)
    # add_supports(ax, points)
    
    # # Legend
    # tension_line = Line2D([0], [0], color=COLOR_AXIAL_TENSION, linewidth=2)
    # compression_line = Line2D([0], [0], color=COLOR_AXIAL_COMPRESSION, linewidth=2)
    # ax.legend([tension_line, compression_line], ['Tension (T)', 'Compression (C)'], 
    #          loc='upper right')
    
    # plt.savefig(os.path.join(output_folder, '6_axial_force.png'), dpi=300, bbox_inches='tight')
    # print(f"Saved: {output_folder}/6_axial_force.png")
    # plt.show()