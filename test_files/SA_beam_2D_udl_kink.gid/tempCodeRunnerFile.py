if save_figures:
        filepath = os.path.join(output_folder, 'rotation_diagram.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")