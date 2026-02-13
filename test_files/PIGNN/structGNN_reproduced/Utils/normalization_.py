"""
Utils/normalization.py

Functions for normalizing and denormalizing structural analysis data.

Normalization scales all values to [0, 1] range:
- Improves neural network training
- Handles different scales (coordinates vs forces)
- Ensures numerical stability

Main functions:
- normalize_dataset: Normalize entire dataset
- denormalize_y_linear: Convert predictions back to original scale
- Various helper functions for specific features
"""

import torch
from torch_geometric.loader import DataLoader


def getMinMax_x(dataset, norm_dict):
    """
    Get min/max values for node features (x) across the dataset.
    
    Node features structure:
        [0:3]   - Grid numbers (integers identifying node position in grid)
        [3:6]   - Coordinates (x, y, z positions)
        [6:8]   - Boundary conditions (support types)
        [8]     - Mass
        [9:15]  - Applied forces (6 DOF)
    
    Args:
        dataset: List of graph data objects
        norm_dict: Dictionary to store normalization parameters
    
    Returns:
        norm_dict: Updated with x normalization parameters
    """
    # Load all data at once to find global min/max
    loader = DataLoader(dataset, batch_size=len(dataset))
    data = next(iter(loader))
    x, edge_attr = data.x, data.edge_attr

    # Grid numbers: typically integers like 1, 2, 3...
    # Min is 0, max is the largest grid number
    min_grid_num = 0
    max_grid_num = torch.max(torch.abs(x[:, :3]))
    norm_dict['grid_num'] = [min_grid_num, max_grid_num]

    # Coordinates: x, y, z positions in space
    min_coord = torch.min(torch.abs(x[:, 3:6]))
    max_coord = torch.max(torch.abs(x[:, 3:6]))
    norm_dict['coord'] = [min_coord, max_coord]

    # Mass: always positive, min is 0
    min_mass = 0
    max_mass = torch.max(torch.abs(x[:, 8]))
    norm_dict['mass'] = [min_mass, max_mass]

    # Forces: can be positive or negative
    min_force = 0
    max_force = torch.max(torch.abs(x[:, 9:]))
    norm_dict['force'] = [min_force, max_force]

    # Edge length: always positive
    min_length = 0
    max_length = torch.max(torch.abs(edge_attr[:, 2]))
    norm_dict['length'] = [min_length, max_length]

    # Clean up memory
    del x, edge_attr
    
    return norm_dict


def getMinMax_y_linear(dataset, norm_dict):
    """
    Get min/max values for output targets (y) across the dataset.
    
    Output structure (linear analysis):
        [0:2]   - Displacement (X, Z)
        [2:8]   - Moment Y (6 values)
        [8:14]  - Moment Z (6 values)
        [14:20] - Shear Y (6 values)
        [20:26] - Shear Z (6 values)
        [26:32] - Axial Force (6 values)
        [32:38] - Torsion (6 values)
    
    Args:
        dataset: List of graph data objects
        norm_dict: Dictionary to store normalization parameters
    
    Returns:
        norm_dict: Updated with y normalization parameters
    """
    loader = DataLoader(dataset, batch_size=len(dataset))
    data = next(iter(loader))
    y = data.y

    # Displacement
    min_disp = 0
    max_disp = torch.max(torch.abs(y[:, :2]))
    norm_dict['disp'] = [min_disp, max_disp]

    # Moment Y
    min_momentY = 0
    max_momentY = torch.max(torch.abs(y[:, 2:8]))
    norm_dict['momentY'] = [min_momentY, max_momentY]

    # Moment Z
    min_momentZ = 0
    max_momentZ = torch.max(torch.abs(y[:, 8:14]))
    norm_dict['momentZ'] = [min_momentZ, max_momentZ]

    # Shear Y
    min_shearY = 0
    max_shearY = torch.max(torch.abs(y[:, 14:20]))
    norm_dict['shearY'] = [min_shearY, max_shearY]

    # Shear Z
    min_shearZ = 0
    max_shearZ = torch.max(torch.abs(y[:, 20:26]))
    norm_dict['shearZ'] = [min_shearZ, max_shearZ]

    # Axial Force (if available)
    if y.shape[1] > 26:
        min_axialForce = 0
        max_axialForce = torch.max(torch.abs(y[:, 26:32]))
        norm_dict['axialForce'] = [min_axialForce, max_axialForce]

    # Torsion (if available)
    if y.shape[1] > 32:
        min_torsion = 0
        max_torsion = torch.max(torch.abs(y[:, 32:38]))
        norm_dict['torsion'] = [min_torsion, max_torsion]

    del y
    return norm_dict


def normalize_linear(data, norm_dict):
    """
    Normalize a single data sample using the normalization dictionary.
    
    Formula: normalized = (value - min) / (max - min)
    
    Args:
        data: PyTorch Geometric Data object
        norm_dict: Dictionary with min/max values for each feature
    
    Note: This function modifies data in-place!
    """
    # Normalize node features (x)
    # Grid numbers
    data.x[:, :3] = (data.x[:, :3] - norm_dict['grid_num'][0]) / (norm_dict['grid_num'][1] - norm_dict['grid_num'][0])
    
    # Coordinates
    data.x[:, 3:6] = (data.x[:, 3:6] - norm_dict['coord'][0]) / (norm_dict['coord'][1] - norm_dict['coord'][0])
    
    # Mass
    data.x[:, 8] = (data.x[:, 8] - norm_dict['mass'][0]) / (norm_dict['mass'][1] - norm_dict['mass'][0])
    
    # Forces
    data.x[:, 9:] = (data.x[:, 9:] - norm_dict['force'][0]) / (norm_dict['force'][1] - norm_dict['force'][0])

    # Normalize edge attributes
    # Length is at index 2
    data.edge_attr[:, 2] = (data.edge_attr[:, 2] - norm_dict['length'][0]) / (norm_dict['length'][1] - norm_dict['length'][0])

    # Normalize outputs (y)
    data.y[:, 0:2] = (data.y[:, 0:2] - norm_dict['disp'][0]) / (norm_dict['disp'][1] - norm_dict['disp'][0])
    data.y[:, 2:8] = (data.y[:, 2:8] - norm_dict['momentY'][0]) / (norm_dict['momentY'][1] - norm_dict['momentY'][0])
    data.y[:, 8:14] = (data.y[:, 8:14] - norm_dict['momentZ'][0]) / (norm_dict['momentZ'][1] - norm_dict['momentZ'][0])
    data.y[:, 14:20] = (data.y[:, 14:20] - norm_dict['shearY'][0]) / (norm_dict['shearY'][1] - norm_dict['shearY'][0])
    data.y[:, 20:26] = (data.y[:, 20:26] - norm_dict['shearZ'][0]) / (norm_dict['shearZ'][1] - norm_dict['shearZ'][0])
    
    # Axial force and torsion (if available)
    if data.y.shape[1] > 26 and 'axialForce' in norm_dict:
        data.y[:, 26:32] = (data.y[:, 26:32] - norm_dict['axialForce'][0]) / (norm_dict['axialForce'][1] - norm_dict['axialForce'][0])
    if data.y.shape[1] > 32 and 'torsion' in norm_dict:
        data.y[:, 32:38] = (data.y[:, 32:38] - norm_dict['torsion'][0]) / (norm_dict['torsion'][1] - norm_dict['torsion'][0])


def normalize_dataset(dataset, analysis='linear'):
    """
    Normalize the entire dataset.
    
    Steps:
    1. Compute global min/max for all features
    2. Apply normalization to each sample
    3. Return normalized dataset and normalization dictionary
    
    Args:
        dataset: List of graph data objects
        analysis: Type of analysis ('linear' supported)
    
    Returns:
        tuple: (normalized_dataset, norm_dict)
               - normalized_dataset: Same dataset with normalized values
               - norm_dict: Dictionary of normalization parameters (save this!)
    
    Example:
        >>> dataset = get_dataset(structure_num=100)
        >>> dataset, norm_dict = normalize_dataset(dataset)
        >>> # Save norm_dict to denormalize predictions later
    """
    norm_dict = {}
    
    # Get min/max values
    norm_dict = getMinMax_x(dataset, norm_dict)
    norm_dict = getMinMax_y_linear(dataset, norm_dict)
    
    # Normalize each sample
    for data in dataset:
        normalize_linear(data, norm_dict)

    return dataset, norm_dict


def normalize_dataset_byNormDict(dataset, norm_dict, analysis='linear'):
    """
    Normalize a new dataset using existing normalization parameters.
    
    Use this for test data - normalize using training data's min/max.
    
    Args:
        dataset: List of graph data objects
        norm_dict: Normalization dictionary from training data
        analysis: Type of analysis ('linear' supported)
    
    Returns:
        dataset: Normalized dataset
    """
    for data in dataset:
        normalize_linear(data, norm_dict)
    return dataset


# ============================================================
# Denormalization functions (convert predictions back to original scale)
# ============================================================

def denormalize_grid_num(data, norm_dict):
    """Denormalize grid numbers."""
    return data * (norm_dict['grid_num'][1] - norm_dict['grid_num'][0]) + norm_dict['grid_num'][0]


def denormalize_coord(data, norm_dict):
    """Denormalize coordinates."""
    return data * (norm_dict['coord'][1] - norm_dict['coord'][0]) + norm_dict['coord'][0]


def denormalize_disp(disp, norm_dict):
    """Denormalize displacement predictions."""
    return disp * (norm_dict['disp'][1] - norm_dict['disp'][0]) + norm_dict['disp'][0]


def denormalize_momentY(data, norm_dict):
    """Denormalize moment Y predictions."""
    return data * (norm_dict['momentY'][1] - norm_dict['momentY'][0]) + norm_dict['momentY'][0]


def denormalize_momentZ(data, norm_dict):
    """Denormalize moment Z predictions."""
    return data * (norm_dict['momentZ'][1] - norm_dict['momentZ'][0]) + norm_dict['momentZ'][0]


def denormalize_shearY(data, norm_dict):
    """Denormalize shear Y predictions."""
    return data * (norm_dict['shearY'][1] - norm_dict['shearY'][0]) + norm_dict['shearY'][0]


def denormalize_shearZ(data, norm_dict):
    """Denormalize shear Z predictions."""
    return data * (norm_dict['shearZ'][1] - norm_dict['shearZ'][0]) + norm_dict['shearZ'][0]


def denormalize_y_linear(y, norm_dict):
    """
    Denormalize all output predictions back to original scale.
    
    Use this to convert model predictions to actual physical values.
    
    Args:
        y: Normalized predictions, shape (num_nodes, num_outputs)
        norm_dict: Normalization dictionary used for training
    
    Returns:
        y: Denormalized predictions in original scale
    
    Example:
        >>> # After model prediction
        >>> normalized_pred = model(data.x, data.edge_index, data.edge_attr)
        >>> actual_pred = denormalize_y_linear(normalized_pred, norm_dict)
        >>> print(f"Actual displacement: {actual_pred[:, 0:2]} meters")
    """
    # Make a copy to avoid modifying original
    y = y.clone()
    
    # Denormalize each output type
    y[:, 0:2] = y[:, 0:2] * (norm_dict['disp'][1] - norm_dict['disp'][0]) + norm_dict['disp'][0]
    y[:, 2:8] = y[:, 2:8] * (norm_dict['momentY'][1] - norm_dict['momentY'][0]) + norm_dict['momentY'][0]
    y[:, 8:14] = y[:, 8:14] * (norm_dict['momentZ'][1] - norm_dict['momentZ'][0]) + norm_dict['momentZ'][0]
    y[:, 14:20] = y[:, 14:20] * (norm_dict['shearY'][1] - norm_dict['shearY'][0]) + norm_dict['shearY'][0]
    y[:, 20:26] = y[:, 20:26] * (norm_dict['shearZ'][1] - norm_dict['shearZ'][0]) + norm_dict['shearZ'][0]
    
    # Axial force and torsion (if available)
    if y.shape[1] > 26 and 'axialForce' in norm_dict:
        y[:, 26:32] = y[:, 26:32] * (norm_dict['axialForce'][1] - norm_dict['axialForce'][0]) + norm_dict['axialForce'][0]
    if y.shape[1] > 32 and 'torsion' in norm_dict:
        y[:, 32:38] = y[:, 32:38] * (norm_dict['torsion'][1] - norm_dict['torsion'][0]) + norm_dict['torsion'][0]
    
    return y


def print_norm_dict(norm_dict):
    """
    Pretty print the normalization dictionary.
    
    Args:
        norm_dict: Normalization dictionary
    """
    print("\n  Normalization Parameters:")
    print("  " + "─" * 50)
    print(f"  {'Feature':<20} {'Min':<15} {'Max':<15}")
    print("  " + "─" * 50)
    
    for key, (min_val, max_val) in norm_dict.items():
        if isinstance(min_val, torch.Tensor):
            min_str = f"{min_val.item():.6f}"
            max_str = f"{max_val.item():.6f}"
        else:
            min_str = f"{min_val:.6f}" if isinstance(min_val, float) else str(min_val)
            max_str = f"{max_val:.6f}" if isinstance(max_val, float) else str(max_val)
        print(f"  {key:<20} {min_str:<15} {max_str:<15}")
    
    print("  " + "─" * 50)