"""
Utils/normalization.py

Flexible normalization functions that automatically adapt to any data format.
"""

import torch
from torch_geometric.loader import DataLoader


def get_global_stats(dataset):
    """
    Get global min/max statistics for all features in the dataset.
    Automatically adapts to any data format.
    """
    loader = DataLoader(dataset, batch_size=len(dataset))
    data = next(iter(loader))
    
    norm_dict = {}
    
    # Node features (x)
    x = data.x
    norm_dict['x_min'] = x.min(dim=0).values
    norm_dict['x_max'] = x.max(dim=0).values
    norm_dict['x_dim'] = x.shape[1]
    
    # Edge attributes
    if data.edge_attr is not None:
        edge_attr = data.edge_attr
        norm_dict['edge_attr_min'] = edge_attr.min(dim=0).values
        norm_dict['edge_attr_max'] = edge_attr.max(dim=0).values
        norm_dict['edge_attr_dim'] = edge_attr.shape[1]
    
    # Output targets (y)
    y = data.y
    norm_dict['y_min'] = y.min(dim=0).values
    norm_dict['y_max'] = y.max(dim=0).values
    norm_dict['y_dim'] = y.shape[1]
    
    return norm_dict


def normalize_tensor(tensor, min_vals, max_vals, eps=1e-8):
    """Normalize a tensor to [0, 1] range."""
    range_vals = max_vals - min_vals
    range_vals = torch.clamp(range_vals, min=eps)
    return (tensor - min_vals) / range_vals


def denormalize_tensor(tensor, min_vals, max_vals):
    """Denormalize a tensor back to original scale."""
    return tensor * (max_vals - min_vals) + min_vals


def normalize_data(data, norm_dict):
    """Normalize a single data sample in-place."""
    data.x = normalize_tensor(data.x, norm_dict['x_min'], norm_dict['x_max'])
    
    if data.edge_attr is not None and 'edge_attr_min' in norm_dict:
        data.edge_attr = normalize_tensor(
            data.edge_attr, 
            norm_dict['edge_attr_min'], 
            norm_dict['edge_attr_max']
        )
    
    data.y = normalize_tensor(data.y, norm_dict['y_min'], norm_dict['y_max'])


def normalize_dataset(dataset, analysis='linear'):
    """
    Normalize the entire dataset.
    
    Returns:
        tuple: (normalized_dataset, norm_dict)
    """
    norm_dict = get_global_stats(dataset)
    
    for data in dataset:
        normalize_data(data, norm_dict)
    
    print(f"  ✓ Normalized {len(dataset)} samples")
    print(f"    Input features: {norm_dict['x_dim']}")
    print(f"    Output features: {norm_dict['y_dim']}")
    
    return dataset, norm_dict


def normalize_dataset_byNormDict(dataset, norm_dict, analysis='linear'):
    """Normalize a dataset using existing normalization parameters."""
    for data in dataset:
        normalize_data(data, norm_dict)
    return dataset


def denormalize_y(y, norm_dict):
    """Denormalize output predictions back to original scale."""
    device = y.device
    y_min = norm_dict['y_min'].to(device)
    y_max = norm_dict['y_max'].to(device)
    return denormalize_tensor(y, y_min, y_max)


def denormalize_y_linear(y, norm_dict):
    """Alias for denormalize_y."""
    return denormalize_y(y, norm_dict)


def denormalize_grid_num(data, norm_dict):
    """Denormalize grid numbers (first 3 features)."""
    device = data.device
    return denormalize_tensor(
        data, 
        norm_dict['x_min'][:3].to(device), 
        norm_dict['x_max'][:3].to(device)
    )


def denormalize_disp(disp, norm_dict):
    """Denormalize displacement (outputs 0:2)."""
    device = disp.device
    return denormalize_tensor(
        disp, 
        norm_dict['y_min'][:2].to(device), 
        norm_dict['y_max'][:2].to(device)
    )


def print_norm_dict(norm_dict):
    """Pretty print normalization info."""
    print("\n  Normalization Statistics:")
    print("  " + "─" * 50)
    print(f"  Input (x):  {norm_dict.get('x_dim', 'N/A')} features")
    print(f"  Output (y): {norm_dict.get('y_dim', 'N/A')} features")
    print(f"  Edges:      {norm_dict.get('edge_attr_dim', 'N/A')} features")