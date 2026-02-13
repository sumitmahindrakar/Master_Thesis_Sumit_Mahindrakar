"""
Utils/datasets.py

Functions for loading and managing structural analysis datasets.

Main functions:
- get_dataset: Load graph data from .pt files
- split_dataset: Split into train/validation/test sets
- get_target_index: Map target names to output column indices
"""

import torch
from torch.utils.data import random_split
import os


def get_dataset(dataset_name='Static_Linear_Analysis', whatAsNode='NodeAsNode', 
                structure_num=300, special_path=None):
    """
    Load structural graph dataset from .pt files.
    
    Each structure is stored as a PyTorch Geometric Data object containing:
    - x: Node features (coordinates, forces, boundary conditions, etc.)
    - edge_index: Graph connectivity (which nodes are connected)
    - edge_attr: Edge attributes (beam properties)
    - y: Target outputs (displacements, moments, shears)
    
    Args:
        dataset_name: Name of the dataset folder (default: 'Static_Linear_Analysis')
        whatAsNode: Type of node representation (default: 'NodeAsNode')
                    Options: 'NodeAsNode', 'NodeAsNode_pseudo'
        structure_num: Number of structures to load (default: 300)
        special_path: Custom path to data (overrides default)
    
    Returns:
        list: List of PyTorch Geometric Data objects
    
    Expected folder structure:
        Data/
        └── {dataset_name}/
            ├── structure_1/
            │   └── structure_graph_{whatAsNode}.pt
            ├── structure_2/
            │   └── structure_graph_{whatAsNode}.pt
            └── ...
    
    Example:
        >>> dataset = get_dataset(dataset_name='Static_Linear_Analysis',
        ...                       whatAsNode='NodeAsNode',
        ...                       structure_num=100)
        >>> print(f"Loaded {len(dataset)} structures")
        >>> print(dataset[0])  # First structure's graph
    """
    # Determine the root path
    if special_path is not None:
        root = special_path
    else:
        root = 'Data/' + dataset_name + '/'
    
    data_list = []
    loaded_count = 0
    missing_count = 0
    
    # Load each structure
    for index in range(1, structure_num + 1):
        # Construct file path
        folder_name = root + 'structure_' + str(index) + '/'
        structure_graph_path = folder_name + 'structure_graph_' + whatAsNode + '.pt'
        
        try:
            # Load the graph
            graph = torch.load(structure_graph_path)
            data_list.append(graph)
            loaded_count += 1
        except FileNotFoundError:
            print(f"  [Warning] File not found: {structure_graph_path}")
            missing_count += 1
        except Exception as e:
            print(f"  [Error] Failed to load {structure_graph_path}: {e}")
            missing_count += 1
    
    if loaded_count == 0:
        print(f"\n  [ERROR] No data files found in {root}")
        print(f"  Expected structure: {root}structure_1/structure_graph_{whatAsNode}.pt")
    else:
        print(f"  Loaded {loaded_count} structures" + 
              (f" ({missing_count} missing)" if missing_count > 0 else ""))
    
    return data_list


def split_dataset(dataset, train_ratio=0.9, valid_ratio=0.1, test_ratio=None):
    """
    Split dataset into training, validation, and optionally test sets.
    
    Uses a fixed random seed (731) for reproducibility.
    
    Args:
        dataset: List of data samples (or any iterable with length)
        train_ratio: Fraction of data for training (default: 0.9)
        valid_ratio: Fraction of data for validation (default: 0.1)
        test_ratio: Fraction of data for testing (default: None)
                    If None, only train/valid split is performed
    
    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
               test_dataset is None if test_ratio is None
    
    Example:
        >>> dataset = get_dataset(structure_num=100)
        >>> train, valid, test = split_dataset(dataset, 
        ...                                     train_ratio=0.8, 
        ...                                     valid_ratio=0.1,
        ...                                     test_ratio=0.1)
        >>> print(f"Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")
        Train: 80, Valid: 10, Test: 10
    """
    # Get dataset length
    if hasattr(dataset, '__len__'):
        length = len(dataset)
    else:
        length = dataset.len()  # For some PyTorch Geometric datasets
    
    if test_ratio is None:
        # Two-way split: train and validation only
        train_len = int(length * train_ratio)
        valid_len = length - train_len
        
        train_dataset, valid_dataset = random_split(
            dataset,
            [train_len, valid_len],
            generator=torch.Generator().manual_seed(731)  # Fixed seed for reproducibility
        )
        
        return train_dataset, valid_dataset, None
    
    else:
        # Three-way split: train, validation, and test
        train_len = int(length * train_ratio)
        valid_len = int(length * valid_ratio)
        test_len = length - train_len - valid_len
        
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset,
            [train_len, valid_len, test_len],
            generator=torch.Generator().manual_seed(731)  # Fixed seed for reproducibility
        )
        
        return train_dataset, valid_dataset, test_dataset


def get_target_index(target):
    """
    Get the column indices for a specific output target.
    
    The model outputs 26 values per node (or 38 in some cases):
    - Columns 0-1:   Displacement (X, Z)
    - Columns 2-7:   Moment Y (6 values for different positions)
    - Columns 8-13:  Moment Z (6 values)
    - Columns 14-19: Shear Y (6 values)
    - Columns 20-25: Shear Z (6 values)
    - Columns 26-31: Axial Force (6 values) [optional]
    - Columns 32-37: Torsion (6 values) [optional]
    
    Args:
        target: Name of the target output. Options:
                'disp_x', 'disp_z', 'disp' - Displacements
                'momentY', 'momentZ', 'moment' - Bending moments
                'shearY', 'shearZ', 'shear' - Shear forces
                'axialForce' - Axial forces
                'torsion' - Torsional moments
                'all' - All outputs (columns 0-25)
    
    Returns:
        tuple: (start_index, end_index) for slicing
               Use as: output[:, start:end]
    
    Example:
        >>> y_start, y_end = get_target_index('displacement')
        >>> displacement_prediction = model_output[:, y_start:y_end]
    """
    # Define mapping of target names to column ranges
    target_mapping = {
        # Displacements
        'disp_x': (0, 1),       # Just X displacement
        'disp_z': (1, 2),       # Just Z displacement
        'disp': (0, 2),         # Both displacements
        
        # Moments
        'momentY': (2, 8),      # Moment about Y axis (6 values)
        'momentZ': (8, 14),     # Moment about Z axis (6 values)
        'moment': (2, 14),      # All moments (12 values)
        
        # Shears
        'shearY': (14, 20),     # Shear in Y direction (6 values)
        'shearZ': (20, 26),     # Shear in Z direction (6 values)
        'shear': (14, 26),      # All shears (12 values)
        
        # Additional outputs (if available in data)
        'axialForce': (26, 32), # Axial forces (6 values)
        'torsion': (32, 38),    # Torsional moments (6 values)
        
        # All primary outputs
        'all': (0, 26)          # Displacement + Moment + Shear
    }
    
    if target not in target_mapping:
        available = ', '.join(target_mapping.keys())
        raise ValueError(f"Unknown target: '{target}'. Available options: {available}")
    
    return target_mapping[target]


def get_target_names():
    """
    Get a list of all available target names.
    
    Returns:
        list: Available target names
    """
    return [
        'disp_x', 'disp_z', 'disp',
        'momentY', 'momentZ', 'moment',
        'shearY', 'shearZ', 'shear',
        'axialForce', 'torsion',
        'all'
    ]


def describe_dataset(dataset):
    """
    Print information about a loaded dataset.
    
    Args:
        dataset: List of PyTorch Geometric Data objects
    """
    if len(dataset) == 0:
        print("  Dataset is empty!")
        return
    
    # Get first sample for structure info
    sample = dataset[0]
    
    print(f"\n  Dataset Information:")
    print(f"  {'─' * 40}")
    print(f"  Number of structures: {len(dataset)}")
    print(f"  ")
    print(f"  Sample structure (first one):")
    print(f"    Nodes: {sample.x.shape[0]}")
    print(f"    Node features (x): {sample.x.shape}")
    print(f"    Edges: {sample.edge_index.shape[1]}")
    print(f"    Edge features: {sample.edge_attr.shape}")
    print(f"    Outputs (y): {sample.y.shape}")
    print(f"  ")
    print(f"  Node feature breakdown:")
    print(f"    [0:3]   Grid numbers")
    print(f"    [3:6]   Coordinates (x, y, z)")
    print(f"    [6:8]   Boundary conditions")
    print(f"    [8]     Mass")
    print(f"    [9:15]  Applied forces")
    print(f"  ")
    print(f"  Output breakdown:")
    print(f"    [0:2]   Displacement (X, Z)")
    print(f"    [2:8]   Moment Y")
    print(f"    [8:14]  Moment Z")
    print(f"    [14:20] Shear Y")
    print(f"    [20:26] Shear Z")