"""
GNN Package

This package contains Graph Neural Network components for structural analysis:
- layers: Building blocks (MLP, GraphNetwork_layer)
- losses: Custom loss functions (L1_Loss, L2_Loss)
- models: Complete GNN architectures (Structure_GraphNetwork, Structure_GCN, etc.)

Usage:
    from GNN import Structure_GraphNetwork, L1_Loss
    from GNN.layers import MLP, GraphNetwork_layer
    from GNN.models import Structure_GCN, Structure_GAT, Structure_GIN
"""

# Import from layers
from .layers import MLP, GraphNetwork_layer

# Import from losses
from .losses import L1_Loss, L2_Loss

# Import from models
from .models import (
    Structure_GraphNetwork,
    Structure_GCN,
    Structure_GAT,
    Structure_GIN,
    Structure_GraphNetwork_pseudo,
    Structure_GCN_pseudo,
    Structure_GAT_pseudo,
    Structure_GIN_pseudo
)

from .models import (
    Structure_GraphNetwork_pseudo,
    Structure_GCN_pseudo,
    Structure_GAT_pseudo,
    Structure_GIN_pseudo
)

# Define what gets exported with "from GNN import *"
__all__ = [
    # Layers
    'MLP',
    'GraphNetwork_layer',
    
    # Losses
    'L1_Loss',
    'L2_Loss',
    
    # Models
    'Structure_GraphNetwork',
    'Structure_GCN',
    'Structure_GAT',
    'Structure_GIN',

    'Structure_GraphNetwork_pseudo',
    'Structure_GCN_pseudo',
    'Structure_GAT_pseudo',
    'Structure_GIN_pseudo'
]

# Package version
__version__ = '1.0.0'