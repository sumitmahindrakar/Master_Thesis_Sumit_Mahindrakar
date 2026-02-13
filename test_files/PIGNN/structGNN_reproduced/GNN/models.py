import torch
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F
from .layers import MLP, GraphNetwork_layer

class Structure_GraphNetwork(torch.nn.Module):

    def __init__(self, layer_num, input_dim, hidden_dim, edge_attr_dim, aggr,
                 gnn_act=True, gnn_dropout=True, dropout_p=0.0,
                 node_out_dispX_dim=1, node_out_dispZ_dim=1,
                 node_out_momentY_dim=6, node_out_momentZ_dim=6,
                 node_out_shearY_dim=6, node_out_shearZ_dim=6,
                 device="cuda",
                 **kwargs):
        super(Structure_GraphNetwork, self).__init__()

        self.layer_num=layer_num
        self.gnn_act=gnn_act
        self.gnn_dropout=gnn_dropout
        self.dropout_p=dropout_p
        self.device=device

        self.encoder = MLP(input_dim, [], hidden_dim, act=False, dropout=False)


        #GNN layer: massage passing with edge attributes
        self.conv_layer = GraphNetwork_layer(
            hidden_dim,
            hidden_dim,
            aggr=aggr,
            edge_attr_dim=edge_attr_dim
        )

        #decoder
        self.node_dispX_decoder=MLP(hidden_dim,[64], node_out_dispX_dim, act=True, dropout=False)
        self.node_dispZ_decoder=MLP(hidden_dim,[64], node_out_dispZ_dim, act=True, dropout=False)
        self.node_momentY_decoder=MLP(hidden_dim,[64], node_out_momentY_dim, act=True, dropout=False)
        self.node_momentZ_decoder=MLP(hidden_dim,[64], node_out_momentZ_dim, act=True, dropout=False)
        self.node_shearY_decoder=MLP(hidden_dim,[64], node_out_shearY_dim, act=True, dropout=False)
        self.node_shearZ_decoder=MLP(hidden_dim,[64], node_out_shearZ_dim, act=True, dropout=False)

    def forward(self, x, edge_index, edge_attr):
        x = self.encoder(x)

        for i in range(self.layer_num):
            x = self.conv_layer(x, edge_index, edge_attr)

            if self.gnn_act:
                x = F.relu(x)

            if self.gnn_dropout:
                x = F.dropout(x, p=self.dropout_p, training=self.training)

        node_out_dispX=self.node_dispX_decoder(x)
        node_out_dispZ=self.node_dispZ_decoder(x)
        node_out_momentY=self.node_momentY_decoder(x)
        node_out_momentZ=self.node_momentZ_decoder(x)
        node_out_shearY=self.node_shearY_decoder(x)
        node_out_shearZ=self.node_shearZ_decoder(x)

        node_out = torch.zeros(size=(x.shape[0], 26)).to(self.device)
        node_out[:, 0] = node_out_dispX[:, 0]
        node_out[:, 1] = node_out_dispZ[:, 0]
        node_out[:, 2:8] = node_out_momentY[:, :]
        node_out[:, 8:14] = node_out_momentZ[:, :]
        node_out[:, 14:20] = node_out_shearY[:, :]
        node_out[:, 20:26] = node_out_shearZ[:, :]

        return node_out


class Structure_GCN(torch.nn.Module):
    """
    Graph Convolutional Network for structural analysis.
    
    Uses standard GCNConv layers (does NOT use edge attributes).
    This is a baseline model for comparison.
    
    Note: GCN ignores edge attributes, so it may be less accurate
    for structural analysis where beam properties matter.
    """
    
    def __init__(self, layer_num, input_dim, hidden_dim, aggr, edge_attr_dim,
                 gnn_act=True, gnn_dropout=True, dropout_p=0.0,
                 node_out_dispX_dim=1, node_out_dispZ_dim=1,
                 node_out_momentY_dim=6, node_out_momentZ_dim=6,
                 node_out_shearY_dim=6, node_out_shearZ_dim=6,
                 device="cuda", 
                 **kwargs):
        super(Structure_GCN, self).__init__()
        
        self.layer_num = layer_num
        self.gnn_act = gnn_act
        self.gnn_dropout = gnn_dropout
        self.dropout_p = dropout_p
        self.device = device

        # Encoder
        self.encoder = MLP(input_dim, [], hidden_dim, act=False, dropout=False)
        
        # Standard GCN layer (no edge attributes)
        self.conv_layer = tg.nn.GCNConv(hidden_dim, hidden_dim, aggr=aggr)
        
        # Decoders
        self.node_dispX_decoder = MLP(hidden_dim, [64], node_out_dispX_dim, act=True, dropout=False)
        self.node_dispZ_decoder = MLP(hidden_dim, [64], node_out_dispZ_dim, act=True, dropout=False)
        self.node_momentY_decoder = MLP(hidden_dim, [64], node_out_momentY_dim, act=True, dropout=False)
        self.node_momentZ_decoder = MLP(hidden_dim, [64], node_out_momentZ_dim, act=True, dropout=False)
        self.node_shearY_decoder = MLP(hidden_dim, [64], node_out_shearY_dim, act=True, dropout=False)
        self.node_shearZ_decoder = MLP(hidden_dim, [64], node_out_shearZ_dim, act=True, dropout=False)

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass (edge_attr is ignored in GCN).
        """
        x = self.encoder(x)
        
        for i in range(self.layer_num):
            x = self.conv_layer(x, edge_index)  # Note: no edge_attr
            if self.gnn_act:
                x = F.relu(x)
            if self.gnn_dropout:
                x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        # Decode outputs
        node_out_dispX = self.node_dispX_decoder(x)
        node_out_dispZ = self.node_dispZ_decoder(x)
        node_out_momentY = self.node_momentY_decoder(x)
        node_out_momentZ = self.node_momentZ_decoder(x)
        node_out_shearY = self.node_shearY_decoder(x)
        node_out_shearZ = self.node_shearZ_decoder(x)
        
        node_out = torch.zeros(size=(x.shape[0], 26)).to(self.device)
        node_out[:, 0] = node_out_dispX[:, 0]
        node_out[:, 1] = node_out_dispZ[:, 0]
        node_out[:, 2:8] = node_out_momentY[:, :]
        node_out[:, 8:14] = node_out_momentZ[:, :]
        node_out[:, 14:20] = node_out_shearY[:, :]
        node_out[:, 20:26] = node_out_shearZ[:, :]
        
        return node_out


class Structure_GAT(torch.nn.Module):
    """
    Graph Attention Network for structural analysis.
    
    Uses attention mechanism to weight neighbor contributions.
    Does NOT use edge attributes directly.
    """
    
    def __init__(self, layer_num, input_dim, hidden_dim, aggr, edge_attr_dim,
                 gnn_act=True, gnn_dropout=True, dropout_p=0.0,
                 node_out_dispX_dim=1, node_out_dispZ_dim=1,
                 node_out_momentY_dim=6, node_out_momentZ_dim=6,
                 node_out_shearY_dim=6, node_out_shearZ_dim=6,
                 device="cuda", 
                 **kwargs):
        super(Structure_GAT, self).__init__()
        
        self.layer_num = layer_num
        self.gnn_act = gnn_act
        self.gnn_dropout = gnn_dropout
        self.dropout_p = dropout_p
        self.device = device

        self.encoder = MLP(input_dim, [], hidden_dim, act=False, dropout=False)
        self.conv_layer = tg.nn.GATConv(hidden_dim, hidden_dim, aggr=aggr)
        
        self.node_dispX_decoder = MLP(hidden_dim, [64], node_out_dispX_dim, act=True, dropout=False)
        self.node_dispZ_decoder = MLP(hidden_dim, [64], node_out_dispZ_dim, act=True, dropout=False)
        self.node_momentY_decoder = MLP(hidden_dim, [64], node_out_momentY_dim, act=True, dropout=False)
        self.node_momentZ_decoder = MLP(hidden_dim, [64], node_out_momentZ_dim, act=True, dropout=False)
        self.node_shearY_decoder = MLP(hidden_dim, [64], node_out_shearY_dim, act=True, dropout=False)
        self.node_shearZ_decoder = MLP(hidden_dim, [64], node_out_shearZ_dim, act=True, dropout=False)

    def forward(self, x, edge_index, edge_attr):
        x = self.encoder(x)
        
        for i in range(self.layer_num):
            x = self.conv_layer(x, edge_index)
            if self.gnn_act:
                x = F.relu(x)
            if self.gnn_dropout:
                x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        node_out_dispX = self.node_dispX_decoder(x)
        node_out_dispZ = self.node_dispZ_decoder(x)
        node_out_momentY = self.node_momentY_decoder(x)
        node_out_momentZ = self.node_momentZ_decoder(x)
        node_out_shearY = self.node_shearY_decoder(x)
        node_out_shearZ = self.node_shearZ_decoder(x)
        
        node_out = torch.zeros(size=(x.shape[0], 26)).to(self.device)
        node_out[:, 0] = node_out_dispX[:, 0]
        node_out[:, 1] = node_out_dispZ[:, 0]
        node_out[:, 2:8] = node_out_momentY[:, :]
        node_out[:, 8:14] = node_out_momentZ[:, :]
        node_out[:, 14:20] = node_out_shearY[:, :]
        node_out[:, 20:26] = node_out_shearZ[:, :]
        
        return node_out


class Structure_GIN(torch.nn.Module):
    """
    Graph Isomorphism Network for structural analysis.
    
    GIN is provably as powerful as the Weisfeiler-Lehman graph isomorphism test.
    Does NOT use edge attributes directly.
    """
    
    def __init__(self, layer_num, input_dim, hidden_dim, aggr, edge_attr_dim,
                 gnn_act=True, gnn_dropout=True, dropout_p=0.0,
                 node_out_dispX_dim=1, node_out_dispZ_dim=1,
                 node_out_momentY_dim=6, node_out_momentZ_dim=6,
                 node_out_shearY_dim=6, node_out_shearZ_dim=6,
                 device="cuda", 
                 **kwargs):
        super(Structure_GIN, self).__init__()
        
        self.layer_num = layer_num
        self.gnn_act = gnn_act
        self.gnn_dropout = gnn_dropout
        self.dropout_p = dropout_p
        self.device = device

        self.encoder = MLP(input_dim, [], hidden_dim, act=False, dropout=False)
        
        # GIN requires a neural network for transformation
        self.conv_nn = nn.Linear(hidden_dim, hidden_dim)
        self.conv_layer = tg.nn.GINConv(self.conv_nn, aggr=aggr)
        
        self.node_dispX_decoder = MLP(hidden_dim, [64], node_out_dispX_dim, act=True, dropout=False)
        self.node_dispZ_decoder = MLP(hidden_dim, [64], node_out_dispZ_dim, act=True, dropout=False)
        self.node_momentY_decoder = MLP(hidden_dim, [64], node_out_momentY_dim, act=True, dropout=False)
        self.node_momentZ_decoder = MLP(hidden_dim, [64], node_out_momentZ_dim, act=True, dropout=False)
        self.node_shearY_decoder = MLP(hidden_dim, [64], node_out_shearY_dim, act=True, dropout=False)
        self.node_shearZ_decoder = MLP(hidden_dim, [64], node_out_shearZ_dim, act=True, dropout=False)

    def forward(self, x, edge_index, edge_attr):
        x = self.encoder(x)
        
        for i in range(self.layer_num):
            x = self.conv_layer(x, edge_index)
            if self.gnn_act:
                x = F.relu(x)
            if self.gnn_dropout:
                x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        node_out_dispX = self.node_dispX_decoder(x)
        node_out_dispZ = self.node_dispZ_decoder(x)
        node_out_momentY = self.node_momentY_decoder(x)
        node_out_momentZ = self.node_momentZ_decoder(x)
        node_out_shearY = self.node_shearY_decoder(x)
        node_out_shearZ = self.node_shearZ_decoder(x)
        
        node_out = torch.zeros(size=(x.shape[0], 26)).to(self.device)
        node_out[:, 0] = node_out_dispX[:, 0]
        node_out[:, 1] = node_out_dispZ[:, 0]
        node_out[:, 2:8] = node_out_momentY[:, :]
        node_out[:, 8:14] = node_out_momentZ[:, :]
        node_out[:, 14:20] = node_out_shearY[:, :]
        node_out[:, 20:26] = node_out_shearZ[:, :]
        
        return node_out
    

######### PSUEDO METHOD ##########

class Structure_GraphNetwork_pseudo(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, edge_attr_dim, aggr,
                 gnn_act=True, gnn_dropout=True, dropout_p=0.0,
                 node_out_dispX_dim=1, node_out_dispZ_dim=1,
                 node_out_momentY_dim=6, node_out_momentZ_dim=6,
                 node_out_shearY_dim=6, node_out_shearZ_dim=6,
                 device="cuda",
                 **kwargs):
        super(Structure_GraphNetwork_pseudo, self).__init__()

        # self.layer_num=layer_num
        self.gnn_act=gnn_act
        self.gnn_dropout=gnn_dropout
        self.dropout_p=dropout_p
        self.device=device

        self.encoder = MLP(input_dim, [], hidden_dim, act=False, dropout=False)


        #GNN layer: massage passing with edge attributes
        self.conv_layer = GraphNetwork_layer(
            hidden_dim,
            hidden_dim,
            aggr=aggr,
            edge_attr_dim=edge_attr_dim
        )

        #decoder
        self.node_dispX_decoder=MLP(hidden_dim,[64], node_out_dispX_dim, act=True, dropout=False)
        self.node_dispZ_decoder=MLP(hidden_dim,[64], node_out_dispZ_dim, act=True, dropout=False)
        self.node_momentY_decoder=MLP(hidden_dim,[64], node_out_momentY_dim, act=True, dropout=False)
        self.node_momentZ_decoder=MLP(hidden_dim,[64], node_out_momentZ_dim, act=True, dropout=False)
        self.node_shearY_decoder=MLP(hidden_dim,[64], node_out_shearY_dim, act=True, dropout=False)
        self.node_shearZ_decoder=MLP(hidden_dim,[64], node_out_shearZ_dim, act=True, dropout=False)

    def forward(self, x, edge_index, edge_attr, layer_num):
        x = self.encoder(x)

        for i in range(layer_num):# layer taken as custom not default from global structure
            x = self.conv_layer(x, edge_index, edge_attr)

            if self.gnn_act:
                x = F.relu(x)

            if self.gnn_dropout:
                x = F.dropout(x, p=self.dropout_p, training=self.training)

        node_out_dispX=self.node_dispX_decoder(x)
        node_out_dispZ=self.node_dispZ_decoder(x)
        node_out_momentY=self.node_momentY_decoder(x)
        node_out_momentZ=self.node_momentZ_decoder(x)
        node_out_shearY=self.node_shearY_decoder(x)
        node_out_shearZ=self.node_shearZ_decoder(x)

        node_out = torch.zeros(size=(x.shape[0], 26)).to(self.device)
        node_out[:, 0] = node_out_dispX[:, 0]
        node_out[:, 1] = node_out_dispZ[:, 0]
        node_out[:, 2:8] = node_out_momentY[:, :]
        node_out[:, 8:14] = node_out_momentZ[:, :]
        node_out[:, 14:20] = node_out_shearY[:, :]
        node_out[:, 20:26] = node_out_shearZ[:, :]

        return node_out


class Structure_GCN_pseudo(torch.nn.Module):
    """
    Graph Convolutional Network for structural analysis.
    
    Uses standard GCNConv layers (does NOT use edge attributes).
    This is a baseline model for comparison.
    
    Note: GCN ignores edge attributes, so it may be less accurate
    for structural analysis where beam properties matter.
    """
    
    def __init__(self, input_dim, hidden_dim, aggr, edge_attr_dim,
                 gnn_act=True, gnn_dropout=True, dropout_p=0.0,
                 node_out_dispX_dim=1, node_out_dispZ_dim=1,
                 node_out_momentY_dim=6, node_out_momentZ_dim=6,
                 node_out_shearY_dim=6, node_out_shearZ_dim=6,
                 device="cuda", 
                 **kwargs):
        super(Structure_GCN_pseudo, self).__init__()
        
        # self.layer_num = layer_num
        self.gnn_act = gnn_act
        self.gnn_dropout = gnn_dropout
        self.dropout_p = dropout_p
        self.device = device

        # Encoder
        self.encoder = MLP(input_dim, [], hidden_dim, act=False, dropout=False)
        
        # Standard GCN layer (no edge attributes)
        self.conv_layer = tg.nn.GCNConv(hidden_dim, hidden_dim, aggr=aggr)
        
        # Decoders
        self.node_dispX_decoder = MLP(hidden_dim, [64], node_out_dispX_dim, act=True, dropout=False)
        self.node_dispZ_decoder = MLP(hidden_dim, [64], node_out_dispZ_dim, act=True, dropout=False)
        self.node_momentY_decoder = MLP(hidden_dim, [64], node_out_momentY_dim, act=True, dropout=False)
        self.node_momentZ_decoder = MLP(hidden_dim, [64], node_out_momentZ_dim, act=True, dropout=False)
        self.node_shearY_decoder = MLP(hidden_dim, [64], node_out_shearY_dim, act=True, dropout=False)
        self.node_shearZ_decoder = MLP(hidden_dim, [64], node_out_shearZ_dim, act=True, dropout=False)

    def forward(self, x, edge_index, edge_attr, layer_num):
        """
        Forward pass (edge_attr is ignored in GCN).
        """
        x = self.encoder(x)
        
        for i in range(layer_num):# layer taken as custom not default from global structure
            x = self.conv_layer(x, edge_index)  # Note: no edge_attr
            if self.gnn_act:
                x = F.relu(x)
            if self.gnn_dropout:
                x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        # Decode outputs
        node_out_dispX = self.node_dispX_decoder(x)
        node_out_dispZ = self.node_dispZ_decoder(x)
        node_out_momentY = self.node_momentY_decoder(x)
        node_out_momentZ = self.node_momentZ_decoder(x)
        node_out_shearY = self.node_shearY_decoder(x)
        node_out_shearZ = self.node_shearZ_decoder(x)
        
        node_out = torch.zeros(size=(x.shape[0], 26)).to(self.device)
        node_out[:, 0] = node_out_dispX[:, 0]
        node_out[:, 1] = node_out_dispZ[:, 0]
        node_out[:, 2:8] = node_out_momentY[:, :]
        node_out[:, 8:14] = node_out_momentZ[:, :]
        node_out[:, 14:20] = node_out_shearY[:, :]
        node_out[:, 20:26] = node_out_shearZ[:, :]
        
        return node_out


class Structure_GAT_pseudo(torch.nn.Module):
    """
    Graph Attention Network for structural analysis.
    
    Uses attention mechanism to weight neighbor contributions.
    Does NOT use edge attributes directly.
    """
    
    def __init__(self, input_dim, hidden_dim, aggr, edge_attr_dim,
                 gnn_act=True, gnn_dropout=True, dropout_p=0.0,
                 node_out_dispX_dim=1, node_out_dispZ_dim=1,
                 node_out_momentY_dim=6, node_out_momentZ_dim=6,
                 node_out_shearY_dim=6, node_out_shearZ_dim=6,
                 device="cuda", 
                 **kwargs):
        super(Structure_GAT_pseudo, self).__init__()
        
        # self.layer_num = layer_num
        self.gnn_act = gnn_act
        self.gnn_dropout = gnn_dropout
        self.dropout_p = dropout_p
        self.device = device

        self.encoder = MLP(input_dim, [], hidden_dim, act=False, dropout=False)
        self.conv_layer = tg.nn.GATConv(hidden_dim, hidden_dim, aggr=aggr)
        
        self.node_dispX_decoder = MLP(hidden_dim, [64], node_out_dispX_dim, act=True, dropout=False)
        self.node_dispZ_decoder = MLP(hidden_dim, [64], node_out_dispZ_dim, act=True, dropout=False)
        self.node_momentY_decoder = MLP(hidden_dim, [64], node_out_momentY_dim, act=True, dropout=False)
        self.node_momentZ_decoder = MLP(hidden_dim, [64], node_out_momentZ_dim, act=True, dropout=False)
        self.node_shearY_decoder = MLP(hidden_dim, [64], node_out_shearY_dim, act=True, dropout=False)
        self.node_shearZ_decoder = MLP(hidden_dim, [64], node_out_shearZ_dim, act=True, dropout=False)

    def forward(self, x, edge_index, edge_attr, layer_num):
        x = self.encoder(x)
        
        for i in range(layer_num):# layer taken as custom not default from global structure
            x = self.conv_layer(x, edge_index)
            if self.gnn_act:
                x = F.relu(x)
            if self.gnn_dropout:
                x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        node_out_dispX = self.node_dispX_decoder(x)
        node_out_dispZ = self.node_dispZ_decoder(x)
        node_out_momentY = self.node_momentY_decoder(x)
        node_out_momentZ = self.node_momentZ_decoder(x)
        node_out_shearY = self.node_shearY_decoder(x)
        node_out_shearZ = self.node_shearZ_decoder(x)
        
        node_out = torch.zeros(size=(x.shape[0], 26)).to(self.device)
        node_out[:, 0] = node_out_dispX[:, 0]
        node_out[:, 1] = node_out_dispZ[:, 0]
        node_out[:, 2:8] = node_out_momentY[:, :]
        node_out[:, 8:14] = node_out_momentZ[:, :]
        node_out[:, 14:20] = node_out_shearY[:, :]
        node_out[:, 20:26] = node_out_shearZ[:, :]
        
        return node_out


class Structure_GIN_pseudo(torch.nn.Module):
    """
    Graph Isomorphism Network for structural analysis.
    
    GIN is provably as powerful as the Weisfeiler-Lehman graph isomorphism test.
    Does NOT use edge attributes directly.
    """
    
    def __init__(self, input_dim, hidden_dim, aggr, edge_attr_dim,
                 gnn_act=True, gnn_dropout=True, dropout_p=0.0,
                 node_out_dispX_dim=1, node_out_dispZ_dim=1,
                 node_out_momentY_dim=6, node_out_momentZ_dim=6,
                 node_out_shearY_dim=6, node_out_shearZ_dim=6,
                 device="cuda", 
                 **kwargs):
        super(Structure_GIN_pseudo, self).__init__()
        
        # self.layer_num = layer_num
        self.gnn_act = gnn_act
        self.gnn_dropout = gnn_dropout
        self.dropout_p = dropout_p
        self.device = device

        self.encoder = MLP(input_dim, [], hidden_dim, act=False, dropout=False)
        
        # GIN requires a neural network for transformation
        self.conv_nn = nn.Linear(hidden_dim, hidden_dim)
        self.conv_layer = tg.nn.GINConv(self.conv_nn, aggr=aggr)
        
        self.node_dispX_decoder = MLP(hidden_dim, [64], node_out_dispX_dim, act=True, dropout=False)
        self.node_dispZ_decoder = MLP(hidden_dim, [64], node_out_dispZ_dim, act=True, dropout=False)
        self.node_momentY_decoder = MLP(hidden_dim, [64], node_out_momentY_dim, act=True, dropout=False)
        self.node_momentZ_decoder = MLP(hidden_dim, [64], node_out_momentZ_dim, act=True, dropout=False)
        self.node_shearY_decoder = MLP(hidden_dim, [64], node_out_shearY_dim, act=True, dropout=False)
        self.node_shearZ_decoder = MLP(hidden_dim, [64], node_out_shearZ_dim, act=True, dropout=False)

    def forward(self, x, edge_index, edge_attr, layer_num):
        x = self.encoder(x)
        
        for i in range(layer_num):# layer taken as custom not default from global structure
            x = self.conv_layer(x, edge_index)
            if self.gnn_act:
                x = F.relu(x)
            if self.gnn_dropout:
                x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        node_out_dispX = self.node_dispX_decoder(x)
        node_out_dispZ = self.node_dispZ_decoder(x)
        node_out_momentY = self.node_momentY_decoder(x)
        node_out_momentZ = self.node_momentZ_decoder(x)
        node_out_shearY = self.node_shearY_decoder(x)
        node_out_shearZ = self.node_shearZ_decoder(x)
        
        node_out = torch.zeros(size=(x.shape[0], 26)).to(self.device)
        node_out[:, 0] = node_out_dispX[:, 0]
        node_out[:, 1] = node_out_dispZ[:, 0]
        node_out[:, 2:8] = node_out_momentY[:, :]
        node_out[:, 8:14] = node_out_momentZ[:, :]
        node_out[:, 14:20] = node_out_shearY[:, :]
        node_out[:, 20:26] = node_out_shearZ[:, :]
        
        return node_out