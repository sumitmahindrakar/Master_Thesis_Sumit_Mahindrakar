import torch

import os
from pathlib import Path
CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)

# # Fix normalized data
# data_list = torch.load("DATA/graph_dataset_norm.pt", weights_only=False)
# data_list = [d.cpu() for d in data_list]
# torch.save(data_list, "DATA/graph_dataset_norm.pt")
# print(f"Fixed graph_dataset_norm.pt: {len(data_list)} graphs on CPU")

# # Fix raw data
# raw_data = torch.load("DATA/graph_dataset.pt", weights_only=False)
# raw_data = [d.cpu() for d in raw_data]
# torch.save(raw_data, "DATA/graph_dataset.pt")
# print(f"Fixed graph_dataset.pt: {len(raw_data)} graphs on CPU")

import torch
from torch_geometric.data import Data

def force_cpu(data):
    """Move ALL tensors in a PyG Data to CPU."""
    for key in data.keys():
        val = data[key]
        if isinstance(val, torch.Tensor):
            data[key] = val.cpu()
    return data

# Fix both files
for path in ["DATA/graph_dataset_norm.pt", 
             "DATA/graph_dataset.pt"]:
    data_list = torch.load(path, weights_only=False)
    
    # Debug: find CUDA tensors
    for i, d in enumerate(data_list):
        for key in d.keys():
            val = d[key]
            if isinstance(val, torch.Tensor) and val.is_cuda:
                print(f"  Graph {i}, key '{key}': "
                      f"device={val.device}")
        data_list[i] = force_cpu(d)
    
    torch.save(data_list, path)
    print(f"Fixed {path}")