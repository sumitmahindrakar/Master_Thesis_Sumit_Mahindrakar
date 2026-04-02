import os
from pathlib import Path
print(f"Working directory: {os.getcwd()}")
CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)
print(f"Working directory: {os.getcwd()}")
from step_2_grapg_constr import FrameData



import torch

data = torch.load("DATA/graph_dataset.pt")

print(type(data))
print(data[200])
g = data[200]

# for key, value in g.to_dict().items():
#     print(f"\n{key}:")
#     print(value)

# for key, value in g.to_dict().items():
#     print(f"\n----- {key} -----")
    
#     if hasattr(value, "shape"):   # tensor
#         print("shape:", value.shape)
#         print(value[:10])         # first 10 rows
#     else:
#         print(value)

# ===== Node features =====
# print("x (node features):")
# print(g.x)

# # ===== Graph connectivity =====
# print("\nedge_index:")
# print(g.edge_index)

# # ===== Edge features =====
# print("\nedge_attr:")
# print(g.edge_attr)

# # ===== Element loads =====Corrected
# print("\nelem_load:")
# print(g.elem_load) 

# # ===== Node targets =====
# print("\ny_node:")
# print(g.y_node)

# # ===== Element targets =====
# print("\ny_element:")
# print(g.y_element)

# # ===== Element map =====
# print("\nelement_map:")
# print(g.element_map)

# # ===== Element connectivity =====
# print("\nconnectivity:")
# print(g.connectivity)

# # ===== Node coordinates =====
# print("\ncoords:")
# print(g.coords)

# # ===== Element lengths =====
# print("\nelem_lengths:")
# print(g.elem_lengths)

# # ===== Element directions =====
# print("\nelem_directions:")
# print(g.elem_directions)

# # ===== Material properties =====
# print("\nprop_E:")
# print(g.prop_E)

# print("\nprop_A:")
# print(g.prop_A)

# print("\nprop_I22:")
# print(g.prop_I22)

# # ===== Boundary conditions =====
# print("\nbc_disp:")
# print(g.bc_disp)

# print("\nbc_rot:")
# print(g.bc_rot)

# # ===== Face information =====
# print("\nface_mask:")
# print(g.face_mask)

# print("\nface_element_id:") #not validated yet
# print(g.face_element_id)

# print("\nface_is_A_end:")
# print(g.face_is_A_end)

# # ===== External forces =====i think it is correct. issue to be discussed with Prof.
# print("\nF_ext:")# end node have half of element load . UDL*length/2 per element contribution on one node
# print(g.F_ext) # in interior 2 element contribution on 1 node so they are UDL*length/2 *2

# # ===== Metadata =====
# print("\nnum_nodes_val:")
# print(g.num_nodes_val)

# print("\nn_elements:")
# print(g.n_elements)

# print("\ncase_id:")
# print(g.case_id)

# print("\nnearest_node_id:")
# print(g.nearest_node_id)

# print("\ntraced_element_id:")
# print(g.traced_element_id)

# import torch

# data = torch.load("DATA_400mixedcase_2setWithDiffParam/graph_dataset.pt")

# # Initialize containers
# prop_E_all = []
# prop_A_all = []
# prop_I22_all = []
# F_ext_all = []

# for g in data:
#     prop_E_all.append(g.prop_E.view(-1))
#     prop_A_all.append(g.prop_A.view(-1))
#     prop_I22_all.append(g.prop_I22.view(-1))
#     F_ext_all.append(g.F_ext.view(-1))

# # Concatenate everything into single tensors
# prop_E_all = torch.cat(prop_E_all)
# prop_A_all = torch.cat(prop_A_all)
# prop_I22_all = torch.cat(prop_I22_all)
# F_ext_all = torch.cat(F_ext_all)

# def analyze(name, tensor):
#     unique_vals = torch.unique(tensor)
#     print(f"\n===== {name} =====")
#     print(f"Min: {tensor.min().item()}")
#     print(f"Max: {tensor.max().item()}")
#     print(f"Number of unique values: {len(unique_vals)}")
    
#     if len(unique_vals) < 10:
#         print("Unique values:", unique_vals)
#     else:
#         print("Sample unique values:", unique_vals[:10])

# # Run analysis
# analyze("prop_E", prop_E_all)
# analyze("prop_A", prop_A_all)
# analyze("prop_I22", prop_I22_all)
# analyze("F_ext", F_ext_all)


# import torch
# import hashlib

# data = torch.load("DATA_400mixedcase_2setWithDiffParam/graph_dataset.pt")

# def tensor_to_bytes(t):
#     return t.detach().cpu().numpy().tobytes()

# def graph_hash(g):
#     m = hashlib.md5()

#     # Include all important attributes
#     attrs = [
#         g.x, g.edge_index, g.edge_attr,
#         g.elem_load, g.y_node, g.y_element,
#         g.element_map, g.connectivity,
#         g.coords, g.elem_lengths, g.elem_directions,
#         g.prop_E, g.prop_A, g.prop_I22,
#         g.F_ext
#     ]

#     for attr in attrs:
#         if attr is not None:
#             m.update(tensor_to_bytes(attr))

#     return m.hexdigest()

# # Compute hashes
# hashes = [graph_hash(g) for g in data]

# # Find duplicates
# from collections import Counter
# counter = Counter(hashes)

# duplicates = {h: c for h, c in counter.items() if c > 1}

# print(f"Total samples: {len(data)}")
# print(f"Unique samples: {len(counter)}")
# print(f"Number of duplicate groups: {len(duplicates)}")

# if duplicates:
#     print("\nDuplicate counts:")
#     for h, c in duplicates.items():
#         print(f"{h}: {c} times")

import torch
import hashlib

def tensor_to_bytes(t):
    # Optional: rounding to avoid floating precision issues
    t = torch.round(t * 1e6) / 1e6
    return t.detach().cpu().numpy().tobytes()

def graph_hash(g):
    m = hashlib.md5()

    # Only INPUT features (important!)
    attr_names = [
        "coords",
        "connectivity",
        "edge_index",
        "edge_attr",
        "prop_E",
        "prop_A",
        "prop_I22",
        "F_ext",
        "point_moment_My",
        "bc_disp",
        "bc_rot",
        "bc_mask",
        "q_c"
    ]

    for name in attr_names:
        if hasattr(g, name):
            attr = getattr(g, name)

            if torch.is_tensor(attr):
                m.update(tensor_to_bytes(attr))
            else:
                # scalar (like q_c)
                m.update(str(attr).encode())
        else:
            m.update(f"{name}_missing".encode())

    return m.hexdigest()


from collections import defaultdict

hash_map = defaultdict(list)

for i, g in enumerate(data):
    h = graph_hash(g)
    hash_map[h].append(i)

duplicates = [indices for indices in hash_map.values() if len(indices) > 1]

print(f"Total samples: {len(data)}")
print(f"Duplicate groups: {len(duplicates)}")

for group in duplicates:
    print("Duplicate indices:", group)