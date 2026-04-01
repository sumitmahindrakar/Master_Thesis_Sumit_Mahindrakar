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
print(data[180])
g = data[180]

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
print("\nF_ext:")# end node have half of element load . UDL*length/2 per element contribution on one node
print(g.F_ext) # in interior 2 element contribution on 1 node so they are UDL*length/2 *2

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