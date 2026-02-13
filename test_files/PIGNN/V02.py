import torch

obj1 = torch.load("test_files\PIGNN\StructGNN-main\StructGNN-main\Data\Static_Linear_Analysis\structure_1\structure_graph_NodeAsNode.pt", map_location="cpu")
obj2 = torch.load("test_files\PIGNN\StructGNN-main\StructGNN-main\Data\Static_Linear_Analysis\structure_1\structure_graph_NodeAsNode_pseudo.pt", map_location="cpu")


print("Type:", type(obj1))

if isinstance(obj1, dict):
    print("Dictionary keys:", obj1.keys())
elif torch.is_tensor(obj1):
    print("Tensor shape:", obj1.shape)
else:
    print(obj1)

print("normal = ",obj1)
print("\npseudo = ",obj2)

print("normal [x]= ",obj1["x"])
print("pseudo [x]= ",obj2["x"])