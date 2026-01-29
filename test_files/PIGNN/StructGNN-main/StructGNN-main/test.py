import os

base = "Data/Static_Linear_Analysis"
count = 0

for i in range(1, 101):
    f = os.path.join(base, f"structure_{i}", "structure_graph_NodeAsNode.pt")
    if os.path.exists(f):
        count += 1

print("Found graphs:", count)

import os
print("CWD:", os.getcwd())

