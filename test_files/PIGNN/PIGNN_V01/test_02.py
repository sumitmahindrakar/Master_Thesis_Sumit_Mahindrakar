import pickle


with open("test_files/PIGNN/PIGNN_V01/DATA/frame_dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

print("Keys in case 1:", list(dataset[0].keys()))
print("\nPer-case node/element counts:")
for c in dataset:
    n = c['n_nodes']
    e = c['n_elements']
    resp_node = c['nearest_node_id']
    in_range = 0 <= resp_node < n
    print(f"  Case {c['case_num']}: "
          f"nodes={n}, elements={e}, "
          f"response_node={resp_node}, "
          f"in_range={in_range}")