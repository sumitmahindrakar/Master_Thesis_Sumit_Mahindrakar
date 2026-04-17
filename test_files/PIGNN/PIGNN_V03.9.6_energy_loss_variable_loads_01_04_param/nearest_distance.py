import os
import pickle
from pathlib import Path

import torch
import numpy as np

CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)

from step_2_grapg_constr import FrameData


def load_data():
    paths = ["DATA/graph_dataset_norm.pt", "DATA/graph_dataset.pt"]
    for p in paths:
        if os.path.exists(p):
            data_list = torch.load(p, weights_only=False)
            print(f"Loaded {len(data_list)} graphs from {p}")
            return data_list

    pkl_path = "DATA/frame_dataset.pkl"
    if os.path.exists(pkl_path):
        from step_2_graph_constr import FrameGraphBuilder
        from normalizer import PhysicsScaler, MinMaxNormalizer

        with open(pkl_path, "rb") as f:
            dataset = pickle.load(f)

        builder = FrameGraphBuilder()
        data_list = builder.build_dataset(dataset)
        data_list = PhysicsScaler.compute_and_store_list(data_list)

        normalizer = MinMaxNormalizer()
        normalizer.fit(data_list)
        data_list = normalizer.transform_list(data_list)

        print(f"Built {len(data_list)} graphs from {pkl_path}")
        return data_list

    raise FileNotFoundError("No data found!")


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_load_vector(sample):
    """
    Extract external load vector for one sample.
    Adjust this if your dataset uses a different attribute name.
    """
    if hasattr(sample, "F_ext"):
        return to_numpy(sample.F_ext).reshape(-1)
    raise AttributeError("Sample does not have attribute 'F_ext'")


def main():
    train_ratio = 0.85

    print("Loading data...")
    data_list = load_data()

    n_total = len(data_list)
    n_train = max(1, int(train_ratio * n_total))

    train_set = [data_list[i] for i in range(n_train)]
    test_set  = [data_list[i] for i in range(n_train, n_total)]

    if len(test_set) == 0:
        print("No test samples found.")
        return

    print(f"Train samples: {len(train_set)}")
    print(f"Test samples:  {len(test_set)}")
    print("\nNearest training distance for each test case:\n")

    train_loads = [get_load_vector(s) for s in train_set]

    for i, test_sample in enumerate(test_set):
        test_load = get_load_vector(test_sample)

        distances = []
        for j, train_load in enumerate(train_loads):
            d = np.linalg.norm(test_load - train_load)
            distances.append((j, d))

        nearest_idx, nearest_distance = min(distances, key=lambda x: x[1])

        print(
            f"Test {i:3d}: nearest train idx = {nearest_idx:3d}, "
            f"distance = {nearest_distance:.6e}"
        )


if __name__ == "__main__":
    main()