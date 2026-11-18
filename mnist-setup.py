import os
import numpy as np
from sklearn.datasets import fetch_openml

os.makedirs("assets", exist_ok=True)

print("Downloading MNIST dataset from OpenML...")
mnist = fetch_openml("mnist_784", version=1, parser="liac-arff", as_frame=False)
X = mnist["data"]                     # (70000, 784)
y = mnist["target"].astype(np.int64)  # labels arrive as strings
print(f"Downloaded: X {X.shape}, y {y.shape}, labels {sorted(set(y.tolist()))}")

data = np.column_stack([X, y]).astype(np.int64)  # label is the last column
print(f"Combined array: {data.shape}")

full_path = os.path.join("assets", "mnist_784_full.csv")
print(f"Saving full dataset to {full_path}...")
np.savetxt(full_path, data, fmt="%d", delimiter=",")
print(f"  wrote {len(data)} rows, {os.path.getsize(full_path) / 1e6:.1f} MB")

print("Generating randomized 10% sample (seed 42)...")
rng = np.random.default_rng(42)
idx = rng.choice(len(data), size=7000, replace=False)
sample = data[idx]

sample_path = os.path.join("assets", "mnist_784.csv")
print(f"Saving sample dataset to {sample_path}...")
np.savetxt(sample_path, sample, fmt="%d", delimiter=",")
print(f"  wrote {len(sample)} rows, {os.path.getsize(sample_path) / 1e6:.1f} MB")

counts = np.bincount(sample[:, -1], minlength=10)
print(f"Sample class counts: {counts.tolist()}")

print("Successfully saved both the full and sampled datasets to the assets directory!")
