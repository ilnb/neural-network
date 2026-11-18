import os
import numpy as np
from sklearn.datasets import load_digits

os.makedirs("demos", exist_ok=True)

print("Loading sklearn digits dataset...")
digits = load_digits()
X = digits.data                    # (1797, 64), pixel values 0-16
y = digits.target.astype(np.int64) # (1797,)
print(f"Loaded: X {X.shape}, y {y.shape}, labels {sorted(set(y.tolist()))}")
print(f"Pixel range: {X.min():.0f} to {X.max():.0f}")

data = np.column_stack([X, y]).astype(np.int64)  # label is the last column
print(f"Combined array: {data.shape}")

path = os.path.join("demos", "digits.csv")
print(f"Saving to {path}...")
np.savetxt(path, data, fmt="%d", delimiter=",")
print(f"  wrote {len(data)} rows, {os.path.getsize(path) / 1e3:.1f} KB")

counts = np.bincount(y, minlength=10)
print(f"Class counts: {counts.tolist()}")

print("Done.")
