import numpy as np

# 3 samples x 4 features, like a tiny dataset
X = np.array([[10, 11, 12, 13],
              [20, 21, 22, 23],
              [30, 31, 32, 33]])

print("X.shape:", X.shape)

# --- Single elements: [row, col], counting starts at 0
print("\n1) X[1, 2] =", X[1, 2])
print("   X[-1, -1] =", X[-1, -1])  # negative index counts from the end

# --- Whole rows and columns: ':' means "everything on this axis"
print("\n2) X[0]    =", X[0], "  shape:", X[0].shape)
print("   X[:, 1] =", X[:, 1], "  shape:", X[:, 1].shape)

# --- Slices: start:stop, stop is excluded
print("\n3) X[0:2, 1:3] =\n", X[0:2, 1:3])

# Keeping a dimension: a slice keeps the axis, an integer drops it.
# This matters when the next step expects a 2D (samples, features) array.
print("\n4) X[:, 1].shape   =", X[:, 1].shape)
print("   X[:, 1:2].shape =", X[:, 1:2].shape)

# --- Boolean masks: filter rows by a condition (how we select samples in ML)
mask = X[:, 0] > 15
print("\n5) mask          =", mask)
print("   X[mask] =\n", X[mask])

# --- Fancy indexing: pick specific rows in any order (used for shuffling data)
order = np.array([2, 0, 1])
print("\n6) X[[2, 0, 1]] =\n", X[order])

# --- Slices are views: changing them changes the original array.
# Copy explicitly when the original must stay untouched.
view = X[0, :2]
view[0] = 999
print("\n7) after editing the view, X[0] =", X[0])

X[0, 0] = 10
safe = X[0, :2].copy()
safe[0] = 999
print("   after editing the copy, X[0] =", X[0])
