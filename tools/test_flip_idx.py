import numpy as np

names = ["tomato_calyx", "tomato_bottom", "tomato_left", "tomato_right"]
kpts = np.array([
    [0.35, 0.25, 2],
    [0.36, 0.75, 2],
    [0.15, 0.48, 2],
    [0.58, 0.48, 2],
], dtype=float)

mirrored = kpts.copy()
mirrored[:, 0] = 1.0 - mirrored[:, 0]

flip_idx = [0, 1, 3, 2]
correct = mirrored[flip_idx]

print("Horizontal flip test for tomato keypoints")
for name, old, new in zip(names, kpts, correct):
    print(f"{name:14s} before={old[:2]} after={new[:2]}")

print("\nExpected:")
print("calyx and bottom keep their semantic indexes.")
print("left and right should be swapped after horizontal flip.")
