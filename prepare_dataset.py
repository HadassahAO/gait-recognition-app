# =========================================================
# IMPORTS
# =========================================================

import os
import numpy as np
from PIL import Image
from collections import Counter

# =========================================================
# DATASET PATH
# =========================================================

dataset_path = r"G:\My Drive\gait-dataset\Unoccluded_GEI\Unoccluded_GEI\Unoccluded_GEI"

# =========================================================
# STORAGE
# =========================================================

X = []
y = []

valid_extensions = (".png", ".jpg", ".jpeg")

# =========================================================
# LABEL EXTRACTION
# subject_118_M_... → subject_118
# =========================================================

def extract_label(filename):
    parts = filename.split("_")
    return parts[0] + "_" + parts[1]

# =========================================================
# LOAD DATASET
# =========================================================

print("🚀 Loading dataset...")

for root, dirs, files in os.walk(dataset_path):

    for file in files:

        if not file.lower().endswith(valid_extensions):
            continue

        img_path = os.path.join(root, file)

        try:
            # =================================================
            # SAFE IMAGE LOADING (PIL FIX)
            # =================================================
            img = Image.open(img_path).convert("L")  # grayscale
            img = img.resize((64, 64))

            # Convert to numpy array
            img = np.array(img)

            # Store image
            X.append(img)

            # Extract label
            label = extract_label(file)
            y.append(label)

        except Exception as e:
            print("⚠ Skipping file:", img_path)
            print("Reason:", e)

# =========================================================
# CONVERT TO NUMPY ARRAYS
# =========================================================

X = np.array(X)
y = np.array(y)

# =========================================================
# SUMMARY
# =========================================================

print("\n========== DATASET SUMMARY ==========")
print("Total images loaded:", len(X))
print("Total labels:", len(y))
print("Unique subjects:", len(set(y)))

print("\nTop 10 label distribution:")
print(Counter(y).most_common(10))

# =========================================================
# SAVE FOR TRAINING
# =========================================================

np.save("X.npy", X)
np.save("y.npy", y)

print("\n✅ Dataset saved successfully (X.npy, y.npy)")
print("🎯 Ready for CNN training stage")