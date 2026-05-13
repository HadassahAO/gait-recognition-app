import os
from sklearn.preprocessing import LabelEncoder
import pickle

dataset_path = r"G:\My Drive\gait-dataset\Unoccluded_GEI\Unoccluded_GEI\Unoccluded_GEI"

labels = []

for file in os.listdir(dataset_path):
    if file.endswith(".png"):
        label = file.split("_")[0]   # adjust if needed
        labels.append(label)

le = LabelEncoder()
le.fit(labels)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Encoder rebuilt successfully!")
print("Classes:", le.classes_) 