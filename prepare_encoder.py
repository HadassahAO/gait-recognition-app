import os
from sklearn.preprocessing import LabelEncoder
import pickle

dataset_path = r"G:\My Drive\gait-dataset\Unoccluded_GEI\Unoccluded_GEI\Unoccluded_GEI"

labels = []

for file in os.listdir(dataset_path):

    if file.endswith(".png") or file.endswith(".jpg"):

        parts = file.split("_")

        label = parts[0] + "_" + parts[1]
        labels.append(label)

print("Total images:", len(labels))
print("Unique classes:", len(set(labels)))

le = LabelEncoder()
le.fit(labels)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✔ Encoder rebuilt successfully")
print(le.classes_[:10])