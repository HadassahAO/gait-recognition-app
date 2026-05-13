import os

dataset_path = r"G:\My Drive\gait-dataset\Unoccluded_GEI\Unoccluded_GEI\Unoccluded_GEI"

labels = []

for file in os.listdir(dataset_path):

    if file.endswith(".png") or file.endswith(".jpg"):

        parts = file.split("_")

        if len(parts) > 1:
            label = parts[0] + "_" + parts[1]
            labels.append(label)

print("Total images:", len(labels))
print("Unique classes:", len(set(labels)))
print("Sample labels:", labels[:10])