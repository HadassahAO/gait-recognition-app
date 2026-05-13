import os

base_path = r"G:\My Drive\gait-dataset"

image_files = []

for root, dirs, files in os.walk(base_path):
    for f in files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(root, f))

print("Total images found:", len(image_files))
print("Sample:", image_files[:5])