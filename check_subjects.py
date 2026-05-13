import os

dataset_path = "C:\Users\seuna\Desktop\gait_project"

subjects = [
    d for d in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, d))
]

print("Subjects in dataset:")
print(subjects)
print("Total subjects:", len(subjects))