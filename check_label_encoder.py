import os
import pickle

path = os.path.join(os.getcwd(), "label_encoder.pkl")
print("Looking in:", path)

le = pickle.load(open(path, "rb"))

print("Classes:", le.classes_)

