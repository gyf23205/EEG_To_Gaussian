import glob
import os
import shutil
import numpy as np

from data import load_data, my_get_subject_files
from feature_extractor import feature_extractor

# Repack data according to their class (sleep stages)
subject_files = glob.glob(os.path.join("./band_power_features/train/abs", "*.npz"))
x, y, fs, _ = load_data(subject_files, data_from_cluster=False)

c0 = []
c1 = []
c2 = []
c3 = []
cRem = []
for i in range(len(x)):
    for j in range(x[i].shape[0]):
        if y[i][j] == 0:
            c0.append(x[i][j, :])
        elif y[i][j] == 1:
            c1.append(x[i][j, :])
        elif y[i][j] == 2:
            c2.append(x[i][j, :])
        elif y[i][j] == 3:
            c3.append(x[i][j, :])
        else:
            cRem.append(x[i][j])
feature_per_class = [c0, c1, c2, c3, cRem]
for i in range(len(feature_per_class)):
    feature_per_class[i] = np.array(feature_per_class[i])
# Output dir
data_output_dir = './features_per_class/train/abs'
if not os.path.exists(data_output_dir):
    os.makedirs(data_output_dir)
else:
    shutil.rmtree(data_output_dir)
    os.makedirs(data_output_dir)
# Save
for i in range(len(feature_per_class)):
    filename = f"feature_class{str(i)}.npz"
    save_dict = {
        "x": feature_per_class[i].astype(np.float32)
    }
    np.savez(os.path.join(data_output_dir, filename), **save_dict)
