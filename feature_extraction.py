import glob
import os
import shutil
import numpy as np

from data import load_data, my_get_subject_files
from feature_extractor import feature_extractor

subject_files = glob.glob(os.path.join("D:/pycharm/Projects/EEGToGaussian/trainnpz", "*.npz"))
test_files = my_get_subject_files(files=subject_files)
x, y, fs, _ = load_data(test_files, data_from_cluster=False)
for i in range(len(x)):
    x[i] = np.squeeze(x[i])

features = []
for i in range(len(x)):
    print(i)
    temp = np.zeros((x[i].shape[0], 4))
    for j in range(x[i].shape[0]):
        temp[j, :] = feature_extractor(x[i][j, :], fs)
    features.append(temp)
# Output dir
data_output_dir = './band_power_features/train/abs'
if not os.path.exists(data_output_dir):
    os.makedirs(data_output_dir)
else:
    shutil.rmtree(data_output_dir)
    os.makedirs(data_output_dir)
for i in range(len(features)):
    # Save
    filename = f"{str(i).zfill(2)}.npz"
    save_dict = {
        "x": features[i].astype(np.float32),
        "y": y[i].astype(np.int32),
        "fs": 100
    }
    np.savez(os.path.join(data_output_dir, filename), **save_dict)
