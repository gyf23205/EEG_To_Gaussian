import numpy as np
from data import load_data
import os

class AFT:
    def __init__(self, ratio):
        self.num_stage = 5
        self.centers = np.zeros(self.num_stage)
        self.bounds = np.zeros((2, self.num_stage))
        self.ranges = np.zeros(self.num_stage)
        self.ratio = ratio

    def train(self, x, y):
        for i in range(self.num_stage):
            x_tmp = np.sort(x[y == i])
            self.bounds[0, i] = x_tmp[int(len(x_tmp) * 0.5 * self.ratio)]  # lower bound of ith stage
            self.bounds[1, i] = x_tmp[int(len(x_tmp) * (1 - 0.5 * self.ratio))]  # upper bound of ith stage
            self.centers[i] = np.mean(x_tmp)
        self.ranges = self.bounds[1, :] - self.bounds[0, :]

    def transf(self, x):
        x_tmp = np.tile(x, [self.num_stage, 1])
        for i in range(self.num_stage):
            idx = np.logical_and(self.bounds[0, i] < x_tmp[i, :], x_tmp[i, :] < self.bounds[1, i])
            x_tmp[i, idx] = (x_tmp[i, idx] - self.bounds[0, i])/self.ranges[i]
            x_tmp[i, idx] = np.log(np.divide(x_tmp[i, idx], 1 - x_tmp[i, idx])) / 10 * self.ranges[i] + self.centers[i]
        return np.mean(x_tmp, axis=0)


train_path = []
test_path = []
train_subjects = os.listdir('./band_power_features/test/relative')
test_subjects = os.listdir('./band_power_features/test/relative')
for i in range(len(train_subjects)):
    train_path.append(os.path.join('./band_power_features/test/relative', train_subjects[i]))
for i in range(len(test_subjects)):
    test_path.append(os.path.join('./band_power_features/test/relative', test_subjects[i]))

band = {"beta": 0, "alpha": 1, "theta": 2, "delta": 3}
X_train, Y_train, _, _ = load_data(train_path)
X_test, Y_test, _, _ = load_data(test_path)
X_train = np.concatenate(X_train, axis=0)[:, band['delta']]
Y_train = np.concatenate(Y_train, axis=0)
X_test = np.concatenate(X_test, axis=0)[:, band['delta']]
Y_test = np.concatenate(Y_test, axis=0)
T = AFT(0.1)
T.train(X_train, Y_train)
X_transformed = T.transf(X_test)
