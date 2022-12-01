import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import numpy as np
import data
import glob
import os
import matplotlib.pyplot as plt
from radial_based_layer import RBFN, choose_params
tf.config.run_functions_eagerly(True)


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(400, activation='relu')
        self.d2 = Dense(1)

    def call(self, x):
        t = self.d1(x)
        h = self.d2(t)
        return h


def get_params(X, ratio):
    down_idx = int(X.shape[0] * 0.5 * ratio)
    up_idx = X.shape[0] - down_idx
    X_s = np. sort(X, axis=0)
    down = X_s[down_idx]
    up = X_s[up_idx]
    params = {"down": down, "up": up}
    return params


# subject_files = glob.glob(os.path.join('./band_power_features/train/abs', '*.npz'))
class_num = 5
X_real = []
for i in range(class_num):
    X_real.append(data.load_unlabeled_data([f'D:/pycharm/Projects/EEGToGaussian/features_per_class/train/relative/feature_class{i}.npz'])[0])
for i in range(class_num):
    X_real[i] = X_real[i][:, 3, np.newaxis]
params = []
tail_ratio = 0.5
for i in range(class_num):
    params.append(get_params(X_real[i], tail_ratio))

X_train = np.arange(start=0.0001, stop=0.9999, step=0.0001)
# Y_train = np.tile(np.copy(X_train), (class_num, 1))
Y_train = np.copy(X_train)
X_train = X_train[..., np.newaxis]

# One log target
# Y_train = np.log(Y_train / (1 - Y_train))

# Average transformation
# for i in range(class_num):
#     index = np.logical_and(params[i]['down'] < Y_train[i, :], Y_train[i, :] < params[i]['up'])
#     r = params[i]["up"] - params[i]["down"]
#     m = np.mean(Y_train[i, index])
#     Y_train[i, index] = (Y_train[i, index] - params[i]['down']) / r
#     Y_train[i, index] = (np.log(Y_train[i, index] / (1 - Y_train[i, index])) / 10) * r + m
# Y_train = np.mean(Y_train, axis=0)
# Y_train = Y_train[..., np.newaxis]

# Separate transformation
for i in range(class_num):
    index = np.logical_and(params[i]['down'] < Y_train, Y_train < params[i]['up'])
    r = params[i]["up"] - params[i]["down"]
    m = np.mean(Y_train[index])
    Y_train[index] = (Y_train[index] - params[i]['down']) / r
    Y_train[index] = (np.log(Y_train[index] / (1 - Y_train[index])) / 10) * r + m
Y_train = Y_train[..., np.newaxis]
plt.plot(np.squeeze(X_train), np.squeeze(Y_train))
plt.show()
# Shuffle data
indices = tf.range(start=0, limit=tf.shape(X_train)[0], dtype=tf.int32)
idx = tf.random.shuffle(indices, seed=4)
X_train = tf.gather(X_train, idx)
Y_train = tf.gather(Y_train, idx)

val_ratio = 0.2
batch_size = 64
split = int(X_train.shape[0] * (1 - val_ratio))
batch_num = np.ceil(X_train.shape[0] * (1 - val_ratio) / batch_size)
batch_num_val = np.ceil(X_train.shape[0] * val_ratio / batch_size)
train_db = tf.data.Dataset.from_tensor_slices((X_train[:split, :], Y_train[:split, :])).batch(batch_size)
val_db = tf.data.Dataset.from_tensor_slices((X_train[split:, :], Y_train[split:, :])).batch(batch_size)
test_db = tf.data.Dataset.from_tensor_slices(tf.range(start=0, limit=1, delta=0.001, dtype=float)[..., tf.newaxis]).batch(1)
train_loss = tf.keras.metrics.Mean(name='train_loss')
val_loss = tf.keras.metrics.Mean(name='val_loss')
# num_c = 250
# centers = np.transpose(np.linspace(start=0+0.001, stop=1-0.001, num=num_c, dtype='float32')[..., np.newaxis])
# model = RBFN(num_c, centers, 0.005)
model = MyModel()
optimizer = tf.keras.optimizers.Adam()


@tf.function
def train_step(x, y_true):
    with tf.GradientTape() as tape:
        y = model(x)
        mse = tf.keras.losses.MeanSquaredError()
        loss = mse(y_true, y)
    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    train_loss(loss)


@tf.function
def val_step(x_val, y_true):
    y = model(x_val)
    mse = tf.keras.losses.MeanSquaredError()
    loss = mse(y_true, y)
    val_loss(loss)


epochs = 400
for epoch in range(epochs):
    train_loss.reset_states()
    val_loss.reset_states()

    for x, y in train_db:
        train_step(x, y)
    for x, y in val_db:
        val_step(x, y)
    print(
        f'Epoch {epoch + 1},'
        f'Loss: {train_loss.result()}, '
        f'Val Loss: {val_loss.result()}, '
    )

model.save('saved_model/init_400_5_classes')
test_y = []
test_x = []
for x in test_db:
    y = model(x)
    test_y.extend(y.numpy())
    test_x.extend(x.numpy())
test_y = np.squeeze(np.array(test_y))
test_x = np.squeeze(np.array(test_x))
s = np.argsort(test_x, axis=0)
test_x = test_x[s]
test_y = test_y[s]
plt.plot(test_x, test_y)
plt.show()