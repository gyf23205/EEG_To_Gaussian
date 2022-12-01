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
        self.d1 = Dense(150, activation='relu')
        self.d2 = Dense(1)

    def call(self, x):
        t = self.d1(x)
        h = self.d2(t)
        return h

subject_files = glob.glob(os.path.join('./band_power_features/train', '*.npz'))
X_real, _, _, _ = data.load_data(subject_files, data_from_cluster=False)
total_power = np.sum(X_real[0], axis=1)[..., np.newaxis]
X_real = np.divide(X_real[0], total_power)
X_real = X_real[:, 3, np.newaxis]
mean_X = np.mean(X_real)
X_train = np.arange(start=0+0.001, stop=10-0.001, step=0.01)[..., np.newaxis]
# X_train = np.arange(start=0.001, stop=1-0.001, step=0.0001)[..., np.newaxis]
Y_train = X_train
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
test_db = tf.data.Dataset.from_tensor_slices(tf.range(start=0, limit=10, delta=0.02, dtype=float)[..., tf.newaxis]).batch(1)
train_loss = tf.keras.metrics.Mean(name='train_loss')
val_loss = tf.keras.metrics.Mean(name='val_loss')
# num_c = 2
# centers = np.transpose(choose_params(X_train.numpy(), num_c))
# centers = np.transpose(np.linspace(start=params["down"]+0.001, stop=params["up"]-0.001, num=num_c))
# model = RBFN(num_c, centers, 0.05)
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


epochs = 50
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

model.save('saved_model/init_linear')

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