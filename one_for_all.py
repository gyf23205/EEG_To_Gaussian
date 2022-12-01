import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from radial_based_layer import RBFN
import numpy as np
import data
import math
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from scipy.stats import anderson, jarque_bera
tf.config.run_functions_eagerly(True)


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(400, activation='relu')
        self.d2 = Dense(1)

    def call(self, x):
        y = self.d1(x)
        y = self.d2(y)
        return y


def klloss(y, G_d, band_width):
    # notation "d" stands for "desired"
    prob_d = G_d.prob(tf.squeeze(y))
    n = y.shape[0]
    # Following lines estimate probability density for each point in y using Gaussian kernel
    ext1 = tf.reshape(tf.tile(y, [n, 1]), [n, n])
    ext2 = tf.reshape(tf.tile(y, [1, n]), [n, n])
    t = ext2 - ext1
    t = (1 / tf.sqrt(2 * math.pi)) * tf.exp(-1 * (tf.square(t) / 2))
    t = tf.reduce_sum(t, axis=1) * 0.5 * (1 / band_width)
    loss = tf.reduce_sum(t * tf.math.log(t / prob_d))
    return loss


def jbloss(y, mean_x):
    n = y.shape[0]
    # n = batch_size
    mean = tf.reduce_mean(y, axis=0)
    mu1 = tf.divide(tf.reduce_sum(tf.pow(y - mean, 3.0), axis=0), n)
    sigma1 = tf.pow(tf.divide(tf.reduce_sum(tf.pow(y - mean, 2.0), axis=0), n), 1.5)
    mu2 = tf.divide(tf.reduce_sum(tf.pow(y - mean, 4.0), axis=0), n)
    sigma2 = tf.pow(tf.divide(tf.reduce_sum(tf.pow(y - mean, 2.0), axis=0), n), 2)
    S = tf.divide(mu1, sigma1)
    K = tf.divide(mu2, sigma2)
    loss = tf.reduce_sum(tf.multiply(n / 6, tf.pow(S, 2.0) + tf.multiply(tf.pow(K - 3, 2.), 0.25)))\
           + 0.1 * tf.square(mean - mean_x) #+ 1 * tf.square(cov - cov_x)

    return loss


def adloss(y, dist):
    n = y.shape[0]
    sort = np.argsort(y.numpy(), axis=0)
    idx = np.ones(y.shape[0])
    for s, i in enumerate(sort):
        idx[i] = s
    idx = tf.constant(idx + 1, dtype=float)
    mean = tf.reduce_mean(y, axis=0)
    var = tf.math.reduce_variance(y, axis=0)
    y = (y - mean) / tf.sqrt(var)
    loss = -n - (1. / n) * \
           tf.reduce_sum(tf.expand_dims(2. * idx - 1., axis=1) * tf.math.log(dist.cdf(y)) \
                         + tf.expand_dims(2. * (n - idx) + 1., axis=1) * tf.math.log(1. - dist.cdf(y)))
    return loss


class_num = 5
band = 3
X_train = []
means = []
stds = []
length = []
for i in range(class_num):
    x = data.load_unlabeled_data([f'D:/pycharm/Projects/EEGToGaussian/features_per_class/train/relative/feature_class{i}.npz'])[0]
    X_train.append(tf.random.shuffle(x[:, band, np.newaxis], seed=1))
    means.append(np.mean(x))
    stds.append(np.std(x))
    length.append(x.shape[0])
length = np.array(length)

X_test = []
for i in range(class_num):
    x = data.load_unlabeled_data([f'D:/pycharm/Projects/EEGToGaussian/features_per_class/train/relative/feature_class{i}.npz'])[0]
    X_test.append(tf.random.shuffle(x[:, band, np.newaxis], seed=1))

test_db = []
for i in range(class_num):
    plt.subplot(3, 2, i+1)
    s, _ = jarque_bera(np.squeeze(X_test[i].numpy()))
    mean = tf.reduce_mean(X_test[i], axis=0)
    cov = tf.math.reduce_variance(X_test[i], axis=0)
    plt.title(f"s:{s}, mean:{mean}", fontsize=18)
    plt.hist(np.squeeze(X_test[i].numpy()), 80)
    test_db.append(tf.data.Dataset.from_tensor_slices(X_test[i]).batch(1))
plt.show()

X_test = tf.concat(X_test, axis=0)

val_ratio = 0.1
batch_size = 128
split = np.array(length * (1 - val_ratio), dtype=int)
train_db = []
val_db = []
for i in range(class_num):
    train_db.append(tf.data.Dataset.from_tensor_slices(X_train[i][:split[i], :]).batch(batch_size))
    val_db.append(tf.data.Dataset.from_tensor_slices(X_train[i][split[i]:, :]).batch(batch_size))

plot_db = tf.data.Dataset.from_tensor_slices(tf.range(start=0, limit=1, delta=0.001, dtype=float)[..., tf.newaxis]).batch(1)
tfd = tfp.distributions
G_d = []
for i in range(class_num):
    G_d.append(tfd.Normal(loc=means[i], scale=stds[i]))
band_width = 0.3  # need to be tuned
# model = MyModel()
model = tf.keras.models.load_model('D:/pycharm/Projects/EEGToGaussian/saved_model/init_400_5_classes')
# num_c = 250
# centers = np.transpose(np.linspace(start=0+0.001, stop=1-0.001, num=num_c, dtype='float32')[..., np.newaxis])
# model = RBFN(num_c, centers, 0.05)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)


@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        y = model(x)
        mean_x = tf.reduce_mean(x, axis=0)
        # cov_x = tf.math.reduce_variance(x, axis=0)
        loss = jbloss(y, mean_x)
        # loss = adloss(y, dist)
    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    # train_loss(loss)
    return loss


@tf.function
def val_step(x_val):
    y = model(x_val)
    mean_x = tf.reduce_mean(x_val, axis=0)
    # cov_x = tf.math.reduce_variance(x_val, axis=0)
    loss = jbloss(y, mean_x)
    # loss = adloss(y, dist)
    return loss


train_loss_hist = [] 
val_loss_hist = []
epochs = 500
for epoch in range(epochs):
    # train_loss.reset_states()
    # val_loss.reset_states()
    dist = G_d[epoch % class_num]
    for x in train_db[epoch % class_num]:
        loss_train = train_step(x).numpy()[0]  # only need index when using jbloss
    for x in val_db[epoch % class_num]:
        loss_val = val_step(x).numpy()[0]  # only need index when using jbloss
    train_loss_hist.append(loss_train)
    val_loss_hist.append(loss_val)
    print(
        f'Epoch {epoch + 1},'
        f'Loss: {loss_train}, '
        f'Val Loss: {loss_val}, '
    )

plt.subplot(2, 1, 1)
plt.plot(train_loss_hist)
plt.subplot(2, 1, 2)
plt.plot(val_loss_hist)
plt.show()

# model.save('saved_model/one_for_all_success')

# print train_db
for step, db in enumerate(test_db):
    test_y = []
    for x in db:
        y = model(x)
        test_y.extend(y.numpy())
    test_y = np.array(test_y)
    plt.subplot(3, 2, step + 1)
    s, _ = jarque_bera(np.squeeze(test_y))
    mean = tf.reduce_mean(test_y, axis=0)
    cov = tf.math.reduce_variance(test_y, axis=0)
    plt.title(f"s:{s}, mean:{mean}", fontsize=18)
    plt.hist(np.squeeze(test_y), 80)
plt.show()

plot_y = []
plot_x = []
for x in plot_db:
    y = model(x)
    plot_y.extend(y.numpy())
    plot_x.extend(x.numpy())
plot_y = np.squeeze(np.array(plot_y))
plot_x = np.squeeze(np.array(plot_x))
s = np.argsort(plot_x, axis=0)
plot_x = plot_x[s]
plot_y = plot_y[s]
plt.plot(plot_x, plot_y)
plt.show()