import tensorflow as tf
from sklearn.cluster import KMeans
import numpy as np
import data


class RBFN(tf.keras.Model):
    def __init__(self, num_centers, centers, var):
        super(RBFN, self).__init__()
        self.num_c = int(num_centers)
        self.centers = tf.constant(centers)
        self.var = tf.constant(var)
        self.w = tf.Variable(tf.random.normal([self.num_c, 1], stddev=0.35),
                             name="weights")

    def call(self, x):
        centers = tf.tile(self.centers, [x.shape[0], 1])
        x = tf.tile(x, [1, self.num_c])
        x = tf.square(x - centers)/(2 * self.var)
        x = tf.exp(-x)
        return tf.matmul(x, self.w)


def choose_params(x, num_c):
    kmeans = KMeans(n_clusters=num_c, random_state=0).fit(x)
    # y = kmeans.labels_
    # var = []
    # inertia = kmeans.inertia_
    # for i in range(num_c):
    #     num_s = np.sum(y == i)
    #     var.append(inertia[i]/num_s)
    return np.array(kmeans.cluster_centers_, dtype='float32')#, var


if __name__ == "__main__":
    X_train = data.load_unlabeled_data(
        ['D:/pycharm/Projects/EEGToGaussian/features_per_class/train/feature_class1.npz'])
    total_power = np.sum(X_train[0], axis=1)[..., np.newaxis]
    X_train = np.divide(X_train[0], total_power)
    X_train = X_train[:, 3, np.newaxis]
    train_db = tf.data.Dataset.from_tensor_slices(X_train).batch(32)
    num_c = 10
    centers = np.transpose(choose_params(X_train, num_c))
    model = RBFN(10, centers, 0.1)
    for x in train_db:
        pred = model(x)
        print(f'x:{x}')
        print(f'pred:{pred}')
    print()