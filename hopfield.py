# Q2_graded
# Do not change the above line.
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets


class Hopfield:
    def __init__(self, train, bias):
        self.bias = bias
        self.dimension = len(train[0])
        self.W = np.zeros((self.dimension, self.dimension))
        mean = np.sum([np.sum(t) for t in train]) / (len(train) * self.dimension)
        for sample in train:
            self.W += np.outer(sample - mean, sample - mean)
        np.fill_diagonal(self.W, 0)
        self.W /= len(train)

    def update_input_synchronously(self, x):
        return np.sign(self.W.dot(x) - self.bias)


def scale_and_transform_data(n_patterns):
    train = [X_train[11111], X_train[22222], X_train[33333]]
    train = [np.sign(t.reshape(-1) * 2 - 1) for t in train]
    return train


def addnoise(train, error_rate):
    """
    Adds random noise to the train data and returns it as the test data.
    Noise is added by flipping the sign of some units with the error rate p.
    """
    test = np.copy(train)  # cf. from copy import copy/deepcopy
    for i, t in enumerate(test):
        s = np.random.binomial(1, error_rate, len(t))
        for j in range(len(t)):
            if s[j] != 0:
                t[j] *= -1
    return test


def extract_one_from_each_label():
    labels_indexes = {}
    for i in range(len(y_train)):
        if len(labels_indexes.keys()) == 10:
            break
        if y_train[i] not in labels_indexes.keys():
            labels_indexes[y_train[i]] = i
    return np.sign(X_train[np.array(list(labels_indexes.values()))].reshape(10, -1) * 2 - 1), list(
        labels_indexes.values())

n_patterns = 3
n_units = 28
err = 0.15
(X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()
X_train = X_train / 255
train = scale_and_transform_data(n_patterns)
test = addnoise(train, error_rate=err)
enumerator = 1
for noisy in test:
    plt.subplot(2,n_patterns,enumerator)
    plt.axis("off")
    plt.imshow(noisy.reshape(28, 28), cmap="gray")
    plt.title("added noise")
    enumerator += 1

model = Hopfield(train, bias=40)

predict = []
for t in test:
    predict.append(model.update_input_synchronously(t))
for removed_noise in predict:
    plt.subplot(2,n_patterns,enumerator)
    plt.axis("off")
    plt.imshow(removed_noise.reshape(28, 28), cmap="gray")
    plt.title("removed noise")
    enumerator += 1

plt.show()

# Q2_graded
# Do not change the above line.
def scale_and_transform_data_2(n_patterns):
    train = X_train[:n_patterns]
    train = [np.sign(t.reshape(-1) * 2 - 1) for t in train]
    return train

n_patterns = 12
train = scale_and_transform_data_2(n_patterns)
test_images, train_indexes = extract_one_from_each_label()
test = addnoise(test_images, error_rate=err)

model = Hopfield(train, bias=0)
predict = []
for t in test:
    predict.append(model.update_input_synchronously(t))

fig, ax = plt.subplots(nrows=10, ncols=4, sharex=True, sharey=True)
enumerator = 0
for row in ax:
    row[0].axis("off")
    row[0].imshow(X_train[train_indexes[enumerator]], cmap="gray")
    row[1].axis("off")
    row[1].imshow(test_images[enumerator].reshape(28, 28), cmap="gray")
    row[2].axis("off")
    row[2].imshow(test[enumerator].reshape(28, 28), cmap="gray")
    row[3].axis("off")
    row[3].imshow(predict[enumerator].reshape(28, 28), cmap="gray")
    enumerator += 1
plt.figure()
for i in range(10):
    plt.subplot(10, 4, 4 * i + 1)
    plt.axis("off")
    plt.imshow(X_train[train_indexes[i]], cmap="gray", aspect='auto')
    plt.subplot(10, 4, 4 * i + 2)
    plt.axis("off")
    plt.imshow(test_images[i].reshape(28, 28), cmap="gray", aspect='auto')
    plt.subplot(10, 4, 4 * i + 3)
    plt.axis("off")
    plt.imshow(test[i].reshape(28, 28), cmap="gray", aspect='auto')
    plt.subplot(10, 4, 4 * i + 4)
    plt.axis("off")
    plt.imshow(predict[i].reshape(28, 28), cmap="gray", aspect='auto')

plt.show()

# Q2_graded
# Do not change the above line.
def add_noise_to_image(sample, error_rate):
    test = np.copy(sample)
    s = np.random.binomial(1, error_rate, len(test))
    for j in range(len(test)):
        if s[j] != 0:
            test[j] *= -1
    return test

def compute_accuracy(true_image, removed_noise_image):
    return 1 - np.sum(true_image != removed_noise_image) / (28 * 28)

n_units = 28
patterns_num = [10, 20, 30]
error_rates = [0.1, 0.3, 0.6]
accuracies = {}
for pattern_count in patterns_num:
    train = scale_and_transform_data_2(pattern_count)
    model = Hopfield(train, bias=10)
    predict = []
    for noise_rate in error_rates:
        test = add_noise_to_image(train[0], error_rate=noise_rate)
        updated_noise = model.update_input_synchronously(test)
        predict.append(updated_noise)
        accuracies[(pattern_count, noise_rate)] = compute_accuracy(train[0], updated_noise)
accuracies

