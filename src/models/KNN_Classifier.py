from collections import Counter
import numpy as np


def knn_classifier(train, test, k):
    x_train = train[:, :1]
    y_train = train[:, 2]

    x_val = test[:, :1]
    y_val = test[:, 2]
    """
    Calculate distances
    URL: https://stackoverflow.com/questions/65851217/compute-pairwise-differences-between-two-vectors-in-numpy
    AUTHOR: Arya McCarthy
    DATE ACCESSED: Sept 6th, 2024
    DATE PUBLISHED: Jan 22th, 2021
    """
    distances = np.subtract.outer(x_train, x_val) ** 2

    distances = distances.sum(axis=(1, 3))

    distances = np.sqrt(distances)

    """
    https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
    """
    nearest_arr = np.argsort(distances, axis=0)[:k]

    pred = []

    for i in range(x_val.shape[0]):
        # extract the nearest neighbor labels
        nearest = y_train[nearest_arr[:, i]]

        # determine most voted class
        most_common = Counter(nearest).most_common()
        pred.append(most_common[0][0])

    return pred