import numpy as np


def make_lin_fit_y(data):
    return np.expand_dims(data, axis=1)


def make_lin_fit_x(length, start=0):
    return np.vstack([np.ones(length), np.arange(start, start + length)]).T


def lin_fit(x, y):
    fit = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    return fit[1, 0], fit[0, 0]