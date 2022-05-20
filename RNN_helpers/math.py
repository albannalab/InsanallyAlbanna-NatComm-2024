import numpy as np


def normal(shape, dtype='float64', **kwargs):
    return(np.random.normal(size=shape, **kwargs).astype(dtype))


def uniform(shape, dtype='float64', **kwargs):
    return(np.random.random(size=shape, **kwargs).astype(dtype))


def randbool(shape, prob=0.5, dtype='bool', **kwargs):
    mask = uniform(shape) <= prob
    return(mask.astype(dtype))


def zeros(shape, dtype='float64', **kwargs):
    return(np.zeros(shape).astype(dtype))


def ones(shape, dtype='float64', **kwargs):
    return(np.ones(shape).astype(dtype))


def pos_neg_split(matrix):
    pos_matrix = np.clip(matrix, 0.0, None)
    neg_matrix = -np.clip(matrix, None, 0.0)
    return(pos_matrix, neg_matrix)