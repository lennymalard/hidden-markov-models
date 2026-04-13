import numpy as np

def logsumexp(x, axis=None, keepdims=False):
    x_max = np.max(x, axis=axis, keepdims=True)
    inner_sum = np.sum(np.exp(x - x_max), axis=axis, keepdims=True)
    lse = x_max + np.log(inner_sum)

    if not keepdims:
        if axis is None:
            return np.squeeze(lse)
        else:
            return np.squeeze(lse, axis=axis)

    return lse