import numpy as np

def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a vector of shape 2 * 1.
    Returns:
        y_hat as a numpy.array, a vector of shape m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta shapes are not appropriate.
        None if x or theta is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        if not isinstance(x, np.ndarray):
            return None
    
        if x.ndim == 1:
            x = x.reshape(-1,1)
        if any([n == 0 for n in x.shape]):
            return None

        if theta.shape != (x.shape[1] + 1, 1):
            return None

        xp = np.hstack((np.ones((x.shape[0], 1)), x))

        ypred = np.dot(xp, theta)
        return ypred
    except:
        return None