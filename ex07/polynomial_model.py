import numpy as np


def add_polynomial_features(x, power):
    """Add polynomial features to vector x by raising its values up
    to the power given in argument.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        power: has to be an int, the power up to which the components
               of vector x are going to be raised.
    Return:
        The matrix of polynomial features as a numpy.array, of shape m * n,
        scontaining the polynomial feature values for all training examples.
        None if x is an empty numpy.array.
        None if x or power is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        # Checking type of the parameters
        if (not isinstance(x, np.ndarray)) \
                or (not isinstance(power, int)):
            return None

        # Checking number of dimension and shape of x
        if not (x.ndim in [1, 2]):
            return None
        if (x.ndim == 2) and (x.shape[1] != 1):
            return None

        # Checking dtype of the x ndarray
        if (x.dtype.kind not in ['i', 'f']):
            return None

        # Treating the different cases of power value
        if power < 0:
            return None
        elif power == 0:
            return np.ones((x.shape[0], 1))
        elif power == 1:
            return np.array(x, copy=True).reshape(-1, 1)
        else:
            return np.vander(x.reshape(-1,),
                             N=power + 1,
                             increasing=True)[:, 1:]
    except:
        return None


if __name__ == "__main__":
    x = np.arange(1, 6).reshape(-1, 1)

    print("# Example 0:")
    res = add_polynomial_features(x, 3)
    print(res)
    # Output:
    # array([[ 1,  1,   1],
    #        [ 2,  4,   8],
    #        [ 3,  9,  27],
    #        [ 4, 16,  64],
    #        [ 5, 25, 125]])

    print("# Example 1:")
    res = add_polynomial_features(x, 6)
    print(res)
    # Output:
    # array([[ 1,  1,   1,   1,    1,     1],
    #        [ 2,  4,   8,  16,   32,    64],
    #        [ 3,  9,  27,  81,  243,   729],
    #        [ 4, 16,  64, 256, 1024,  4096],
    #        [ 5, 25, 125, 625, 3125, 15625]])
