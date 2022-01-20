import numpy as np

def loss_(y, y_hat):
    """Computes the half mean squared error of two non-empty numpy.array,
    without any for loop. The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        The half mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions.
        None if y or y_hat is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        loss = (y - y_hat).T @ (y - y_hat) / (2.0 * y.shape[0])
        return float(loss)
    except:
        None

if __name__ == "__main__":
    X = np.array([[0],[ 15],[ -9],[ 7],[ 12],[ 3],[ -21]])
    Y = np.array([[2],[ 14],[ -13],[ 5],[ 12],[ 4],[ -19]])
    # Example 0:
    loss_(X, Y)
    # Output:
    2.1428571428571436
    # Example 1:
    loss_(X, X)
    # Output:
    0.0