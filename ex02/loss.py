import sys
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
        # Checking y and y_hat are numpy array
        if (not isinstance(y, np.ndarray)) \
                or (not isinstance(y_hat, np.ndarray)):
            print("Numpy arrays are expected.", file=sys.stderr)
            return None

        # Checking the shape of y and y_hat
        if (y.shape[1] != 1) \
            or (y_hat.shape[1] != 1) \
                or (y_hat.shape[0] != y.shape[0]):
            s = "Shape issue: either y and/or y_hat are not 2 dimensional," \
                + " or not the same number of lines."
            print(s, file=sys.stderr)
            return None
        loss = (y - y_hat).T @ (y - y_hat) / (2.0 * y.shape[0])
        return float(loss)
    except:
        None


if __name__ == "__main__":
    X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
    Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])

    print("# Example 0:")
    myloss = loss_(X, Y)
    # Output:
    expected_loss = 2.1428571428571436
    print("my loss: ".ljust(20), myloss)
    print("expected loss: ".ljust(20), expected_loss)

    print("\n# Example 1:")
    myloss = loss_(X, X)
    # Output:
    expected_loss = 0.0
    print("my loss: ".ljust(20), myloss)
    print("expected loss: ".ljust(20), expected_loss)
