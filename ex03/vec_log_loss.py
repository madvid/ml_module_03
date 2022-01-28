import numpy as np
import os
import sys

path = os.path.join(os.path.dirname(__file__), '..', 'ex00')
sys.path.insert(1, path)
from sigmoid import sigmoid_

path = os.path.join(os.path.dirname(__file__), '..', 'ex01')
sys.path.insert(1, path)
from log_pred import logistic_predict_


def vec_log_loss_(y, y_hat, eps=1e-30):
    """
    Computes the logistic loss value.
    Args:
        y: has to be an numpy.array, a vector of shape m * 1.
        y_hat: has to be an numpy.array, a vector of shape m * 1.
        eps: has to be a float, epsilon (default=1e-15)
    Return:
        The logistic loss value as a float.
        None otherwise.
    Raises:
        This function should not raise any Exception.
    """
    try:
        if (not isinstance(y, np.ndarray)) \
                or (not isinstance(y_hat, np.ndarray)):
            s = "y or/and y_hat are not of the expected type (numpy array)."
            print(s, file=sys.stderr)
            return None

        if (y.ndim != 2) or (y_hat.ndim != 2) \
                or (y.shape[1] != 1) \
                or (y_hat.shape[1] != 1) \
                or (y.shape[0] != y_hat.shape[0]):
            s = "x or theta not 2 dimensional array " \
                + "or mismatching shape between x and theta"
            print(s, file=sys.stderr)
            return None

        log_loss = np.dot(y.T, np.log(y_hat + eps)) \\
            + np.dot((1 - y).T, np.log(1 - y_hat + eps))
        return - float(log_loss) / y.shape[0]
    except:
        return None


if __name__ == "__main__":
    print("# Example 1:")
    y1 = np.array([[1]])
    x1 = np.array([[4]])
    theta1 = np.array([[2], [0.5]])
    y_hat1 = logistic_predict_(x1, theta1)
    res = vec_log_loss_(y1, y_hat1)
    # Output:
    expected = 0.01814992791780973
    print("my log loss =".ljust(20), res)
    print("expected log loss =".ljust(20), expected)

    print("\n# Example 2:")
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    y_hat2 = logistic_predict_(x2, theta2)
    res = vec_log_loss_(y2, y_hat2)
    # Output:
    expected = 2.4825011602474483
    print("my log loss =".ljust(20), res)
    print("expected log loss =".ljust(20), expected)

    print("\n# Example 3:")
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    y_hat3 = logistic_predict_(x3, theta3)
    res = vec_log_loss_(y3, y_hat3)
    # Output:
    expected = 2.9938533108607053
    print("my log loss =".ljust(20), res)
    print("expected log loss =".ljust(20), expected)
