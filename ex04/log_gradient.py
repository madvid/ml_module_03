import sys
import os
import numpy as np

path = os.path.join(os.path.dirname(__file__), '..', 'ex01')
sys.path.insert(1, path)
from log_pred import logistic_predict_

def log_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array,
    with a for-loop. The three arrays must have compatible shapes.
    Args:
        x: has to be an numpy.array, a matrix of shape m * n.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a vector (n +1) * 1.
    Return:
        The gradient as a numpy.array, a vector of shapes n * 1,
            containing the result of the formula for all j.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible shapes.
        None if x, y or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        if (not isinstance(x, np.ndarray)) \
                or (not isinstance(y, np.ndarray)) \
                    or (not isinstance(theta, np.ndarray)):
            s = "x or/and y or/and theta are not of the expected type" \
                + " (numpy array)."
            print(s, file=sys.stderr)
            return None

        if (y.ndim != 2) or (x.ndim != 2) or ((theta.ndim != 2))\
                or (y.shape[1] != 1) \
                or (y.shape[0] != x.shape[0]) \
                or (theta.shape != (x.shape[1] + 1, 1)):
            s = "Unexpected dimension for at least one of the arrays" \
                + " or mismatching shape between arrays"
            print(s, file=sys.stderr)
            return None
    
        grad = np.zeros(theta.shape)
        for x_ii, pred_ii, y_ii in zip(x, logistic_predict_(x, theta), y):
            grad[0] += pred_ii - y_ii
            grad[1:] += (pred_ii - y_ii) * x_ii.T
        return grad / x.shape[0]
    except:
        return None


if __name__ == "__main__":
    print("# Example 1:")
    y1 = np.array([[1]])
    x1 = np.array([[4]])
    theta1 = np.array([[2], [0.5]])
    res = log_gradient(x1, y1, theta1)
    # Output:
    expected = np.array([[-0.01798621], [-0.07194484]])
    print("my logistic gradient:".ljust(30), res.reshape(1, -1))
    print("expected log gradient:".ljust(30), expected.reshape(1, -1))

    print("\n# Example 2:")
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    res = log_gradient(x2, y2, theta2)
    # Output:
    expected = np.array([[0.3715235 ], [3.25647547]])
    print("my logistic gradient:".ljust(30), res.reshape(1, -1))
    print("expected log gradient:".ljust(30), expected.reshape(1, -1))
    
    print("\n# Example 3:")
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    res = log_gradient(x3, y3, theta3)
    # Output:
    expected = np.array([[-0.55711039],
                      [-0.90334809],
                      [-2.01756886],
                      [-2.10071291],
                      [-3.27257351]])
    print("my logistic gradient:".ljust(30), res.reshape(1, -1))
    print("expected log gradient:".ljust(30), expected.reshape(1, -1))
