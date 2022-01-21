import sys
import os
import numpy as np

path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path)
from prediction import predict_


def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array,
    without any for-loop. The three arrays must have the compatible shapes.
    Args:
        x: has to be an numpy.array, a matrix of shape m * n.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a vector (n +1) * 1.
    Return:
        The gradient as a numpy.array, a vector of shapes n * 1,
          containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible shapes.
        None if x, y or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        # Testing the type of the parameters, numpy array expected.
        if (not isinstance(x, np.ndarray)) \
            or (not isinstance(y, np.ndarray)) \
                or (not isinstance(theta, np.ndarray)):
            return None

        # Testing the shape of the paramters.
        if (y.shape[1] != 1) \
            or (theta.shape[1] != 1) \
                or (x.shape[0] != y.shape[0]) \
                or ((x.shape[1] + 1) != theta.shape[0]):
            return None

        m, grad = x.shape[0], np.zeros(theta.shape)
        xp = np.hstack((np.ones((m, 1)), x))
        grad = xp.T @ (xp @ theta - y)

        return grad / m
    except:
        return None


if __name__ == "__main__":
    x = np.array([
        [ -6, -7, -9],
        [ 13, -2, 14],
        [ -7, 14, -1],
        [ -8, -4, 6],
        [ -5, -9, 6],
        [ 1, -5, 11],
        [ 9, -11, 8]])
    y = np.array([[2],[ 14],[ -13],[ 5],[ 12],[ 4],[ -19]])

    print("# Example 0:")
    theta1 = np.array([[0],[ 3],[ 0.5],[ -6]])
    grad = gradient(x, y, theta1)
    # Output:
    expected_grad = np.array([[ -33.71428571],[ -37.35714286],[ 183.14285714],[ -393.]])
    print("my gradient: ", grad)
    print("expected gradient: ", expected_grad)
    
    print("\n# Example 1:")
    theta2 = np.array([[0],[ 0],[ 0],[ 0]])
    grad = gradient(x, y, theta2)
    # Output:
    expected_grad = np.array([[ -0.71428571],[ 0.85714286],[ 23.28571429],[ -26.42857143]])
    print("my gradient: ", grad)
    print("expected gradient: ", expected_grad)