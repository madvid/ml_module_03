import sys
import os
import numpy as np

## Collecting the path where gradient method is
path = os.path.join(os.path.dirname(__file__), '..', 'ex03')
sys.path.insert(1, path)
from gradient import gradient


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.array, a vector of shape m * 1:
           (number of training examples, 1).
        y: has to be a numpy.array, a vector of shape m * 1:
           (number of training examples, 1).
        theta: has to be a numpy.array, a vector of shape 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done
                  during the gradient descent
    Return:
        new_theta: numpy.array, a vector of shape 2 * 1.
        None if there is a matching shape problem.
        None if x, y, theta, alpha or max_iter is not of the expected
             type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        ## Checking x, y and theta are numpy array
        if (not isinstance(x, np.ndarray)) \
            or (not isinstance(y, np.ndarray)) \
                or (not isinstance(theta, np.ndarray)):
            return None
        ## Checking the shape of x, y and theta
        if (x.shape[1] != 1) \
            or (y.shape[1] != 1) \
                or (x.shape[0] != y.shape[0]) \
                or (theta.shape[0] != x.shape[1] + 1):
            return None
        ## Checking the type and values of max_iter and alpha
        if (not isinstance(max_iter, int)) \
            or (max_iter < 0) \
                or (not isinstance(alpha, float)) \
                or (alpha >= 1 or alpha <= 0):
            return None
        ## Casting theta to float, in case it is integer
        new_theta = np.copy(theta.astype('float64'))
        for _ in range(max_iter):
            grad = gradient(x, y, new_theta)
            new_theta[0] = new_theta[0] - alpha * grad[0]
            new_theta[1] = new_theta[1] - alpha * grad[1]
        return new_theta
    except:
        ## If something unexpected happened, we juste leave
        return None


if __name__ == "__main__":
    x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
    y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta = np.array([[42.], [1.], [1.], [1.]])
    # Example 0:
    theta2 = fit_(X2, Y2, theta2, alpha = 0.0005, max_iter=42000)
    theta2
    # Output:
    expected_theta = np.array([[41.99..],[0.97..], [0.77..], [-1.20..]])
    # Example 1:
    predict_(X2, theta2)
    # Output:
    expected_theta = np.array([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]]