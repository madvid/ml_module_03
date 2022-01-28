import numpy as np


class MyLogisticRegression():
    """
    Description:
    My personnal logistic regression to classify things.
    """

    def __init__(self, theta, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta
        ... Your code ...
        ... other methods ...


if __name__ == "__main__":
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])
    
    mylr = MyLogisticRegression([2, 0.5, 7.1, -4.3, 2.09])
    
    print("# Example 0:")
    my_res = mylr.predict_(X)
    # Output:
    expected = np.array([[0.99930437], [1.], [1.]])
    print("my prediction:".ljust(25), my_res.reshape(1, -1))
    print("expected prediction:".ljust(25), my_res.reshape(1, -1))
    
    print("# Example 1:")
    my_res = mylr.loss_(X,Y)
    # Output:
    expected = 11.513157421577004
    print("my loss:".ljust(25), my_res.reshape(1, -1))
    print("expected loss:".ljust(25), my_res.reshape(1, -1))
    
    print("# Example 2:")
    mylr.fit_(X, Y)
    my_res = mylr.theta
    # Output:
    expected = np.array([[ 1.04565272],
                         [0.62555148],
                         [0.38387466],
                         [0.15622435],
                         [-0.45990099]])
    print("my theta after fit:".ljust(25), my_res.reshape(1, -1))
    print("expected theta after fit:".ljust(25), my_res.reshape(1, -1))

    print("# Example 3:")
    my_res = mylr.predict_(X)
    # Output:
    expected = np.array([[0.72865802],
                         [0.40550072],
                         [0.45241588]])
    print("my prediction:".ljust(25), my_res.reshape(1, -1))
    print("expected prediction:".ljust(25), my_res.reshape(1, -1))

    print("# Example 4:")
    my_res = mylr.loss_(X,Y)
    # Output:
    expected = 0.5432466580663214
    print("my loss:".ljust(25), my_res.reshape(1, -1))
    print("expected loss:".ljust(25), my_res.reshape(1, -1))