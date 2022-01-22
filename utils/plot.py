import numpy as np
import matplotlib.pyplot as plt
from prediction import predict_


def plot(x, y, y_hat, title, b_legend = True,
         axes_labels = ["x (a.u.)", "y (a.u.)"],
         data_labels = {"raw":"raw", "prediction": "prediction"},
         colors = {'raw': '#101214', 'prediction': '#4287f5'}):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a vector of shape 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exception.
    """
    if isinstance(x, np.ndarray) \
        and isinstance(y, np.ndarray) \
            and isinstance(y_hat, np.ndarray):
        fig, axes = plt.subplots(1,1, figsize=(10,8))
        axes.scatter(x, y, label = data_labels['raw'], c=colors['raw'])
        axes.scatter(x, y_hat, label = data_labels['prediction'], c=colors['prediction'])
        plt.legend()
        axes.set_title(title)
        plt.xlabel(axes_labels[0])
        plt.ylabel(axes_labels[1])
        if b_legend:
            plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    x = np.arange(1,6).reshape(-1, 1)
    y = np.array([[3.74013816],[3.61473236],[4.57655287],[4.66793434],[5.95585554]])
    
    # Example 1:
    theta1 = np.array([[4.5],[-0.2]])
    plot(x, y, theta1)
    
    # Example 2:
    theta2 = np.array([[-1.5],[2]])
    plot(x, y, theta2)
    
    # Example 3:
    theta3 = np.array([[3],[0.3]])
    plot(x, y, theta3)